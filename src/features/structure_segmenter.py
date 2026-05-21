"""
Structural segmentation: detect song sections (intro, verse, chorus, etc.)

Uses librosa's self-similarity (recurrence) matrix and novelty-based
boundary detection on MFCCs and chroma features, then clusters segments
by timbral/harmonic similarity to assign section labels.
"""

from dataclasses import dataclass

import librosa
import numpy as np
import torch
from scipy.ndimage import median_filter
from sklearn.cluster import AgglomerativeClustering


@dataclass
class SongSection:
    """A detected section of the song."""
    label: str          # e.g. "Intro", "Verse", "Chorus"
    start_time: float   # seconds
    end_time: float     # seconds
    cluster_id: int     # internal cluster index


class StructureSegmenter:
    """Detect structural sections in audio using self-similarity."""

    # Label assignment order: first section -> Intro, most repeated -> Chorus,
    # second most -> Verse, remaining -> Bridge/Interlude/Outro
    SECTION_NAMES = [
        "Verse", "Chorus", "Bridge", "Interlude",
        "Pre-Chorus", "Post-Chorus", "Breakdown",
    ]

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        min_section_duration: float = 4.0,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_section_duration = min_section_duration

    def segment(self, waveform: torch.Tensor) -> list[SongSection]:
        """
        Detect structural sections from audio.

        Args:
            waveform: Audio tensor (1, N)

        Returns:
            List of SongSection with labels and timing.
        """
        audio = waveform.squeeze().numpy()

        # 1. Detect section boundaries using novelty on the recurrence matrix
        boundaries = self._detect_boundaries(audio)

        # 2. Build feature vectors per segment for clustering
        segment_features, seg_energies = self._compute_segment_features(audio, boundaries)

        # 3. Cluster similar segments together
        cluster_ids = self._cluster_segments(segment_features)

        # 4. Assign musical labels based on position, repetition, and energy
        sections = self._assign_labels(boundaries, cluster_ids, seg_energies)

        return sections

    def _detect_boundaries(self, audio: np.ndarray) -> np.ndarray:
        """Detect section boundaries using spectral novelty."""
        # Compute features for self-similarity
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate,
            hop_length=self.hop_length, n_mfcc=13,
        )
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        # Stack and normalise features
        features = np.vstack([
            librosa.util.normalize(mfcc, axis=1),
            librosa.util.normalize(chroma, axis=1),
        ])

        # Build recurrence matrix (self-similarity)
        rec = librosa.segment.recurrence_matrix(
            features, width=3, mode="affinity", sym=True,
        )

        # Compute novelty curve from the recurrence matrix using a
        # checkerboard kernel — peaks indicate section boundaries
        novelty = self._checkerboard_novelty(rec, kernel_size=64)

        # Smooth the novelty curve
        novelty = median_filter(novelty, size=9)
        novelty = novelty / (novelty.max() + 1e-8)

        # Pick peaks above a threshold
        min_frames = int(
            self.min_section_duration * self.sample_rate / self.hop_length
        )
        peaks = librosa.util.peak_pick(
            novelty,
            pre_max=min_frames // 2,
            post_max=min_frames // 2,
            pre_avg=min_frames,
            post_avg=min_frames,
            delta=0.1,
            wait=min_frames,
        )

        # Convert frames to times, always include start and end
        total_duration = len(audio) / self.sample_rate
        boundary_times = librosa.frames_to_time(
            peaks, sr=self.sample_rate, hop_length=self.hop_length,
        )
        boundary_times = np.concatenate([[0.0], boundary_times, [total_duration]])
        boundary_times = np.unique(boundary_times)

        # Merge sections that are too short
        merged = [boundary_times[0]]
        for t in boundary_times[1:]:
            if t - merged[-1] >= self.min_section_duration:
                merged.append(t)
            # If it's the last boundary (end of song), always keep it
            elif t == boundary_times[-1]:
                if len(merged) > 1:
                    merged[-1] = t
                else:
                    merged.append(t)
        boundary_times = np.array(merged)

        return boundary_times

    @staticmethod
    def _checkerboard_novelty(rec: np.ndarray, kernel_size: int = 64) -> np.ndarray:
        """Compute a novelty curve from a recurrence matrix using a checkerboard kernel."""
        n = rec.shape[0]
        half = kernel_size // 2
        novelty = np.zeros(n)

        # Build checkerboard kernel
        kern = np.ones((kernel_size, kernel_size))
        kern[:half, :half] = -1
        kern[half:, half:] = -1

        for i in range(half, n - half):
            patch = rec[i - half : i + half, i - half : i + half]
            if patch.shape == kern.shape:
                novelty[i] = np.sum(patch * kern)

        # Clip negative values and normalise
        novelty = np.maximum(novelty, 0)
        return novelty

    def _compute_segment_features(
        self, audio: np.ndarray, boundaries: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a feature vector and mean RMS energy per segment for clustering.

        Returns:
            features:      (n_segments, D) array for clustering
            seg_energies:  (n_segments,) mean RMS energy per segment (for label heuristic)

        Feature dimensions:
          - MFCC mean (13): timbral identity — distinguishes sections by instrument texture
          - MFCC std (13): timbral activity — higher in busy sections (chorus)
          - Chroma mean (12): harmonic identity — segments with the same chord
            progression cluster together regardless of dynamics
          - log RMS energy (1): choruses are almost always louder than verses;
            this is the single most reliable acoustic cue for verse/chorus distinction
          - Spectral centroid mean (1): choruses tend to be brighter (more strumming,
            higher-frequency string attack) — complements MFCC texture
          - Spectral flux mean (1): mean frame-to-frame spectral change; choruses have
            higher flux due to strumming density and dynamic variation
        """
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate,
            hop_length=self.hop_length, n_mfcc=13,
        )
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        rms = librosa.feature.rms(
            y=audio, hop_length=self.hop_length,
        )[0]   # shape (T,)
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length,
        )[0]   # shape (T,)

        # Spectral flux: L1 norm of frame-to-frame difference in magnitude spectrum
        S = np.abs(librosa.stft(audio, hop_length=self.hop_length))
        flux = np.concatenate([[0.0], np.sum(np.maximum(np.diff(S, axis=1), 0), axis=0)])

        segment_features = []
        seg_energies = []

        for i in range(len(boundaries) - 1):
            sf = librosa.time_to_frames(
                boundaries[i], sr=self.sample_rate, hop_length=self.hop_length,
            )
            ef = librosa.time_to_frames(
                boundaries[i + 1], sr=self.sample_rate, hop_length=self.hop_length,
            )
            ef = min(ef, mfcc.shape[1])
            if sf >= ef:
                sf = max(0, ef - 1)

            seg_mfcc      = mfcc[:, sf:ef].mean(axis=1)
            seg_mfcc_std  = mfcc[:, sf:ef].std(axis=1)
            seg_chroma    = chroma[:, sf:ef].mean(axis=1)

            seg_rms       = float(rms[sf:ef].mean()) if ef > sf else 0.0
            seg_centroid  = float(centroid[sf:ef].mean()) if ef > sf else 0.0
            seg_flux      = float(flux[sf:ef].mean()) if ef > sf else 0.0

            # Log-compress RMS so that loud sections don't dominate distance
            log_rms = float(np.log1p(seg_rms * 100))
            # Normalise centroid to [0, 1] range (max ~8000 Hz for guitar)
            norm_centroid = min(seg_centroid / 8000.0, 1.0)
            # Normalise flux similarly
            norm_flux = min(seg_flux / (flux.max() + 1e-8), 1.0)

            feat = np.concatenate([
                seg_mfcc,
                seg_mfcc_std,
                seg_chroma,
                [log_rms, norm_centroid, norm_flux],
            ])
            segment_features.append(feat)
            seg_energies.append(seg_rms)

        return np.array(segment_features), np.array(seg_energies)

    def _cluster_segments(self, features: np.ndarray) -> list[int]:
        """Cluster segments by timbral/harmonic similarity."""
        n_segments = len(features)

        if n_segments <= 1:
            return list(range(n_segments))

        if n_segments == 2:
            return [0, 1]

        # Determine number of clusters (heuristic: sqrt of segments, 2-6 range)
        n_clusters = max(2, min(6, int(np.sqrt(n_segments) + 0.5)))
        n_clusters = min(n_clusters, n_segments)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean",
            linkage="ward",
        )
        labels = clustering.fit_predict(features)
        return labels.tolist()

    def _assign_labels(
        self,
        boundaries: np.ndarray,
        cluster_ids: list[int],
        seg_energies: np.ndarray,
    ) -> list[SongSection]:
        """
        Assign musical section names based on cluster frequency, position, and energy.

        Strategy:
          1. Exclude the first and last sections from frequency counting (likely
             Intro / Outro).
          2. Among the remaining clusters, rank by occurrence frequency.
          3. Use mean RMS energy per cluster to break the Verse / Chorus ambiguity:
             the high-energy cluster = Chorus, the low-energy cluster = Verse.
             Energy is the most reliable single acoustic cue — choruses are almost
             always louder than verses in guitar / pop / rock recordings.
          4. Assign Bridge, Pre-Chorus, etc. to the remaining clusters by frequency.
          5. Override the first section to Intro if it is short (< 15 s) or its
             cluster does not repeat later.
          6. Override the last section to Outro if its cluster does not appear earlier.
        """
        from collections import Counter, defaultdict

        n_sections = len(cluster_ids)
        if n_sections == 0:
            return []

        # Compute mean RMS energy per cluster across all segments
        cluster_energy: dict[int, list[float]] = defaultdict(list)
        for i, cid in enumerate(cluster_ids):
            if i < len(seg_energies):
                cluster_energy[cid].append(float(seg_energies[i]))
        cluster_mean_energy: dict[int, float] = {
            cid: float(np.mean(vals)) for cid, vals in cluster_energy.items()
        }

        # Count occurrences, skipping first and last (likely intro/outro)
        inner_ids = cluster_ids[1:-1] if n_sections > 2 else cluster_ids[:]
        freq = Counter(inner_ids)
        ranked = [cid for cid, _ in freq.most_common()]

        cluster_label: dict[int, str] = {}

        if len(ranked) >= 2:
            # Among the two most-frequent clusters, the louder one is the Chorus
            # and the quieter one is the Verse.  This is more reliable than
            # pure-frequency ranking because many songs have 3 verses but only
            # 2 choruses — raw frequency would wrongly label the verse as chorus.
            top_two = ranked[:2]
            energies = [cluster_mean_energy.get(c, 0.0) for c in top_two]
            if energies[0] >= energies[1]:
                chorus_cid, verse_cid = top_two[0], top_two[1]
            else:
                chorus_cid, verse_cid = top_two[1], top_two[0]
            cluster_label[chorus_cid] = "Chorus"
            cluster_label[verse_cid]  = "Verse"

            # Remaining clusters by frequency → Bridge, Pre-Chorus, …
            other_names = ["Bridge", "Pre-Chorus", "Interlude", "Post-Chorus", "Breakdown"]
            name_idx = 0
            for cid in ranked[2:]:
                cluster_label[cid] = other_names[name_idx % len(other_names)]
                name_idx += 1

        elif len(ranked) == 1:
            cluster_label[ranked[0]] = "Verse"

        # Clusters that only appear in intro/outro slots get Interlude
        for cid in cluster_ids:
            if cid not in cluster_label:
                cluster_label[cid] = "Interlude"

        # Build sections
        sections: list[SongSection] = []
        for i in range(n_sections):
            start = float(boundaries[i])
            end   = float(boundaries[i + 1])
            cid   = cluster_ids[i]
            label = cluster_label.get(cid, "Interlude")

            # Override: first section → Intro if short or non-repeating
            if i == 0:
                appears_later = cluster_ids.count(cid) > 1
                if (end - start) < 15.0 or not appears_later:
                    label = "Intro"

            # Override: last section → Outro if its cluster never appeared before
            if i == n_sections - 1:
                appears_earlier = cluster_ids[:i].count(cid) > 0
                if not appears_earlier:
                    label = "Outro"

            sections.append(SongSection(
                label=label,
                start_time=start,
                end_time=end,
                cluster_id=cid,
            ))

        return sections
