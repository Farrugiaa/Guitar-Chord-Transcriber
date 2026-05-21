"""
Chord recognition model using a CRNN (Convolutional Recurrent Neural Network).

Takes CQT spectrogram frames as input and outputs chord class probabilities
per frame. Designed to be pre-trained on piano data (MAESTRO/MAPS) and
fine-tuned on guitar data (DadaGP + separated audio).
"""

import torch
import torch.nn as nn


class ChordCRNN(nn.Module):
    """
    CRNN for frame-level chord recognition.

    Architecture:
        1. CNN layers extract local harmonic features from CQT
        2. Bidirectional GRU captures temporal context
        3. FC layer outputs chord class probabilities

    The chord vocabulary can be configured from basic (maj/min = 25 classes)
    to extended (including 7ths, 9ths, etc.).
    """

    def __init__(
        self,
        n_cqt_bins: int = 84,
        conv_channels: list[int] | None = None,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        n_classes: int = 170,
        dropout: float = 0.3,
        context_frames: int = 21,
    ):
        """
        Args:
            n_cqt_bins: Number of CQT frequency bins.
            conv_channels: Channel sizes for conv layers.
            rnn_hidden: Hidden size for GRU.
            rnn_layers: Number of GRU layers.
            n_classes: Number of chord classes in vocabulary.
            dropout: Dropout rate.
            context_frames: Number of context frames per input window.
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.context_frames = context_frames

        # CNN feature extractor
        cnn_layers = []
        in_ch = 1
        freq_dim = n_cqt_bins

        for out_ch in conv_channels:
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),  # pool only in frequency
                nn.Dropout2d(dropout),
            ])
            in_ch = out_ch
            freq_dim = freq_dim // 2

        self.cnn = nn.Sequential(*cnn_layers)

        # Compute flattened feature size after CNN
        self.cnn_output_size = conv_channels[-1] * freq_dim

        # RNN for temporal modelling
        self.rnn = nn.GRU(
            input_size=self.cnn_output_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CQT spectrogram (B, 1, F, T) where F=n_cqt_bins, T=time frames

        Returns:
            Chord logits (B, T, n_classes) per frame
        """
        batch_size = x.shape[0]
        time_steps = x.shape[3]

        # CNN: extract features per frame
        # (B, 1, F, T) -> (B, C, F', T)
        cnn_out = self.cnn(x)

        # Reshape: (B, C, F', T) -> (B, T, C*F')
        cnn_out = cnn_out.permute(0, 3, 1, 2)
        cnn_out = cnn_out.reshape(batch_size, time_steps, -1)

        # RNN: temporal context
        rnn_out, _ = self.rnn(cnn_out)

        # Classify each frame
        logits = self.classifier(rnn_out)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted chord indices per frame."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


class ChordVocabulary:
    """
    Manages the mapping between chord labels and class indices.

    Supports three vocabulary levels:
        - basic: 12 major + 12 minor + N.C. = 25 classes
        - sevenths: adds dom7, maj7, min7, dim7 = 73 classes
        - extended: adds 9ths, 11ths, 13ths, sus, aug, dim = 170 classes
    """

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    QUALITY_SETS = {
        "basic": ["", "m"],
        "sevenths": ["", "m", "7", "maj7", "m7", "dim", "m7b5", "dim7", "aug"],
        "extended": [
            "", "m", "7", "maj7", "m7", "dim", "m7b5", "dim7", "aug",
            "aug7", "sus2", "sus4", "7sus4", "9", "m9", "maj9",
            "11", "m11", "13", "m13", "maj13",
            "add9", "6", "m6", "5",
        ],
    }

    def __init__(self, level: str = "extended"):
        if level not in self.QUALITY_SETS:
            raise ValueError(f"Unknown vocabulary level: {level}")

        self.level = level
        qualities = self.QUALITY_SETS[level]

        self.label_to_idx = {"N.C.": 0}
        self.idx_to_label = {0: "N.C."}

        idx = 1
        for note in self.NOTE_NAMES:
            for quality in qualities:
                label = f"{note}{quality}"
                self.label_to_idx[label] = idx
                self.idx_to_label[idx] = label
                idx += 1

        self.n_classes = len(self.label_to_idx)

    def encode(self, chord_label: str) -> int:
        """Convert chord label to class index. Falls back to N.C."""
        return self.label_to_idx.get(chord_label, 0)

    def decode(self, idx: int) -> str:
        """Convert class index to chord label."""
        return self.idx_to_label.get(idx, "N.C.")

    def __len__(self) -> int:
        return self.n_classes
