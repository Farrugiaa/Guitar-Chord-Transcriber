"""
Training script for the Guitar Separator U-Net.

Trains on mixed audio / isolated guitar pairs.
Can use MUSDB18 or custom datasets with separated stems.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
import yaml
from tqdm import tqdm

from src.models.separator import GuitarSeparatorUNet


class SeparationDataset(Dataset):
    """
    Dataset for source separation training.

    Expects pairs of (mix, guitar_stem) audio files.

    Directory structure:
        data_root/
            train/
                mix/
                    001.wav
                guitar/
                    001.wav
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        sample_rate: int = 44100,
        segment_length: int = 44100 * 5,  # 5 seconds
        n_fft: int = 4096,
        hop_length: int = 1024,
    ):
        self.root_dir = Path(root_dir) / split
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.mix_dir = self.root_dir / "mix"
        self.guitar_dir = self.root_dir / "guitar"

        self.files = sorted(self.mix_dir.glob("*.wav")) if self.mix_dir.exists() else []

        self.stft = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=2.0
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mix_path = self.files[idx]
        guitar_path = self.guitar_dir / mix_path.name

        mix_wav, sr = torchaudio.load(str(mix_path))
        guitar_wav, _ = torchaudio.load(str(guitar_path))

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            mix_wav = resampler(mix_wav)
            guitar_wav = resampler(guitar_wav)

        # Mono
        mix_wav = mix_wav.mean(dim=0, keepdim=True)
        guitar_wav = guitar_wav.mean(dim=0, keepdim=True)

        # Random crop to segment_length
        if mix_wav.shape[1] > self.segment_length:
            start = torch.randint(0, mix_wav.shape[1] - self.segment_length, (1,)).item()
            mix_wav = mix_wav[:, start : start + self.segment_length]
            guitar_wav = guitar_wav[:, start : start + self.segment_length]
        else:
            # Pad if shorter
            pad = self.segment_length - mix_wav.shape[1]
            mix_wav = nn.functional.pad(mix_wav, (0, pad))
            guitar_wav = nn.functional.pad(guitar_wav, (0, pad))

        # Compute spectrograms
        mix_spec = self.stft(mix_wav)    # (1, F, T)
        guitar_spec = self.stft(guitar_wav)

        return mix_spec, guitar_spec


def train(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    sep_config = config["separator"]
    train_config = config["training"]

    # Model
    model = GuitarSeparatorUNet(
        channels=sep_config["channels"],
        dropout=sep_config["dropout"],
    ).to(device)

    # Dataset
    train_dataset = SeparationDataset(
        root_dir=args.data_dir,
        split="train",
        n_fft=sep_config["n_fft"],
        hop_length=sep_config["hop_length"],
    )
    val_dataset = SeparationDataset(
        root_dir=args.data_dir,
        split="val",
        n_fft=sep_config["n_fft"],
        hop_length=sep_config["hop_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        num_workers=4,
        pin_memory=True,
    )

    # Loss and optimiser
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config["epochs"]
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(train_config["epochs"]):
        # Train
        model.train()
        train_loss = 0.0

        for mix_spec, guitar_spec in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}"
        ):
            mix_spec = mix_spec.to(device)
            guitar_spec = guitar_spec.to(device)

            optimizer.zero_grad()
            estimated = model.separate(mix_spec)
            loss = criterion(estimated, guitar_spec)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mix_spec, guitar_spec in val_loader:
                mix_spec = mix_spec.to(device)
                guitar_spec = guitar_spec.to(device)
                estimated = model.separate(mix_spec)
                loss = criterion(estimated, guitar_spec)
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                Path(args.checkpoint_dir) / "separator_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= train_config["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Guitar Separator")
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to separation training data"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory to save checkpoints"
    )
    args = parser.parse_args()

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train(args)
