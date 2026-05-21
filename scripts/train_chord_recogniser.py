"""
Training script for the Chord Recognition CRNN.

Phase 1: Pre-train on piano data (MAESTRO / MAPS)
Phase 2: Fine-tune on guitar data (DadaGP + separated audio)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from src.models.chord_recogniser import ChordCRNN, ChordVocabulary
from src.features.audio_features import AudioFeatureExtractor
from src.data.maestro_dataset import MAESTRODataset
from src.data.maps_dataset import MAPSDataset


def train_phase(
    model: ChordCRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    checkpoint_path: Path,
    phase_name: str = "pretrain",
):
    """Generic training loop for one phase."""
    train_config = config["training"]

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
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
        model.train()
        train_loss = 0.0
        n_correct = 0
        n_total = 0

        for batch in tqdm(train_loader, desc=f"[{phase_name}] Epoch {epoch + 1}"):
            cqt = batch["cqt"].to(device)
            labels = batch["chord_labels"].to(device)

            optimizer.zero_grad()
            logits = model(cqt)  # (B, T, n_classes)

            # Flatten for cross-entropy
            B, T, C = logits.shape
            loss = criterion(logits.reshape(B * T, C), labels.reshape(B * T))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = labels != -1
            n_correct += (preds[mask] == labels[mask]).sum().item()
            n_total += mask.sum().item()

        train_loss /= max(len(train_loader), 1)
        train_acc = n_correct / max(n_total, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                cqt = batch["cqt"].to(device)
                labels = batch["chord_labels"].to(device)

                logits = model(cqt)
                B, T, C = logits.shape
                loss = criterion(logits.reshape(B * T, C), labels.reshape(B * T))
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                mask = labels != -1
                val_correct += (preds[mask] == labels[mask]).sum().item()
                val_total += mask.sum().item()

        val_loss /= max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step()

        print(
            f"[{phase_name}] Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= train_config["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return best_val_loss


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    vocab_level = config.get("chord_vocabulary", "extended")
    vocabulary = ChordVocabulary(level=vocab_level)
    print(f"Chord vocabulary: {vocab_level} ({len(vocabulary)} classes)")

    chord_config = config["chord_recogniser"]
    model = ChordCRNN(
        n_cqt_bins=chord_config["n_cqt_bins"],
        conv_channels=chord_config["conv_channels"],
        rnn_hidden=chord_config["rnn_hidden"],
        rnn_layers=chord_config["rnn_layers"],
        n_classes=len(vocabulary),
        dropout=chord_config["dropout"],
    ).to(device)

    feature_extractor = AudioFeatureExtractor(
        sample_rate=config["audio"]["sample_rate"],
        hop_length=config["features"]["hop_length"],
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Pre-train on piano data
    if args.maestro_dir or args.maps_dir:
        print("\n=== Phase 1: Pre-training on piano data ===")

        train_datasets = []
        val_datasets = []

        if args.maestro_dir:
            train_datasets.append(
                MAESTRODataset(
                    args.maestro_dir, split="train",
                    feature_extractor=feature_extractor,
                    chord_vocabulary=vocabulary,
                )
            )
            val_datasets.append(
                MAESTRODataset(
                    args.maestro_dir, split="validation",
                    feature_extractor=feature_extractor,
                    chord_vocabulary=vocabulary,
                )
            )

        if args.maps_dir:
            maps = MAPSDataset(
                args.maps_dir,
                feature_extractor=feature_extractor,
                chord_vocabulary=vocabulary,
            )
            # Split MAPS 80/20
            n_train = int(0.8 * len(maps))
            n_val = len(maps) - n_train
            train_maps, val_maps = torch.utils.data.random_split(
                maps, [n_train, n_val]
            )
            train_datasets.append(train_maps)
            val_datasets.append(val_maps)

        if train_datasets:
            train_data = torch.utils.data.ConcatDataset(train_datasets)
            val_data = torch.utils.data.ConcatDataset(val_datasets)

            train_loader = DataLoader(
                train_data,
                batch_size=config["training"]["batch_size"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_data,
                batch_size=config["training"]["batch_size"],
                num_workers=4,
                pin_memory=True,
            )

            train_phase(
                model, train_loader, val_loader, config, device,
                checkpoint_dir / "chord_pretrained_piano.pt",
                phase_name="piano-pretrain",
            )

    # Phase 2: Fine-tune on guitar data
    if args.dadagp_dir:
        print("\n=== Phase 2: Fine-tuning on guitar data ===")

        # Load pre-trained weights if available
        pretrained = checkpoint_dir / "chord_pretrained_piano.pt"
        if pretrained.exists():
            model.load_state_dict(
                torch.load(pretrained, map_location=device, weights_only=True)
            )
            print("Loaded pre-trained piano weights")

        # Lower learning rate for fine-tuning
        config["training"]["learning_rate"] *= 0.1

        from src.data.dadagp_dataset import DadaGPDataset
        dadagp = DadaGPDataset(
            args.dadagp_dir,
            chord_vocabulary=vocabulary,
        )
        n_train = int(0.8 * len(dadagp))
        n_val = len(dadagp) - n_train
        train_dadagp, val_dadagp = torch.utils.data.random_split(
            dadagp, [n_train, n_val]
        )

        train_loader = DataLoader(
            train_dadagp,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dadagp,
            batch_size=config["training"]["batch_size"],
            num_workers=4,
        )

        train_phase(
            model, train_loader, val_loader, config, device,
            checkpoint_dir / "chord_finetuned_guitar.pt",
            phase_name="guitar-finetune",
        )

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chord Recogniser")
    parser.add_argument("--maestro-dir", type=str, help="Path to MAESTRO dataset")
    parser.add_argument("--maps-dir", type=str, help="Path to MAPS dataset")
    parser.add_argument("--dadagp-dir", type=str, help="Path to DadaGP dataset")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints"
    )
    args = parser.parse_args()
    main(args)
