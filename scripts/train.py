#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fnf_ai.dataset import FNFStemDataset, build_tag_vocab, load_dataset_manifest
from fnf_ai.losses import weighted_cvae_loss
from fnf_ai.model import StemCVAE
from fnf_ai.utils import load_json, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Train the FNF v0.1 stem-aware conditional VAE.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default="configs/v0_1.json")
    p.add_argument("--output", default="checkpoints/v0_1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume")
    return p.parse_args()


def split_songs(song_ids: list[str], validation_count: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(song_ids), generator=g).tolist()
    val_n = max(1, min(validation_count, len(song_ids) - 1))
    val = [song_ids[i] for i in order[:val_n]]
    train = [song_ids[i] for i in order[val_n:]]
    return train, val


def save_checkpoint(path: Path, model, optimizer, epoch: int, config: dict, tag_vocab: list[str], val_loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "tag_vocab": tag_vocab,
        "val_loss": val_loss,
    }, path)


def evaluate(model, loader, device, config):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["specs"].to(device)
            tags = batch["tags"].to(device)
            weights = batch["weight"].to(device)
            recon, mu, logvar = model(x, tags)
            loss, _ = weighted_cvae_loss(
                recon, x, mu, logvar, weights,
                config["beta_kl"], config["lambda_consistency"],
            )
            total += float(loss) * x.shape[0]
            count += x.shape[0]
    return total / max(count, 1)


def main():
    args = parse_args()
    config = load_json(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device)

    manifest = load_dataset_manifest(args.dataset)
    song_ids = [s["id"] for s in manifest["songs"]]
    train_ids, val_ids = split_songs(song_ids, int(config["validation_song_count"]), int(config["seed"]))
    tag_vocab = build_tag_vocab(manifest)

    print("train songs:", ", ".join(train_ids))
    print("validation songs:", ", ".join(val_ids))
    print("tag vocabulary:", ", ".join(tag_vocab))

    train_ds = FNFStemDataset(args.dataset, train_ids, config, tag_vocab, training=True)
    val_ds = FNFStemDataset(args.dataset, val_ids, config, tag_vocab, training=False)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    model = StemCVAE(
        n_mels=config["n_mels"], spec_frames=config["spec_frames"], latent_dim=config["latent_dim"],
        base_channels=config["base_channels"], num_tags=len(tag_vocab),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.99))
    start_epoch = 1
    best = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best = float(ckpt.get("val_loss", best))
        if ckpt.get("tag_vocab") != tag_vocab:
            raise RuntimeError("Tag vocabulary changed; cannot safely resume this checkpoint")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "split.json", {"train": train_ids, "validation": val_ids, "tag_vocab": tag_vocab})

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        model.train()
        bar = tqdm(train_loader, desc=f"epoch {epoch:03d}")
        running = 0.0
        for step, batch in enumerate(bar, 1):
            x = batch["specs"].to(device)
            tags = batch["tags"].to(device)
            weights = batch["weight"].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x, tags)
            loss, metrics = weighted_cvae_loss(
                recon, x, mu, logvar, weights,
                config["beta_kl"], config["lambda_consistency"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss)
            bar.set_postfix(loss=f"{running/step:.4f}", rec=f"{metrics['reconstruction']:.4f}")

        val_loss = evaluate(model, val_loader, device, config)
        print(f"epoch {epoch}: val_loss={val_loss:.6f}")
        save_checkpoint(out / "last.pt", model, optimizer, epoch, config, tag_vocab, val_loss)
        if epoch % int(config["checkpoint_every"]) == 0:
            save_checkpoint(out / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, config, tag_vocab, val_loss)
        if val_loss < best:
            best = val_loss
            save_checkpoint(out / "best.pt", model, optimizer, epoch, config, tag_vocab, val_loss)
            print("  new best checkpoint")


if __name__ == "__main__":
    main()
