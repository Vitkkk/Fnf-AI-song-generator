#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import torch

from fnf_ai.audio import decode_with_ffmpeg, ffprobe_duration, load_audio, save_audio, sum_stems


def parse_args():
    p = argparse.ArgumentParser(description="Merge FNF dataset ZIP parts and normalize stems for training.")
    p.add_argument("zips", nargs="+", help="Dataset ZIP parts")
    p.add_argument("--output", required=True, help="Prepared dataset directory")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--styles", help="Optional JSON mapping song IDs to style-tag arrays")
    return p.parse_args()


def load_styles(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): [str(x).lower() for x in v] for k, v in raw.items()}


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    styles = load_styles(args.styles)

    with tempfile.TemporaryDirectory(prefix="fnf_ai_dataset_") as td:
        merged = Path(td) / "merged"
        merged.mkdir()
        for zip_path in args.zips:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(merged)

        manifests = list(merged.rglob("manifest.json"))
        if not manifests:
            raise FileNotFoundError("No manifest.json found in dataset ZIPs")
        with open(manifests[0], "r", encoding="utf-8") as f:
            manifest = json.load(f)

        logical_root = manifests[0].parent
        prepared_songs = []

        for item in manifest:
            song_id = item["id"]
            src_dir = logical_root / "songs" / song_id
            if not src_dir.exists():
                matches = [p for p in merged.rglob(song_id) if p.is_dir()]
                if not matches:
                    raise FileNotFoundError(f"Missing song directory: {song_id}")
                src_dir = matches[0]

            files = item["files"]
            inst_src = src_dir / files["instrumental"]
            voc_src = src_dir / files["vocals"]
            exact_src = src_dir / files["mix_exact"] if files.get("mix_exact") else None
            normal_mix_src = src_dir / files["mix"] if files.get("mix") else None

            inst_dur = ffprobe_duration(inst_src)
            voc_dur = ffprobe_duration(voc_src)
            stem_duration = max(inst_dur, voc_dur)
            target_duration = stem_duration
            mix_source_kind = "synthesized"

            if exact_src and exact_src.exists():
                target_duration = max(target_duration, ffprobe_duration(exact_src))
                mix_source_kind = "mix_exact"
            elif normal_mix_src and normal_mix_src.exists():
                mix_dur = ffprobe_duration(normal_mix_src)
                if abs(mix_dur - stem_duration) <= 1.0:
                    target_duration = max(target_duration, mix_dur)
                    mix_source_kind = "mix"

            dst_dir = out / "songs" / song_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            inst_dst = dst_dir / "instrumental.wav"
            voc_dst = dst_dir / "vocals.wav"
            mix_dst = dst_dir / "mix.wav"

            decode_with_ffmpeg(inst_src, inst_dst, args.sample_rate, args.channels, target_duration)
            decode_with_ffmpeg(voc_src, voc_dst, args.sample_rate, args.channels, target_duration)

            if mix_source_kind == "mix_exact":
                decode_with_ffmpeg(exact_src, mix_dst, args.sample_rate, args.channels, target_duration)
            elif mix_source_kind == "mix":
                decode_with_ffmpeg(normal_mix_src, mix_dst, args.sample_rate, args.channels, target_duration)
            else:
                inst, _ = load_audio(inst_dst)
                voc, _ = load_audio(voc_dst)
                save_audio(mix_dst, sum_stems(inst, voc), args.sample_rate)

            tracks = []
            lengths = []
            for path in (inst_dst, voc_dst, mix_dst):
                wav, _ = load_audio(path)
                tracks.append(wav)
                lengths.append(wav.shape[-1])
            max_len = max(lengths)
            for path, wav in zip((inst_dst, voc_dst, mix_dst), tracks):
                if wav.shape[-1] < max_len:
                    wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[-1]))
                    save_audio(path, wav, args.sample_rate)

            prepared_songs.append({
                "id": song_id,
                "training_weight": float(item.get("training_weight", 1.0)),
                "note": item.get("note", ""),
                "style_tags": sorted(set(["fnf", *styles.get(song_id, [])])),
                "mix_source": mix_source_kind,
                "duration_seconds": max_len / args.sample_rate,
            })
            print(f"prepared {song_id:24s} {max_len/args.sample_rate:7.2f}s mix={mix_source_kind}")

    payload = {
        "format_version": 1,
        "sample_rate": args.sample_rate,
        "channels": args.channels,
        "songs": prepared_songs,
    }
    with open(out / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nPrepared {len(prepared_songs)} songs in {out}")


if __name__ == "__main__":
    main()
