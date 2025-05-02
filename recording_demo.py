"""
recording_demo.py – Gradio portal for COMP5300 voice‑cloning study
---------------------------------------------------------------
• Consistent sentence list (prompts.txt). One prompt shown at a time.
• Volunteer enters Speaker‑ID, records, clicks **Submit & Next**.
• WAV saved locally in  data/raw/<speaker>/  (add --bucket to push to S3 later).
• Metadata appended to data/meta.csv  →  speaker_id,prompt_idx,prompt_text,path

Tested on **Gradio 4.15**, **Python 3.10**, macOS (Intel) – April 2025.
Install deps:
    pip install gradio soundfile numpy boto3 silero-vad
Run locally:
    python recording_demo.py --prompts prompts.txt
Run with S3:
    python recording_demo.py --prompts prompts.txt --bucket my-tts-raw
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
from pathlib import Path
from typing import List, Tuple, Union

import gradio as gr
import numpy as np
import soundfile as sf

try:
    import boto3  # optional – only if you pass --bucket
except ModuleNotFoundError:
    boto3 = None  # noqa: N816

AudioLike = Union[Tuple[int, np.ndarray], str, dict]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_prompts(path: Path) -> List[str]:
    """Load non‑empty lines from prompts.txt."""
    return [ln.strip() for ln in path.read_text(encoding="utf8").splitlines() if ln.strip()]


def audio_to_wav_bytes(audio: AudioLike) -> bytes:
    """Convert Gradio Audio return‑value to raw WAV bytes."""
    # Case 1: old tuple format (sr, np.ndarray)
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, wav = audio  # type: ignore
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        return buf.getvalue()

    # Case 2: new dict format {"filename": .., "data": (<sr>, np.ndarray)}
    if isinstance(audio, dict):
        if "data" in audio and audio["data"]:
            sr, wav = audio["data"]  # type: ignore
            buf = io.BytesIO()
            sf.write(buf, wav, sr, format="WAV")
            return buf.getvalue()
        if "path" in audio and audio["path"]:
            return Path(audio["path"]).read_bytes()  # type: ignore

    # Case 3: filepath string
    if isinstance(audio, str) and Path(audio).exists():
        return Path(audio).read_bytes()

    raise ValueError("Unrecognized audio format from Gradio component")


def upload_to_s3(data: bytes, bucket: str, key: str):
    if boto3 is None:
        raise RuntimeError("boto3 not installed; can't upload to S3.")
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=data, ContentType="audio/wav")


def save_local(data: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

# -----------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------

def record_and_save(speaker_id: str,
                    prompt_idx: int,
                    audio: AudioLike,
                    prompts: list[str],
                    bucket: str | None,
                    local_root: str):
    if not speaker_id.strip():
        return gr.Warning("Please enter Speaker‑ID first."), prompts[prompt_idx], prompt_idx
    if audio is None:
        return gr.Warning("Please record before submitting."), prompts[prompt_idx], prompt_idx

    try:
        wav_bytes = audio_to_wav_bytes(audio)
    except Exception as e:
        return gr.Warning(f"Audio processing error: {e}"), prompts[prompt_idx], prompt_idx

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{speaker_id}_{prompt_idx:03d}_{timestamp}.wav"

    if bucket:
        key = f"raw/{speaker_id}/{fname}"
        upload_to_s3(wav_bytes, bucket, key)
        path_str = f"s3://{bucket}/{key}"
    else:
        path = Path(local_root) / "raw" / speaker_id / fname
        save_local(wav_bytes, path)
        path_str = str(path)

    meta_path = Path(local_root) / "meta.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("a", newline="", encoding="utf8") as f:
        csv.writer(f).writerow([speaker_id, prompt_idx, prompts[prompt_idx], path_str])

    next_idx = (prompt_idx + 1) % len(prompts)
    return f"✅ Saved to {path_str}", prompts[next_idx], next_idx

# -----------------------------------------------------------------------------
# UI builder
# -----------------------------------------------------------------------------

def build_ui(prompts: list[str], bucket: str | None, local_root: str):
    with gr.Blocks(title="COMP5300 Voice‑Recording Portal") as demo:
        gr.Markdown("""## Speaking Phase\n### Record sentences for the voice‑cloning study\n1. Put on headphones and find a quiet space.\n2. Click the microphone, read the sentence **exactly**, click stop.\n3. Hit **Submit & Next**. Repeat until done.\n""")

        speaker = gr.Text(label="Speaker‑ID (email or alias)")
        prompt_box = gr.Textbox(value=prompts[0], interactive=False, label="Sentence to read")
        idx_state = gr.State(0)

        mic = gr.Audio(sources=["microphone"], format="wav", label="🎙️ Record here")
        status = gr.Markdown()
        btn = gr.Button("Submit & Next ➡️")

        btn.click(record_and_save,
                 inputs=[speaker, idx_state, mic, gr.State(prompts), gr.State(bucket), gr.State(local_root)],
                 outputs=[status, prompt_box, idx_state])
    return demo

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True, help="Text file with one sentence per line")
    ap.add_argument("--bucket", default=None, help="S3 bucket name (omit for local‑only)")
    ap.add_argument("--local_root", default="data", help="Local save directory root")
    ap.add_argument("--server_port", type=int, default=7860)
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    ui = build_ui(prompts, args.bucket, args.local_root)
    ui.launch(server_port=args.server_port, share=False)

if __name__ == "__main__":
    main()