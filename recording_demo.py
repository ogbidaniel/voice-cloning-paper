"""
recording_demo.py – Gradio portal for COMP5300 voice‑cloning study
---------------------------------------------------------------
• Consistent sentence list (prompts.txt). One prompt shown at a time.
• Volunteer enters Speaker‑ID, records, clicks **Submit & Next**.
• WAV saved locally in  data/raw/<speaker>/  (add --bucket to push to S3 later).
• Metadata appended to data/meta.csv  →  speaker_id,prompt_idx,prompt_text,path
• Tracks completed prompts and total recording duration in progress.json.
• Resumes from the next incomplete prompt for a given Speaker-ID.

Tested on **Gradio 4.15**, **Python 3.10**, macOS (Intel) – May 2025.
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

import json
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

def load_progress(progress_file: Path) -> dict:
    """Load progress data from JSON file."""
    if progress_file.exists():
        try:
            with progress_file.open("r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Error decoding progress.json. Starting with an empty progress.")
            return {}
    else:
        return {}


def save_progress(progress_file: Path, speaker_id: str, prompt_idx: int, audio_duration: float) -> None:
    """Save progress to a JSON file."""
    progress = load_progress(progress_file)

    if speaker_id not in progress:
        progress[speaker_id] = {
            "completed_prompts": [],
            "total_duration_seconds": 0.0,
        }

    # Update completed prompts and total duration
    if prompt_idx not in progress[speaker_id]["completed_prompts"]:
        progress[speaker_id]["completed_prompts"].append(prompt_idx)
        progress[speaker_id]["total_duration_seconds"] += audio_duration
        progress[speaker_id]["completed_prompts"] = sorted(list(set(progress[speaker_id]["completed_prompts"]))) # Ensure unique and sorted

    with progress_file.open("w") as f:
        json.dump(progress, f, indent=2)


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
                    local_root: str,
                    progress_file: Path):
    if not speaker_id.strip():
        return gr.Warning("Please enter Speaker‑ID first."), prompts[prompt_idx], prompt_idx, "", ""
    if audio is None:
        return gr.Warning("Please record before submitting."), prompts[prompt_idx], prompt_idx, "", ""

    try:
        wav_bytes = audio_to_wav_bytes(audio)
    except Exception as e:
        return gr.Warning(f"Audio processing error: {e}"), prompts[prompt_idx], prompt_idx, "", ""

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{speaker_id}_{prompt_idx:03d}_{timestamp}.wav"

    local_file_path = Path(local_root) / "raw" / speaker_id / fname
    s3_key = f"raw/{speaker_id}/{fname}" if bucket else None
    path_str = f"s3://{bucket}/{s3_key}" if bucket else str(local_file_path)

    if bucket:
        upload_to_s3(wav_bytes, bucket, s3_key)
    else:
        save_local(wav_bytes, local_file_path)

    meta_path = Path(local_root) / "meta.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("a", newline="", encoding="utf8") as f:
        csv.writer(f).writerow([speaker_id, prompt_idx, prompts[prompt_idx], path_str])

    try:
        # Load the saved audio file to get its duration
        audio_info = sf.info(local_file_path)
        audio_duration = audio_info.duration
    except Exception as e:
        print(f"Error getting audio info: {e}")
        audio_duration = 0.0

    save_progress(progress_file, speaker_id, prompt_idx, audio_duration)
    progress_data = load_progress(progress_file)
    completed_count = len(progress_data.get(speaker_id, {}).get("completed_prompts", []))
    total_duration = progress_data.get(speaker_id, {}).get("total_duration_seconds", 0.0)

    # Determine the next prompt
    completed_prompts = set(progress_data.get(speaker_id, {}).get("completed_prompts", []))
    next_prompt_idx = -1
    for i in range(len(prompts)):
        if i not in completed_prompts:
            next_prompt_idx = i
            break
    if next_prompt_idx == -1:
        next_prompt_idx = 0 # Loop back to the beginning if all are done

    return f"✅ Saved to {path_str}", prompts[next_prompt_idx], next_prompt_idx, f"Completed: {completed_count}/{len(prompts)}", f"Total Duration: {total_duration:.2f} seconds"

def update_prompt_on_speaker_change(speaker_id: str, prompts: list[str], progress_file: Path) -> Tuple[str, int]:
    """Load progress and determine the next prompt when the speaker ID changes."""
    if not speaker_id.strip():
        return prompts[0], 0
    progress_data = load_progress(progress_file)
    completed_prompts = set(progress_data.get(speaker_id, {}).get("completed_prompts", []))
    next_prompt_idx = -1
    for i in range(len(prompts)):
        if i not in completed_prompts:
            next_prompt_idx = i
            break
    if next_prompt_idx == -1:
        next_prompt_idx = 0
    return prompts[next_prompt_idx], next_prompt_idx

# -----------------------------------------------------------------------------
# UI builder
# -----------------------------------------------------------------------------

def build_ui(prompts: list[str], bucket: str | None, local_root: str):
    progress_file = Path("progress.json") # Define the progress file path here

    with gr.Blocks(title="COMP5300 Voice‑Recording Portal") as demo:
        gr.Markdown("""## Speaking Phase\n### Record sentences for the voice‑cloning study\n1. Put on headphones and find a quiet space.\n2. Click the microphone, read the sentence **exactly**, click stop.\n3. Hit **Submit & Next**. Repeat until done.""")
        gr.Markdown("""**Note:** This is a research study. Your recordings will be used to train a voice model.\nPlease enter your `Speaker-ID` before recording. Use PV username (e.g. Jane Doe = `jdoe`).""")

        speaker = gr.Text(label="Speaker‑ID")
        prompt_box = gr.Textbox(label="Sentence to read")
        idx_state = gr.State(0)
        progress_display = gr.Markdown(label="Progress")
        duration_display = gr.Markdown(label="Total Duration")

        mic = gr.Audio(sources=["microphone"], format="wav", label="🎙️ Record here")
        status = gr.Markdown()
        btn = gr.Button("Submit & Next ➡️")

        speaker.change(fn=update_prompt_on_speaker_change,
                       inputs=[speaker, gr.State(prompts), gr.State(progress_file)],
                       outputs=[prompt_box, idx_state])

        btn.click(record_and_save,
                 inputs=[speaker, idx_state, mic, gr.State(prompts), gr.State(bucket), gr.State(local_root), gr.State(progress_file)],
                 outputs=[status, prompt_box, idx_state, progress_display, duration_display])
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