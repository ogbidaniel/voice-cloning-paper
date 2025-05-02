"""
recording_demo.py – Gradio portal for COMP5300 voice‑cloning study
------------------------------------------------------------------
• One sentence prompt shown at a time (same order for everyone)
• Volunteer enters a unique speaker‑ID (e.g. uni email or alias)
• Records the sentence via microphone, reviews playback, then hits
  *Submit* to upload the WAV to Amazon S3 **or** fallback local disk.
• Filenames: <speaker>_<prompt_idx>_<YYYYMMDD‑HHMMSS>.wav
• Metadata row (<speaker>, <prompt_idx>, <prompt_text>, <s3key>) is
  appended to prompts.csv (stored in the same bucket or ./data/meta).

Prereqs
--------
conda activate voicetrust
pip install gradio boto3 soundfile numpy

AWS setup (any one works)
-------------------------
1. Export creds in environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
   AWS_DEFAULT_REGION) ––OR––
2. Attach an IAM role with PutObject permission on the bucket.

Run
---
python recording_demo.py --prompts prompts.txt --bucket my‑tts‑raw

"""

import argparse, io, os, time, uuid, csv, datetime as dt
from pathlib import Path

import boto3
import gradio as gr
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts(path: Path):
    """Returns list[str] without empty lines"""
    return [ln.strip() for ln in path.read_text(encoding="utf8").splitlines() if ln.strip()]


def save_wav_to_buffer(audio: tuple[np.ndarray, int]):
    """Gradio mic returns (sr, np.ndarray); we return bytes IO"""
    sr, wav = audio
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return buf.getvalue()


def upload_to_s3(data: bytes, bucket: str, key: str):
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="audio/wav")


def save_local(data: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def record_and_save(speaker_id: str, prompt_idx: int, audio, prompts, bucket, local_root):
    if speaker_id.replace(" ", "") == "":
        return gr.Warning("Please enter Speaker ID first."), None, prompt_idx

    if audio is None:
        return gr.Warning("Record the sentence before submitting."), None, prompt_idx

    data = save_wav_to_buffer(audio)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{speaker_id}_{prompt_idx:03d}_{timestamp}.wav"

    if bucket:
        key = f"raw/{speaker_id}/{fname}"
        upload_to_s3(data, bucket, key)
        loc = f"s3://{bucket}/{key}"
    else:
        path = Path(local_root) / "raw" / speaker_id / fname
        save_local(data, path)
        loc = str(path)

    # append metadata
    meta_row = [speaker_id, prompt_idx, prompts[prompt_idx], loc]
    meta_dest = Path(local_root) / "meta.csv" if not bucket else Path("meta.csv")
    with meta_dest.open("a", newline="", encoding="utf8") as f:
        csv.writer(f).writerow(meta_row)

    # prepare next prompt (cyclic)
    next_idx = (prompt_idx + 1) % len(prompts)
    return f"✅ Saved to {loc}", prompts[next_idx], next_idx

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui(prompts, bucket, local_root):
    with gr.Blocks(title="Voice‑Cloning Recording Portal") as demo:
        gr.Markdown("""# COMP5300 Voice Recording\nPlease wear headphones, sit in a quiet room, and read each sentence **exactly** as shown.\n""")

        speaker = gr.Text(label="Speaker ID (use school email or alias)")
        prompt_txt = gr.Textbox(value=prompts[0], interactive=False, label="Sentence to read")
        prompt_idx = gr.State(0)

        mic = gr.Audio(source="microphone", type="numpy", label="Record here (click and speak)")
        status = gr.Markdown()

        submit = gr.Button("Submit & Next ➡️")
        submit.click(record_and_save,
                     inputs=[speaker, prompt_idx, mic, gr.State(prompts), gr.State(bucket), gr.State(local_root)],
                     outputs=[status, prompt_txt, prompt_idx])

        gr.Markdown("You can close the tab when you've completed all prompts. Thank you!")
    return demo

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--bucket", help="S3 bucket name (omit for local)")
    ap.add_argument("--local_root", default="data", help="Local save dir if no S3")
    ap.add_argument("--server_port", type=int, default=7860)
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    demo = build_ui(prompts, args.bucket, args.local_root)
    demo.launch(server_port=args.server_port, share=False)

if __name__ == "__main__":
    main()