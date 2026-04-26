# -*- coding: utf-8 -*-
"""
video_processor.py
------------------
Functions to download a YouTube video, extract frames, audio, and transcribe to text.

Overflow Guard (same pattern as pipeline.py _partition_once()):
    - download_video()   : skips download if input_vid.mp4 already exists
    - video_to_images()  : skips frame extraction if frame*.png files already exist
    Both accept force=True to bypass the guard for a fresh run.

Requirements (pip install):
    yt-dlp, moviepy, SpeechRecognition, openai-whisper, soundfile, pydub

System Requirements:
    - FFmpeg installed and in PATH (Windows: https://ffmpeg.org/download.html)

Usage:
    from video_processor import download_video, video_to_images, video_to_audio, audio_to_text
"""

import os
import glob
import shutil
import yt_dlp
import speech_recognition as sr
from moviepy import VideoFileClip


def _existing_frame_count(folder: str) -> int:
    """Return the number of frame*.png files already in *folder*."""
    if not os.path.isdir(folder):
        return 0
    return len(glob.glob(os.path.join(folder, "frame*.png")))


def clear_video_cache(video_dir: str, mixed_dir: str, lancedb_uri: str | None = None) -> None:
    """
    Remove cached video artifacts from prior runs.

    Every new user-provided URL starts from a clean slate so stale video,
    frames, transcript, and LanceDB data cannot leak across runs.
    """
    for path in (video_dir, mixed_dir, lancedb_uri):
        if path and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(mixed_dir, exist_ok=True)
    print("  Cleared cached video artifacts from previous run.")


def download_video(url: str, output_path: str, force: bool = False):
    """
    Download a YouTube video using yt-dlp and return its metadata.

    Overflow Guard: If `input_vid.mp4` already exists in *output_path* and
    ``force=False``, the download is skipped entirely and cached stub
    metadata is returned. Pass ``force=True`` to re-download.

    Args:
        url         (str):  YouTube video URL.
        output_path (str):  Directory to save 'input_vid.mp4'.
        force       (bool): If True, re-download even if file exists.

    Returns:
        dict | None: {'Author', 'Title', 'Views'} or None on failure.
    """
    video_file = os.path.join(output_path, "input_vid.mp4")
    os.makedirs(output_path, exist_ok=True)

    # ── Guard: skip re-download if already present ─────────────────
    if not force and os.path.exists(video_file):
        print(
            f"  Video already exists at '{video_file}' — skipping download "
            "(pass force=True to re-download)."
        )
        return {"Author": "Cached", "Title": os.path.basename(video_file), "Views": 0}

    # ── Fresh download ──────────────────────────────────────────────
    ydl_opts = {
        "outtmpl": video_file,
        "format" : "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet"  : True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            metadata  = {
                "Author": info_dict.get("uploader", "N/A"),
                "Title" : info_dict.get("title", "N/A"),
                "Views" : info_dict.get("view_count", 0),
            }
            ydl.download([url])
            print(f"  Downloaded: {metadata['Title']}")
            return metadata
    except Exception as e:
        print(f"  Download error: {e}")
        return None


def video_to_images(video_path: str, output_folder: str, fps: float = 0.2, force: bool = False):
    """
    Extract frames from video and save as PNG files.

    Overflow Guard: If frame*.png files already exist in *output_folder* and
    ``force=False``, extraction is skipped to prevent duplicate accumulation.
    Pass ``force=True`` to always re-extract.

    Args:
        video_path    (str):   Path to .mp4 file.
        output_folder (str):   Directory to save frame PNGs.
        fps         (float):   Frames per second (default 0.2 = 1 frame/5s).
        force        (bool):   If True, extract even if frames exist.
    """
    os.makedirs(output_folder, exist_ok=True)
    existing = _existing_frame_count(output_folder)

    # ── Guard: skip re-extraction if frames already present ────────
    if not force and existing > 0:
        print(
            f"  Found {existing} existing frame(s) in '{output_folder}' — "
            "skipping extraction (pass force=True to override)."
        )
        return

    # ── Fresh extraction ────────────────────────────────────────────
    print(f"  Extracting frames from: {video_path}")
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(os.path.join(output_folder, "frame%04d.png"), fps=fps)
    print(f"  Frames saved to: {output_folder}")


def video_to_audio(video_path: str, output_audio_path: str):
    """
    Extract audio track from video and save as .wav.

    Args:
        video_path        (str): Path to .mp4 file.
        output_audio_path (str): Destination .wav path.
    """
    clip  = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)
    print(f"  Audio saved to: {output_audio_path}")


def audio_to_text(audio_path: str) -> str:
    """
    Transcribe a .wav file to text using OpenAI Whisper.

    Args:
        audio_path (str): Path to .wav file.

    Returns:
        str: Transcribed text, or empty string on failure.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_whisper(audio_data)
        print("  Transcription complete.")
        return text
    except sr.UnknownValueError:
        print("  Whisper could not understand the audio.")
        return ""


if __name__ == "__main__":
    import video_config as cfg
    metadata = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", cfg.OUTPUT_VIDEO_DIR)
    print(metadata)
