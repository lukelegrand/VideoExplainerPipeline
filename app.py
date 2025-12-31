import os
import sys
import time
import torch
import cv2
import gradio as gr
import numpy as np
import pandas as pd
import yt_dlp
import threading
import subprocess
import soundfile as sf
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from kokoro import KPipeline

# --- IMPORT FIXES ---
# vfx is needed for audio speed effects in MoviePy v1.x
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx

# ==============================================================================
# CONFIGURATION & GLOBAL STATE
# ==============================================================================

BASE_DIR = "pipeline_output"
DIRS = {
    "downloads": os.path.join(BASE_DIR, "1_downloads"),
    "scripts": os.path.join(BASE_DIR, "2_scripts"),
    "audio": os.path.join(BASE_DIR, "3_audio"),
    "final": os.path.join(BASE_DIR, "4_final_videos")
}
CSV_LOG_PATH = os.path.join(BASE_DIR, "pipeline_log.csv")

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

if not os.path.exists(CSV_LOG_PATH):
    df = pd.DataFrame(
        columns=["Timestamp", "Topic", "Video_Link", "Input_Video_Path", "Generated_Script", "Final_Output_Path",
                 "Status"])
    df.to_csv(CSV_LOG_PATH, index=False)

# Qwen-3B is faster and uses less RAM.
# If you have >24GB VRAM/RAM, you can switch back to "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_FRAMES = 16
RESIZE_HEIGHT = 480

model_qwen = None
processor_qwen = None
pipeline_kokoro = None


# ==============================================================================
# 1. VIDEO DOWNLOADER (STRICT H.264)
# ==============================================================================
def search_and_download(topic, count=1):
    downloaded_files = []
    print(f"🔎 Searching for {count} videos on topic: '{topic}'...")

    ydl_opts = {
        # FORCE H.264 (avc1) to prevent OpenCV crashes with AV1/VP9
        'format': 'bestvideo[height<=720][ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4][vcodec^=avc]',
        'outtmpl': os.path.join(DIRS["downloads"], f"{int(time.time())}_%(id)s.%(ext)s"),
        'quiet': False,
        'noplaylist': True,
        'download_ranges': yt_dlp.utils.download_range_func(None, [(0, 60)]),
        'overwrites': True,
    }

    try:
        search_query = f"ytsearch{count}:{topic} explained"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=True)
            if 'entries' in info:
                for entry in info['entries']:
                    vid_id = entry['id']
                    for f in os.listdir(DIRS["downloads"]):
                        if vid_id in f:
                            full_path = os.path.join(DIRS["downloads"], f)
                            downloaded_files.append(
                                {"path": full_path, "url": entry['webpage_url'], "title": entry['title']})
                            break
            else:
                vid_id = info['id']
                for f in os.listdir(DIRS["downloads"]):
                    if vid_id in f:
                        full_path = os.path.join(DIRS["downloads"], f)
                        downloaded_files.append({"path": full_path, "url": info['webpage_url'], "title": info['title']})
    except Exception as e:
        print(f"❌ Download Error: {e}")

    return downloaded_files


# ==============================================================================
# 2. ANALYSIS ENGINE (QWEN)
# ==============================================================================
def load_qwen():
    global model_qwen, processor_qwen
    if model_qwen is None:
        print(f"⏳ Loading {QWEN_MODEL_ID}...")
        model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        processor_qwen = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        print("✅ Qwen Loaded.")


def analyze_video(video_path):
    load_qwen()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Could not open {video_path}")

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, total_frames // MAX_FRAMES)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            scale = RESIZE_HEIGHT / min(h, w)
            if scale < 1.0:
                frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
            frames.append(frame_rgb)
            if len(frames) >= MAX_FRAMES: break
        idx += 1
    cap.release()

    if not frames: raise ValueError("No frames extracted.")
    video_tensor = np.stack(frames)

    prompt = """
    You are writing a script for an educational video.
    Characters:
    1. Heart (Student): Curious, asks short questions about visual details.
    2. George (Expert): Knowledgeable, explains clearly based on the video.

    Format:
    Heart: [Question]
    George: [Answer]

    Task: Look at the video and write a dialogue script that matches the visuals.
    """

    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text_input = processor_qwen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor_qwen(text=[text_input], videos=[video_tensor], padding=True, return_tensors="pt").to(
        model_qwen.device)

    output_ids = model_qwen.generate(**inputs, max_new_tokens=512)
    generated_ids = [output_ids[len(inputs.input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    return processor_qwen.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ==============================================================================
# 3. AUDIO & SUBTITLE ENGINE
# ==============================================================================
def load_kokoro():
    global pipeline_kokoro
    if pipeline_kokoro is None:
        print("⏳ Loading Kokoro Pipeline...")
        pipeline_kokoro = KPipeline(lang_code='a')
        print("✅ Kokoro Loaded.")


def format_srt_time(seconds):
    millis = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes = seconds // 60
    hours = minutes // 60
    minutes %= 60
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def generate_audio_and_subs(script_text, base_filename):
    load_kokoro()
    lines = script_text.strip().split('\n')
    full_audio = []

    silence_sec = 0.3
    silence = np.zeros(int(24000 * silence_sec))

    # FILTER: Only process lines starting with these names
    VALID_CHARACTERS = {"heart", "george"}
    VOICES = {"heart": "af_heart", "george": "bm_george"}

    srt_entries = []
    current_time = 0.0
    counter = 1

    print(f"📝 Parsing Script & Generating Audio...")

    for line in lines:
        line = line.strip()
        if ":" in line:
            name_part, dialogue_part = line.split(":", 1)
            clean_name = name_part.strip().lower()
            text = dialogue_part.strip()

            # Skip invalid characters (Removes "Sure, here is the script:" intro)
            if clean_name not in VALID_CHARACTERS or not text:
                continue

            # Generate Audio
            voice = VOICES.get(clean_name)
            generator = pipeline_kokoro(text, voice=voice, speed=1.0, split_pattern=r'\n+')

            line_audio_chunks = []
            for _, _, audio in generator:
                line_audio_chunks.append(audio)

            if not line_audio_chunks: continue

            line_audio = np.concatenate(line_audio_chunks)
            duration = len(line_audio) / 24000

            full_audio.append(line_audio)
            full_audio.append(silence)

            # SRT Entry
            start_str = format_srt_time(current_time)
            end_str = format_srt_time(current_time + duration)

            # Yellow for Heart, Cyan for George
            color = "&H00FFFF" if clean_name == "heart" else "&HFFFF00"
            # SRT doesn't support advanced styling natively, but we will pass styling in FFmpeg later
            # For raw SRT, we just write the text.
            srt_entries.append(f"{counter}\n{start_str} --> {end_str}\n{text}\n\n")

            current_time += duration + silence_sec
            counter += 1

    if not full_audio:
        print("❌ No valid dialogue found.")
        return None, None

    final_audio = np.concatenate(full_audio)
    audio_path = os.path.join(DIRS["audio"], base_filename + ".wav")
    sf.write(audio_path, final_audio, 24000)

    srt_path = os.path.join(DIRS["audio"], base_filename + ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.writelines(srt_entries)

    return audio_path, srt_path


# ==============================================================================
# 4. MERGE ENGINE (LOOPING + BURN SUBS)
# ==============================================================================
def merge_smart_loop(video_path, audio_path, srt_path, output_filename):
    try:
        temp_video_path = os.path.join(DIRS["final"], "temp_" + output_filename)
        final_output_path = os.path.join(DIRS["final"], output_filename)

        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        print(f"⏱️ Video: {video_clip.duration:.2f}s | Audio: {audio_clip.duration:.2f}s")

        # Loop video if shorter than audio
        if audio_clip.duration > video_clip.duration:
            loop_count = int(audio_clip.duration // video_clip.duration) + 1
            clips = [video_clip] * loop_count
            extended_video = concatenate_videoclips(clips, method="compose")
            final_video = extended_video.subclip(0, audio_clip.duration)
        else:
            final_video = video_clip.subclip(0, audio_clip.duration)

        final_video = final_video.set_audio(audio_clip)

        # Write clean video first
        final_video.write_videofile(temp_video_path, codec="libx264", audio_codec="aac", logger=None)
        video_clip.close()
        audio_clip.close()

        # Burn Subtitles via FFmpeg
        print("🔥 Burning subtitles...")
        abs_srt = os.path.abspath(srt_path)
        abs_temp = os.path.abspath(temp_video_path)
        abs_out = os.path.abspath(final_output_path)

        # Style: White text, Black outline, Bottom margin
        style = "FontSize=16,PrimaryColour=&HFFFFFF,Outline=1,Shadow=1,MarginV=25"

        cmd = [
            "ffmpeg", "-y", "-i", abs_temp,
            "-vf", f"subtitles='{abs_srt}':force_style='{style}'",
            "-c:a", "copy", abs_out
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(temp_video_path): os.remove(temp_video_path)

        return final_output_path

    except Exception as e:
        print(f"❌ Merge Error: {e}")
        return None


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run_pipeline(topic, num_videos, progress=gr.Progress()):
    log_entries = []

    progress(0.1, desc="Downloading videos...")
    videos = search_and_download(topic, int(num_videos))

    if not videos: return "❌ No videos found.", pd.DataFrame()

    results_text = ""

    for i, vid in enumerate(videos):
        progress(0.2 + (i * 0.2), desc=f"Processing {vid['title'][:15]}...")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{timestamp}_{i}"

            print(f"🧠 Generating script...")
            script = analyze_video(vid['path'])
            with open(os.path.join(DIRS["scripts"], f"{base_name}.txt"), "w", encoding="utf-8") as f:
                f.write(script)

            print(f"🗣️ Synthesizing...")
            audio_path, srt_path = generate_audio_and_subs(script, base_name)

            if not audio_path:
                print("Skipping merge (No audio)")
                continue

            print(f"🎬 Merging...")
            final_vid_path = merge_smart_loop(vid['path'], audio_path, srt_path, f"{base_name}_final.mp4")

            entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Topic": topic,
                "Video_Link": vid['url'],
                "Input_Video_Path": vid['path'],
                "Generated_Script": script[:50] + "...",
                "Final_Output_Path": final_vid_path,
                "Status": "Success" if final_vid_path else "Merge Failed"
            }
            log_entries.append(entry)

            df_new = pd.DataFrame([entry])
            df_new.to_csv(CSV_LOG_PATH, mode='a', header=not os.path.exists(CSV_LOG_PATH), index=False)
            results_text += f"✅ Completed: {vid['title']}\n"

        except Exception as e:
            print(f"❌ Error: {e}")
            results_text += f"❌ Failed: {vid['title']} ({e})\n"

    if os.path.exists(CSV_LOG_PATH):
        full_df = pd.read_csv(CSV_LOG_PATH)
    else:
        full_df = pd.DataFrame()

    return results_text, full_df


# ==============================================================================
# UI
# ==============================================================================
with gr.Blocks(title="AI Video Factory") as demo:
    gr.Markdown("# AI Video Explainer ")
    gr.Markdown("Pipeline: Download (H.264) -> Qwen Script -> Kokoro TTS -> Subtitles -> Final Video")

    with gr.Row():
        with gr.Column(scale=1):
            t_topic = gr.Textbox(label="Topic", value="How differential gears work")
            n_count = gr.Slider(1, 5, value=1, step=1, label="Count")
            btn_run = gr.Button("🚀 Start Pipeline", variant="primary")
            out_status = gr.Textbox(label="Status")
        with gr.Column(scale=2):
            out_df = gr.Dataframe(label="Log")

    btn_run.click(run_pipeline, inputs=[t_topic, n_count], outputs=[out_status, out_df])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)