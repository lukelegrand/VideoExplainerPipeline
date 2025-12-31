# **AI Video Explainer Factory 🎬🤖**

A fully automated video production pipeline that scrapes YouTube content, "watches" it using a Vision Language Model (VLM), generates an educational dialogue script, synthesizes voiceovers, and produces a final subtitled video.

## **🚀 Key Features**

* **Automated Content Sourcing**: Searches and downloads YouTube videos based on a topic, strictly ensuring H.264 formatting for maximum compatibility11.

* **Visual Analysis**: Uses **Qwen2.5-VL-3B-Instruct** to extract frames and understand visual context, generating a script based on what is actually shown on screen2.

* **Dialogue Generation**: Creates an educational back-and-forth script between two characters ("Heart" and "George")3.

* **High-Quality TTS**: Utilizes **Kokoro** (via KPipeline) for natural-sounding speech synthesis4.

* **Smart Video Editing**:  
  * Automatically loops video footage if the generated audio is longer than the source clip5.

  * Burns hardcoded subtitles (SRT) into the video using FFmpeg with custom styling6.

* **Interactive UI**: Built with **Gradio** for easy input and real-time log monitoring7.

## ---

**🛠️ Tech Stack**

* **Language**: Python 3.10 88

* **UI Framework**: Gradio 9

* **AI Models**:  
  * Vision: Qwen2.5-VL-3B-Instruct 10

  * Audio: Kokoro v0.19 (via KPipeline) 11

* **Video Processing**: yt-dlp, moviepy (v1.x), OpenCV, FFmpeg121212.

## ---

**📋 Prerequisites**

* **Docker** & **Docker Compose** installed on your machine.  
* **(Optional but Recommended)** NVIDIA GPU with drivers and **NVIDIA Container Toolkit** installed for hardware acceleration.  
  * *Note: Running Qwen-VL and TTS on CPU will be significantly slower.*

## ---

**⚙️ Installation & Usage**

### **1\. Project Setup**

Ensure your project directory contains the following files:

* app.py  
* Dockerfile  
* docker-compose.yml  
* requirements.txt

### **2\. Run with Docker (Recommended)**

The provided docker-compose.yml handles volume mounting and networking automatically.

1. **Build and Start the Container:**  
   Bash  
   docker-compose up \--build

2. Access the Interface:  
   Open your browser and navigate to:  
   http://localhost:7860

   13

### **3\. Enable GPU Support (Optional)**

To use your NVIDIA GPU, open docker-compose.yml and uncomment the deploy section:

YAML

    \# deploy:  
    \#   resources:  
    \#     reservations:  
    \#       devices:  
    \#         \- driver: nvidia  
    \#           count: 1  
    \#           capabilities: \[gpu\]

14

Then rebuild the container:

Bash

docker-compose up \--build \--force-recreate

## ---

**📂 Directory Structure**

The application persists data to your local machine using Docker volumes. You will see a pipeline\_output folder created in your project directory containing the following:

Plaintext

pipeline\_output/  
├── 1\_downloads/       \# Raw downloaded YouTube videos  
├── 2\_scripts/         \# Generated scripts (Heart/George dialogue)  
├── 3\_audio/           \# Generated .wav files and .srt subtitles  
├── 4\_final\_videos/    \# The finished video with burnt-in subtitles  
└── pipeline\_log.csv   \# A CSV log of all processed videos

15

Note: A hf\_cache folder is also created to store downloaded HuggingFace models so you don't need to re-download them on every restart16.

## ---

**🧩 Configuration**

You can modify app.py to tweak the following settings:

* Model Selection:  
  By default, the 3B parameter model is used to save RAM. If you have high VRAM (\>24GB), you can switch to the 7B model by editing:  
  Python  
  QWEN\_MODEL\_ID \= "Qwen/Qwen2.5-VL-7B-Instruct"

  17

* Video Sampling:  
  Adjust how many frames the AI "sees" by changing MAX\_FRAMES \= 16 in app.py. Higher numbers increase accuracy but consume more VRAM18.

## ---

**⚠️ Troubleshooting**

**1\. "RuntimeError: CUDA out of memory"**

* **Cause**: The Qwen model or TTS pipeline is using too much VRAM.  
* **Fix**: Lower MAX\_FRAMES in app.py or ensure no other GPU-intensive tasks are running. If you are on CPU, this error shouldn't occur, but generation will be slow.

**2\. Video Generation Failures (OpenCV/Codec issues)**

* The script explicitly requests avc1 (H.264) video from YouTube to prevent compatibility issues with OpenCV19. If a video fails to download, it might not have an H.264 stream available.

**3\. Audio Speed/Effects Errors**

* This project specifically requires moviepy\<2.0.0 due to API changes in version 2.0 regarding vfx. Ensure you do not upgrade MoviePy manually20.  