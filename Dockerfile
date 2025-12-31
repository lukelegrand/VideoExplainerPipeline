# Use Python 3.10 as base
FROM python:3.10-slim

# 1. Install System Dependencies
# ffmpeg: For video merging
# libsndfile1: For audio processing
# git: For installing some python packages if needed
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Environment Variables
# Prevents Python from buffering stdout/stderr (See logs instantly)
ENV PYTHONUNBUFFERED=1

# 3. Set Working Directory
WORKDIR /app

# 4. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY app.py .

# 6. Expose Gradio Port
EXPOSE 7860

# 7. Run the App
CMD ["python", "app.py"]