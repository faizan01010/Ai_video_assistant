import os
import shutil
import time
 
import yt_dlp
from pydub import AudioSegment
 
DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
 
# Project root = one level above utils/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
 
 
def download_youtube_audio(url: str) -> str:
    # %(id)s is safer than %(title)s: no special characters, no clashes between videos
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    ydl_opts = {
        "remote_components": ["ejs:github"],
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
 
    # Cookies: YT_COOKIES_FILE in .env, else <project root>/cookies.txt (only if it exists).
    # NEVER commit cookies.txt to GitHub, it contains your logged-in YouTube session.
    cookies = os.getenv("YT_COOKIES_FILE") or os.path.join(PROJECT_ROOT, "cookies.txt")
    if os.path.exists(cookies):
        ydl_opts["cookiefile"] = cookies
 
    # Deno (JS runtime used by yt-dlp): DENO_PATH in .env, else whatever is on PATH.
    deno = os.getenv("DENO_PATH") or shutil.which("deno")
    if deno:
        ydl_opts["js_runtimes"] = {"deno": {"path": deno}}
 
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # final file is always <id>.wav after FFmpegExtractAudio
        filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    return filename
 
 
def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16 kHz mono WAV format using pydub."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(DOWNLOAD_DIR, base + "_converted.wav")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16khz
    audio.export(output_path, format="wav")
    return output_path
 
 
def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    # Downsample BEFORE cutting: YouTube audio arrives as 44.1/48 kHz stereo, which makes
    # every chunk (and every piece sent to the transcription API) ~5x bigger than needed.
    audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)
    chunk_ms = chunk_minutes * 60 * 1000
 
    chunks = []
 
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
 
        chunks.append(chunk_path)
 
    return chunks
 
 
def process_input(source: str) -> list:
    t0 = time.perf_counter()
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
        print(f"Download + WAV conversion took {time.perf_counter() - t0:.1f}s")
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)
        print(f"Conversion took {time.perf_counter() - t0:.1f}s")
 
    t1 = time.perf_counter()
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Chunking took {time.perf_counter() - t1:.1f}s")
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks