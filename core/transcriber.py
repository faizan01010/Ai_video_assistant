import whisper
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from dotenv import load_dotenv
 
# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25
 
# How many pieces are sent to Sarvam at the same time.
# If you see 429 (rate limit) errors, lower this (e.g. SARVAM_WORKERS=2 in .env).
SARVAM_WORKERS = int(os.getenv("SARVAM_WORKERS", "6"))
SARVAM_RETRIES = 4
 
 
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
 
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", ".env")   # core ke ek level upar = project root
 
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
 
_model = None
_sarvam_key = None
 
 
def get_sarvam_key() -> str:
    """Read the key once and cache it (the old version re-read .env for every piece)."""
    global _sarvam_key
    if _sarvam_key:
        return _sarvam_key
 
    load_dotenv(ENV_PATH, override=True, encoding="utf-8-sig")
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        raise RuntimeError(f"SARVAM_API_KEY not found. Looked for .env at: {os.path.abspath(ENV_PATH)}")
    _sarvam_key = key.strip().strip('"').strip("'")
    return _sarvam_key
 
 
def load_model():
 
    global _model
 
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model
 
 
def transcribe_chunk_whisper(chunk_path: str) -> str:
 
    model = load_model()
 
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]
 
 
def _send_to_sarvam(piece_path: str, api_key: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript.
    Retries on rate limits (429), server errors (5xx) and network hiccups."""
    headers = {"api-subscription-key": api_key}
    last_error = None
 
    for attempt in range(SARVAM_RETRIES):
        try:
            with open(piece_path, "rb") as f:
                files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
                data = {"model": SARVAM_MODEL, "with_diarization": "false"}
                response = requests.post(
                    SARVAM_STT_TRANSLATE_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=120,
                )
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            time.sleep(2 ** attempt)
            continue
 
        if response.status_code == 429 or response.status_code >= 500:
            wait = response.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else 2 ** attempt)
            last_error = f"{response.status_code}: {response.text[:200]}"
            continue
 
        if not response.ok:
            print(f"\n❌ Sarvam returned {response.status_code}")
            print(f"Response body: {response.text}\n")
            response.raise_for_status()
 
        return response.json().get("transcript", "")
 
    raise RuntimeError(f"Sarvam failed after {SARVAM_RETRIES} attempts: {last_error}")
 
 
def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send them to Sarvam in parallel, and join the
    transcripts in the original order.
    """
    api_key = get_sarvam_key()  # key missing ho to yahin pe saaf error aa jayega
 
    audio = AudioSegment.from_wav(chunk_path)
    # Speech models only need 16 kHz mono. This makes every upload ~5x smaller
    # than 44.1 kHz stereo, which matters a lot when many pieces upload at once.
    audio = audio.set_frame_rate(16000).set_channels(1)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    starts = list(range(0, len(audio), piece_ms))
    total_pieces = len(starts)
 
    # 1) Cut and save all pieces first (fast, local disk only)
    piece_paths = []
    for i, start in enumerate(starts):
        piece_path = f"{chunk_path}_sv_{i}.wav"
        audio[start: start + piece_ms].export(piece_path, format="wav")
        piece_paths.append(piece_path)
 
    # 2) Send them in parallel
    def work(item):
        i, path = item
        t0 = time.perf_counter()
        size_kb = os.path.getsize(path) / 1024
        try:
            text = _send_to_sarvam(path, api_key)
            print(f"  → Sarvam piece {i + 1}/{total_pieces} done "
                  f"({size_kb:.0f} KB, {time.perf_counter() - t0:.1f}s)")
            return text.strip()
        finally:
            if os.path.exists(path):
                os.remove(path)
 
    try:
        with ThreadPoolExecutor(max_workers=SARVAM_WORKERS) as ex:
            texts = list(ex.map(work, enumerate(piece_paths)))   # map keeps the original order
    finally:
        # clean up anything left behind if a piece failed
        for p in piece_paths:
            if os.path.exists(p):
                os.remove(p)
 
    return " ".join(t for t in texts if t)
 
 
def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)
 
 
def transcribe_all(chunks: list, language: str = "english") -> str:
 
    full_transcript = ""
 
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")
 
    for i, chunk in enumerate(chunks):
 
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
 
        text = transcribe_chunk(chunk, language=language)
 
        full_transcript += text + " "
 
    print("Transcription complete.")
 
    return full_transcript.strip()