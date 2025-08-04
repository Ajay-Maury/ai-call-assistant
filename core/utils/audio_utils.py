import asyncio
import os
import re
import uuid
import subprocess
import numpy as np
import webrtcvad
import logging

from aiVoiceAssistant.constants import AUDIO_CHUNK_DIR, AUDIO_SILENCE_THRESHOLDS, RESPONSE_AUDIO_CHUNK_DIR

# Setup logger
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

# Create a VAD instance with aggressiveness level (0–3)
vad = webrtcvad.Vad(2)

def _generate_mulaw_to_pcm16_table() -> np.ndarray:
    BIAS = 0x84
    table = np.zeros(256, dtype=np.int16)

    for i in range(256):
        mu_law_byte = ~i & 0xFF
        sign = mu_law_byte & 0x80
        exponent = (mu_law_byte >> 4) & 0x07
        mantissa = mu_law_byte & 0x0F

        magnitude = ((mantissa << 4) + 0x08) << exponent
        sample = magnitude - BIAS

        if sign != 0:
            sample = -sample

        table[i] = sample

    return table

_MULAW_DECODE_TABLE = _generate_mulaw_to_pcm16_table()

def is_silent_mulaw_audio(
    mulaw_audio_bytes: bytes,
    max_amplitude_threshold: int = AUDIO_SILENCE_THRESHOLDS["MAX_AMPLITUDE"],
    min_rms_dbfs_threshold: float = AUDIO_SILENCE_THRESHOLDS["MIN_RMS_DBFS"],
) -> bool:
    if not mulaw_audio_bytes:
        logger.debug("Received empty µ-law audio bytes, treating as silent.")
        return True

    mulaw_array = np.frombuffer(mulaw_audio_bytes, dtype=np.uint8)
    pcm_samples = _MULAW_DECODE_TABLE[mulaw_array]

    max_amp = np.max(np.abs(pcm_samples))
    if max_amp > max_amplitude_threshold:
        logger.debug(f"Non-silent: Amplitude {max_amp} > threshold {max_amplitude_threshold}")
        return False

    pcm_float = pcm_samples.astype(np.float32)
    rms = np.sqrt(np.mean(pcm_float ** 2))
    if rms == 0:
        return True

    dbfs = 20 * np.log10(rms / 32768.0)
    if dbfs >= min_rms_dbfs_threshold:
        logger.debug(f"Non-silent: RMS={rms:.2f}, dBFS={dbfs:.2f}, Threshold={min_rms_dbfs_threshold} dBFS")

    return dbfs < min_rms_dbfs_threshold

def convert_mulaw_to_wav(call_sid: str, mulaw_audio_bytes: bytes) -> str:
    raw_file_path = os.path.join(AUDIO_CHUNK_DIR, f"{call_sid}_{uuid.uuid4()}.raw")
    wav_file_path = raw_file_path.replace(".raw", ".wav")

    try:
        with open(raw_file_path, "wb") as raw_file:
            raw_file.write(mulaw_audio_bytes)
        logger.debug(f"Wrote raw μ-law audio to {raw_file_path}")

        subprocess.run([
            "ffmpeg",
            "-f", "mulaw",
            "-ar", "8000",
            "-ac", "1",
            "-i", raw_file_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            wav_file_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        logger.debug(f"Converted .raw to .wav successfully: {wav_file_path}")
        return wav_file_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Audio conversion failed: {e.stderr.decode().strip()}")
        return None

    finally:
        if os.path.exists(raw_file_path):
            os.remove(raw_file_path)

def convert_wav_to_mulaw(wav_path: str, output_dir: str = RESPONSE_AUDIO_CHUNK_DIR) -> bytes:
    if output_dir is None:
        output_dir = os.path.dirname(wav_path)

    raw_path = wav_path.replace(".wav", ".raw")

    try:
        subprocess.run([
            "ffmpeg",
            "-i", wav_path,
            "-ar", "8000",
            "-ac", "1",
            "-acodec", "pcm_mulaw",
            "-f", "mulaw",
            "-y",
            raw_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        logger.debug(f"Converted WAV to μ-law: {raw_path}")
        with open(raw_path, "rb") as f:
            return f.read()

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg μ-law conversion failed: {e.stderr.decode().strip()}")
        return None

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(raw_path):
            os.remove(raw_path)

async def convert_wav_chunk_bytes_to_mulaw(wav_chunk: bytes) -> bytes:
    temp_dir = os.path.join(RESPONSE_AUDIO_CHUNK_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    unique_id = uuid.uuid4()
    temp_wav_path = os.path.join(temp_dir, f"temp_{unique_id}.wav")
    temp_raw_path = temp_wav_path.replace(".wav", ".raw")
    try:

        with open(temp_wav_path, "wb") as f:
            f.write(wav_chunk)
        logger.debug(f"Temporary WAV file written: {temp_wav_path}")

        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i", temp_wav_path,
            "-ar", "8000",
            "-ac", "1",
            "-acodec", "pcm_mulaw",
            "-af", "highpass=f=200,lowpass=f=3400",
            "-f", "mulaw",
            "-y",
            temp_raw_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.wait()

        if os.path.exists(temp_raw_path):
            with open(temp_raw_path, "rb") as f:
                audio_data = f.read()
            logger.debug(f"Converted WAV chunk to μ-law: {temp_raw_path}")
            return audio_data
        else:
            logger.error(f"μ-law output not created: {temp_raw_path}")
            return None

    except Exception as e:
        logger.exception(f"WAV chunk μ-law conversion failed: {e}")
        return None

    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)

def mulaw_to_pcm(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.array([], dtype=np.int16)

    mulaw_array = np.frombuffer(audio_bytes, dtype=np.uint8)
    pcm_samples = _MULAW_DECODE_TABLE[mulaw_array]
    return pcm_samples

def is_voiced(audio_bytes: bytes, sample_rate: int = 8000, frame_duration_ms: int = 30) -> bool:
    try:
        pcm_samples = mulaw_to_pcm(audio_bytes)
        frame_size = int(sample_rate * frame_duration_ms / 1000)

        if len(pcm_samples) < frame_size:
            return False

        frame = pcm_samples[:frame_size].tobytes()
        result = vad.is_speech(frame, sample_rate)
        logger.debug(f"WebRTC VAD result: {result} (frame size: {frame_size})")
        return result

    except Exception as e:
        logger.exception(f"[VAD Error]: {e}")
        return False


def split_text_by_speech_phrases(text: str, max_len: int = 200) -> list[str]:
    """
    Split text into smaller speech-friendly chunks using punctuation.
    """
    if not text:
        return []

    chunks = re.split(r'(?<=[.?!,])\s+', text.strip())
    output = []
    buffer = ""
    for chunk in chunks:
        if len(buffer) + len(chunk) <= max_len:
            buffer += " " + chunk
        else:
            output.append(buffer.strip())
            buffer = chunk
    if buffer:
        output.append(buffer.strip())
    return output

def split_text_by_length(text: str, max_words: int = 10) -> list[str]:
    # First split by punctuation (., !, ?), then split further by word count
    raw_chunks = re.split(r'(?<=[.!?])\s+', text)
    final_chunks = []

    for chunk in raw_chunks:
        words = chunk.split()
        while len(words) > max_words:
            final_chunks.append("     ".join(words[:max_words]))
            words = words[max_words:]
        if words:
            final_chunks.append("     ".join(words))

    return [c.strip() for c in final_chunks if c.strip()]


def split_text_by_punctuation_and_word_count(text, max_words=15):
    """
    Splits text into smaller chunks using punctuation and word count.
    Ensures punctuation is preserved and placed at the end of each segment.
    """
    # Step 1: Split text using all meaningful punctuation
    raw_segments = re.split(r'([.!?,;:—–…])', text)

    # Step 2: Reconstruct segments with punctuation
    segments = []
    i = 0
    while i < len(raw_segments):
        part = raw_segments[i].strip()
        if not part:
            i += 1
            continue

        if i + 1 < len(raw_segments) and re.match(r'[.!?,;:—–…]', raw_segments[i + 1]):
            segment = f"{part}{raw_segments[i + 1]}"
            i += 2
        else:
            segment = part
            i += 1
        segments.append(segment.strip())

    # Step 3: Split segments into word-limited chunks
    final_chunks = []
    for seg in segments:
        words = seg.split()
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            if not re.search(r'[.!?,;:—–…]$', chunk.strip()):
                chunk += '.'  # add period if none at end
            final_chunks.append(chunk.strip())

    return final_chunks


import re

def split_text_by_word_count(text: str, max_words: int = 15) -> list[str]:
    """
    Splits text first by punctuation, then by word count within each segment.

    Args:
        text (str): Input text to split.
        max_words (int): Max words per chunk.

    Returns:
        List[str]: List of chunks suitable for TTS.
    """
    chunks = []

    # Step 1: Split by punctuation while capturing it
    segments = re.split(r'([.!?,;:—–…])', text)
    grouped = []

    # Step 2: Merge punctuation back with its preceding text
    for i in range(0, len(segments), 2):
        part = segments[i].strip()
        punctuation = segments[i+1].strip() if i+1 < len(segments) else ""
        if part:
            grouped.append(part + punctuation)

    # Step 3: Split each group by word count
    for group in grouped:
        words = group.strip().split()
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            if not re.search(r'[.!?,;:—–…]$', chunk.strip()):
                chunk += '.'  # Add period if none present at the end
            chunks.append(chunk)

    return chunks

