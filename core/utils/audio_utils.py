import asyncio
import os
import uuid
import subprocess
import numpy as np

import webrtcvad

from aiVoiceAssistant.constants import AUDIO_CHUNK_DIR, AUDIO_SILENCE_THRESHOLDS, RESPONSE_AUDIO_CHUNK_DIR


# Create a VAD instance with aggressiveness level (0–3)
vad = webrtcvad.Vad(2)

# Generate µ-law to PCM16 decode table (based on G.711 standard)
def _generate_mulaw_to_pcm16_table() -> np.ndarray:
    """
    Creates a 256-element lookup table for converting 8-bit µ-law values
    to 16-bit PCM signed integers using the G.711 decoding formula.

    Returns:
        np.ndarray: Lookup table for fast µ-law to PCM conversion.
    """
    BIAS = 0x84  # Bias used in G.711 decoding (132 decimal)
    table = np.zeros(256, dtype=np.int16)

    for i in range(256):
        mu_law_byte = ~i & 0xFF  # Invert 8-bit value (two's complement)
        sign = mu_law_byte & 0x80  # Extract sign bit (bit 7)
        exponent = (mu_law_byte >> 4) & 0x07  # Bits 4–6 (3-bit exponent)
        mantissa = mu_law_byte & 0x0F  # Bits 0–3 (4-bit mantissa)

        # Decode using µ-law algorithm formula: ((mantissa << 4) + 0x08) << exponent
        magnitude = ((mantissa << 4) + 0x08) << exponent  # Reconstruct amplitude
        sample = magnitude - BIAS  # Remove bias

        # Apply sign
        if sign != 0:
            sample = -sample  # Apply sign (negative if sign bit is set)

        table[i] = sample  # Store in lookup table

    return table


# Global: Precomputed decode table for µ-law to PCM conversion
_MULAW_DECODE_TABLE = _generate_mulaw_to_pcm16_table()


def is_silent_mulaw_audio(
    mulaw_audio_bytes: bytes,
    max_amplitude_threshold: int = AUDIO_SILENCE_THRESHOLDS["MAX_AMPLITUDE"],
    min_rms_dbfs_threshold: float = AUDIO_SILENCE_THRESHOLDS["MIN_RMS_DBFS"],
) -> bool:
    """
    Determines if a µ-law encoded audio chunk is considered silent.
    This is useful for real-time media streams like Twilio.

    Args:
        mulaw_audio_bytes (bytes): 8-bit µ-law encoded audio data.
        max_amplitude_threshold (int): Max amplitude value allowed before treating as speech.
        min_rms_dbfs_threshold (float): Min RMS energy (in dBFS) below which audio is considered silent.

    Returns:
        bool: True if the chunk is silent, False otherwise.
    """
    if not mulaw_audio_bytes:
        print("[DEBUG] Received empty µ-law audio bytes, treating as silent.")
        return True  # No data = silent

    # Convert byte stream to numpy array (uint8)
    mulaw_array = np.frombuffer(mulaw_audio_bytes, dtype=np.uint8)

    # Decode to PCM16 using lookup table
    pcm_samples = _MULAW_DECODE_TABLE[mulaw_array]

    # Step 1: Check if max amplitude exceeds threshold
    max_amp = np.max(np.abs(pcm_samples))
    if max_amp > max_amplitude_threshold:
        print(f"[DEBUG] Non-silent: Amplitude {max_amp} > threshold {max_amplitude_threshold}")
        return False

    # Step 2: Compute RMS energy
    pcm_float = pcm_samples.astype(np.float32)
    rms = np.sqrt(np.mean(pcm_float ** 2))

    if rms == 0:
        return True  # Completely flat audio

    # Step 3: Convert RMS to dBFS (decibels relative to full-scale)
    dbfs = 20 * np.log10(rms / 32768.0)  # Max PCM16 = 32768

    if dbfs >= min_rms_dbfs_threshold:
        print(f"[DEBUG] Non-silent: RMS={rms:.2f}, dBFS={dbfs:.2f}, Threshold={min_rms_dbfs_threshold} dBFS")

    return dbfs < min_rms_dbfs_threshold


def convert_mulaw_to_wav(call_sid: str, mulaw_audio_bytes: bytes) -> str:
    """
    Saves a chunk of µ-law audio and converts it to 16-bit PCM WAV using ffmpeg.

    Args:
        call_sid (str): Unique call/session identifier.
        mulaw_audio_bytes (bytes): µ-law encoded audio from Twilio or another source.

    Returns:
        str: Path to the converted WAV file, or None on failure.
    """
    # Generate unique filenames
    raw_file_path = os.path.join(AUDIO_CHUNK_DIR, f"{call_sid}_{uuid.uuid4()}.raw")
    wav_file_path = raw_file_path.replace(".raw", ".wav")

    # Save µ-law raw audio
    with open(raw_file_path, "wb") as raw_file:
        raw_file.write(mulaw_audio_bytes)

    # Convert to WAV using ffmpeg
    try:
        subprocess.run([
            "ffmpeg",
            "-f", "mulaw",           # Input format: µ-law
            "-ar", "8000",           # Input sample rate
            "-ac", "1",              # input channels: Mono
            "-i", raw_file_path,     # Input file
            "-ar", "16000",          # output sample rate, Resample for ASR (e.g., Whisper)
            "-ac", "1",
            "-c:a", "pcm_s16le",     # 16-bit PCM little-endian
            wav_file_path            # output file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[ERROR] Audio conversion failed: {e}")
        return None
    finally:
        # Cleanup raw input file
        if os.path.exists(raw_file_path):
            os.remove(raw_file_path)

    return wav_file_path


def convert_wav_to_mulaw(wav_path: str, output_dir: str = RESPONSE_AUDIO_CHUNK_DIR) -> bytes:
    """
    Converts a WAV file to μ-law encoded PCM format using ffmpeg.

    Args:
        wav_path (str): Path to the input WAV file.
        output_dir (str): Directory to save μ-law raw file (default is same as input).

    Returns:
        bytes: μ-law audio content as byte stream.
    """
    if output_dir is None:
        output_dir = os.path.dirname(wav_path)

    raw_path = wav_path.replace(".wav", ".raw")
    
    try:
        subprocess.run([
                "ffmpeg",
                "-i", wav_path,           # input WAV file
                "-ar", "8000",            # output sample rate (8kHz for Twilio)
                "-ac", "1",               # mono
                "-acodec", "pcm_mulaw",   # explicit μ-law codec
                "-f", "mulaw",            # output format μ-law
                "-y",                     # overwrite output
                raw_path                  # output file
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        with open(raw_path, "rb") as f:
            audio_data = f.read()

        return audio_data

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg μ-law conversion failed: {e}")
        return None

    finally:
        # Clean up
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(raw_path):
            os.remove(raw_path)


async def convert_wav_chunk_bytes_to_mulaw(wav_chunk: bytes) -> bytes:
    """
    Convert a raw WAV byte chunk into μ-law format.

    Args:
        wav_chunk (bytes): Raw bytes of a WAV file.

    Returns:
        bytes: μ-law encoded bytes.
    """
    try:
        temp_dir = os.path.join(RESPONSE_AUDIO_CHUNK_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        unique_id = uuid.uuid4()
        temp_wav_path = os.path.join(temp_dir, f"temp_{unique_id}.wav")
        temp_raw_path = temp_wav_path.replace(".wav", ".raw")

        # Write the input WAV bytes
        with open(temp_wav_path, "wb") as f:
            f.write(wav_chunk)

        # Convert to μ-law using ffmpeg
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i", temp_wav_path,       # input WAV file
            "-ar", "8000",            # output sample rate (8kHz for Twilio)
            "-ac", "1",               # mono
            "-acodec", "pcm_mulaw",   # explicit μ-law codec
            "-af", "highpass=f=200,lowpass=f=3400",  # Apply audio filters to reduce noise
            "-f", "mulaw",            # output format μ-law
            "-y",                     # overwrite output
            temp_raw_path,            # output file
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )

        await process.wait()

        # Read and return μ-law data
        if os.path.exists(temp_raw_path):
            with open(temp_raw_path, "rb") as f:
                return f.read()
            
            # Clean up temporary files
            os.remove(temp_raw_path)
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            
        else:
            print(f"[ERROR] μ-law output not created: {temp_raw_path}")
            return None

    except Exception as e:
        print(f"[ERROR] WAV chunk μ-law conversion failed: {e}")
        return None


def mulaw_to_pcm(audio_bytes: bytes) -> np.ndarray:
    """
    Converts 8-bit µ-law audio bytes to 16-bit PCM using a precomputed lookup table.
    """
    if not audio_bytes:
        return np.array([], dtype=np.int16)

    mulaw_array = np.frombuffer(audio_bytes, dtype=np.uint8)
    pcm_samples = _MULAW_DECODE_TABLE[mulaw_array]
    return pcm_samples


def is_voiced(audio_bytes: bytes, sample_rate: int = 8000, frame_duration_ms: int = 30) -> bool:
    """
    Determine if the given μ-law audio contains speech using WebRTC VAD.
    - audio_bytes: raw μ-law audio bytes (e.g., from Twilio)
    - sample_rate: usually 8000 Hz for μ-law
    - frame_duration_ms: must be 10, 20, or 30
    """
    try:
        pcm_samples = mulaw_to_pcm(audio_bytes)
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        # print("[DEBUG] Frame size for VAD:", frame_size)

        if len(pcm_samples) < frame_size:
            return False  # not enough data

        frame = pcm_samples[:frame_size].tobytes()
        # print("[DEBUG] Frame size in bytes for VAD:", len(frame))
        # print("[DEBUG] vad.is_speech(frame, sample_rate):", vad.is_speech(frame, sample_rate))  # Print first 20 bytes for debugging
        return vad.is_speech(frame, sample_rate)

    except Exception as e:
        print(f"[VAD Error]: {e}")
        return False

