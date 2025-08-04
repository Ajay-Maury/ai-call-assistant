import asyncio
import base64
import logging
import os
import io
import subprocess
import time
import uuid
from sarvamai import AsyncSarvamAI, SarvamAI
from sarvamai.play import save

from aiVoiceAssistant.constants import RESPONSE_AUDIO_CHUNK_DIR
from aiVoiceAssistant.settings import SARVAM_LANGUAGE, SARVAM_PACE, SARVAM_STT_MODEL, SARVAM_SUBSCRIPTION_KEY, SARVAM_VOICE, SARVAM_TTS_MODEL

logger = logging.getLogger(__name__)

# Initialize sync client once
client = SarvamAI(api_subscription_key=SARVAM_SUBSCRIPTION_KEY)


TTS_WARMUP_TEXT = "Initializing voice synthesis."

_warmed_up = False  # Module-level flag

async def warmup_tts():
    global _warmed_up
    if _warmed_up:
        return
    try:
        logger.info("[TTS Warmup] Starting TTS warmup...")
        start = time.time()
        await synthesize_mulaw_sarvam_tts(TTS_WARMUP_TEXT)
        duration = time.time() - start
        logger.info(f"[TTS Warmup] Completed in {duration:.2f}s.")
        _warmed_up = True
    except Exception as e:
        logger.warning(f"[TTS Warmup] Failed: {e}")

# async def synthesize_mulaw_sarvam_tts(text: str, voice: str = SARVAM_VOICE, lang: str = SARVAM_LANGUAGE):
#     try:
#         logger.info(f"[Sarvam TTS] Starting synthesis for text: '{text}'")

#         audio_response = client.text_to_speech.convert(
#             text=text,
#             model=SARVAM_TTS_MODEL,    # type: ignore
#             target_language_code=lang,
#             speech_sample_rate=8000,
#             speaker=voice,
#             enable_preprocessing=True,
#             pace=SARVAM_PACE
#         )

#         if not audio_response or not hasattr(audio_response, "audios") or not audio_response.audios:
#             logger.warning("[Sarvam TTS] No audio content received from API.")
#             return b""

#         logger.debug("[Sarvam TTS] Ensuring audio chunk directory exists.")
#         os.makedirs(RESPONSE_AUDIO_CHUNK_DIR, exist_ok=True)

#         # Decode and save WAV
#         wav_bytes = base64.b64decode(audio_response.audios[0])
#         wav_path = os.path.join(RESPONSE_AUDIO_CHUNK_DIR, f"sarvam_tts_{uuid.uuid4()}.wav")
#         with open(wav_path, "wb") as f:
#             f.write(wav_bytes)
#         logger.info(f"[Sarvam TTS] WAV audio saved at: {wav_path}")

#         # Convert to μ-law format
#         mulaw_path = wav_path.replace(".wav", ".raw")
#         logger.debug(f"[Sarvam TTS] Converting WAV to μ-law format: {mulaw_path}")

#         subprocess.run([
#             "ffmpeg", "-i", wav_path,
#             "-ar", "8000", "-ac", "1",
#             "-acodec", "pcm_mulaw", "-f", "mulaw", "-y", mulaw_path
#         ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         logger.info(f"[Sarvam TTS] Conversion to μ-law successful: {mulaw_path}")

#         with open(mulaw_path, "rb") as f:
#             audio_data = f.read()

#         # Cleanup
#         os.remove(wav_path)
#         os.remove(mulaw_path)
#         logger.debug("[Sarvam TTS] Temporary files cleaned up.")

#         return audio_data

#     except subprocess.CalledProcessError as ffmpeg_err:
#         logger.error(f"[Sarvam TTS] FFmpeg conversion failed: {ffmpeg_err}")
#         return b""

#     except Exception as e:
#         logger.exception(f"[Sarvam TTS] Unexpected error during synthesis: {e}")
#         return b""



# # ✅ Async TTS synthesis with in-memory μ-law conversion and profiling
# async def synthesize_mulaw_sarvam_tts(text: str, voice: str = SARVAM_VOICE, lang: str = SARVAM_LANGUAGE) -> bytes:
#     try:
#         start_total = time.perf_counter()
#         logger.info(f"[Sarvam TTS] Synthesizing audio for: '{text}'")

#         start_tts = time.perf_counter()
#         audio_response = client.text_to_speech.convert(
#             text=text,
#             model=SARVAM_TTS_MODEL,
#             target_language_code=lang,
#             speech_sample_rate=8000,
#             speaker=voice,
#             enable_preprocessing=True,
#             pace=SARVAM_PACE
#         )
#         end_tts = time.perf_counter()
#         logger.info(f"[Sarvam TTS] TTS API time: {end_tts - start_tts:.2f}s")

#         if not audio_response or not hasattr(audio_response, "audios") or not audio_response.audios:
#             logger.warning("[Sarvam TTS] No audio content received.")
#             return b""

#         start_decode = time.perf_counter()
#         wav_bytes = base64.b64decode(audio_response.audios[0])
#         wav_buffer = io.BytesIO(wav_bytes)
#         end_decode = time.perf_counter()
#         logger.info(f"[Sarvam TTS] Base64 decode time: {end_decode - start_decode:.2f}s")

#         start_ffmpeg = time.perf_counter()
#         process = subprocess.run([
#             "ffmpeg", "-f", "wav", "-i", "pipe:0",
#             "-ar", "8000", "-ac", "1",
#             "-acodec", "pcm_mulaw", "-f", "mulaw", "pipe:1"
#         ],
#             input=wav_buffer.read(),
#             stdout=subprocess.PIPE,
#             stderr=subprocess.DEVNULL,
#             check=True
#         )
#         end_ffmpeg = time.perf_counter()
#         logger.info(f"[Sarvam TTS] FFmpeg μ-law conversion time: {end_ffmpeg - start_ffmpeg:.2f}s")

#         total_time = time.perf_counter() - start_total
#         logger.info(f"[Sarvam TTS] Total synthesis time: {total_time:.2f}s")

#         return process.stdout

#     except subprocess.CalledProcessError as ffmpeg_err:
#         logger.error(f"[Sarvam TTS] FFmpeg conversion failed: {ffmpeg_err}")
#         return b""
#     except Exception as e:
#         logger.exception(f"[Sarvam TTS] Error generating TTS audio: {e}")
#         return b""


# ✅ Detect language using Sarvam
def detect_text_language(text):
    try:
        logger.info(f"[Sarvam LangDetect] Detecting language for: '{text}'")
        response = client.text.identify_language(input=text)
        logger.info(f"[Sarvam LangDetect] Detected: {response.language_code}")
        return response.language_code
    except Exception as e:
        logger.exception(f"[Sarvam LangDetect] Error: {e}")
        return SARVAM_LANGUAGE


# ✅ Sync STT (file-based)
def transcribe_audio_sarvam(filepath, lang=SARVAM_LANGUAGE):
    try:
        if not os.path.exists(filepath):
            logger.warning(f"[Sarvam STT] File not found: {filepath}")
            return ""
        
        logger.info(f"[Sarvam STT] Transcribing file: {filepath}")
        with open(filepath, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model=SARVAM_STT_MODEL,
                language_code=lang
            )
        logger.info(f"[Sarvam STT] Transcript: {response.transcript}")
        return response.transcript
    except Exception as e:
        logger.exception(f"[Sarvam STT] Error during transcription: {e}")
        return ""


# ✅ Async streaming STT (WebSocket)
async def transcribe_stream_sarvam(filepath: str, language_code: str = SARVAM_LANGUAGE):
    try:
        if not os.path.exists(filepath):
            logger.warning(f"[Sarvam STT Stream] File does not exist: {filepath}")
            return ""

        logger.info(f"[Sarvam STT Stream] Starting stream transcription for: {filepath}")
        with open(filepath, "rb") as f:
            audio_bytes = f.read()

        client = AsyncSarvamAI(api_subscription_key=SARVAM_SUBSCRIPTION_KEY)

        async with client.speech_to_text_streaming.connect(language_code=language_code) as ws:
            await ws.transcribe(audio=audio_bytes)  # type: ignore
            logger.debug("[Sarvam STT Stream] Audio sent for transcription.")

            response = await ws.recv()
            transcript = getattr(response, "transcript", "")
            logger.info(f"[Sarvam STT Stream] Received transcript: {transcript}")
            return transcript

    except Exception as e:
        logger.exception(f"[Sarvam STT Stream] Error occurred: {e}")
        return ""
    

async def synthesize_chunks_streaming(chunks, concurrency=3):
    """
    Async generator that yields audio buffers as they are ready.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_task(chunk, idx):
        async with semaphore:
            buffer = await synthesize_mulaw_sarvam_tts(chunk)
            return idx, buffer

    tasks = [limited_task(chunk, idx) for idx, chunk in enumerate(chunks)]
    pending = asyncio.as_completed(tasks)

    buffer_map = {}
    next_index = 0

    async for coro in pending:
        idx, buffer = await coro
        buffer_map[idx] = buffer

        # Yield in order
        while next_index in buffer_map:
            yield buffer_map.pop(next_index)
            next_index += 1


def synthesize_mulaw_sarvam_tts_blocking(text: str, voice: str = SARVAM_VOICE, lang: str = SARVAM_LANGUAGE) -> bytes:
    try:
        start_total = time.perf_counter()
        logger.info(f"[Sarvam TTS] Synthesizing audio for: '{text}'")

        start_tts = time.perf_counter()
        audio_response = client.text_to_speech.convert(
            text=text,
            model=SARVAM_TTS_MODEL,
            target_language_code=lang,
            speech_sample_rate=8000,
            speaker=voice,
            enable_preprocessing=True,
            pace=SARVAM_PACE
        )
        end_tts = time.perf_counter()
        logger.info(f"[Sarvam TTS] TTS API time: {end_tts - start_tts:.2f}s")

        if not audio_response or not hasattr(audio_response, "audios") or not audio_response.audios:
            logger.warning("[Sarvam TTS] No audio content received.")
            return b""

        start_decode = time.perf_counter()
        wav_bytes = base64.b64decode(audio_response.audios[0])
        wav_buffer = io.BytesIO(wav_bytes)
        end_decode = time.perf_counter()
        logger.info(f"[Sarvam TTS] Base64 decode time: {end_decode - start_decode:.2f}s")

        start_ffmpeg = time.perf_counter()
        process = subprocess.run([
            "ffmpeg", "-f", "wav", "-i", "pipe:0",
            "-ar", "8000", "-ac", "1",
            "-acodec", "pcm_mulaw", "-f", "mulaw", "pipe:1"
        ],
            input=wav_buffer.read(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True
        )
        end_ffmpeg = time.perf_counter()
        logger.info(f"[Sarvam TTS] FFmpeg μ-law conversion time: {end_ffmpeg - start_ffmpeg:.2f}s")

        total_time = time.perf_counter() - start_total
        logger.info(f"[Sarvam TTS] Total synthesis time: {total_time:.2f}s")

        return process.stdout

    except subprocess.CalledProcessError as ffmpeg_err:
        logger.error(f"[Sarvam TTS] FFmpeg conversion failed: {ffmpeg_err}")
        return b""
    except Exception as e:
        logger.exception(f"[Sarvam TTS] Error generating TTS audio: {e}")
        return b""


async def synthesize_mulaw_sarvam_tts(text: str, voice: str = SARVAM_VOICE, lang: str = SARVAM_LANGUAGE) -> bytes:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, synthesize_mulaw_sarvam_tts_blocking, text, voice, lang)
