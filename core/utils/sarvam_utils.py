import asyncio
import base64
import os
import subprocess
import tempfile
import uuid
import logging
from sarvamai import AsyncSarvamAI, AudioOutput, SarvamAI, EventResponse
from sarvamai.play import save

from aiVoiceAssistant.constants import RESPONSE_AUDIO_CHUNK_DIR
from aiVoiceAssistant.settings import (
    SARVAM_LANGUAGE,
    SARVAM_PACE,
    SARVAM_PITCH,
    SARVAM_STT_MODEL,
    SARVAM_SUBSCRIPTION_KEY,
    SARVAM_VOICE,
    SARVAM_TTS_MODEL,
)

logger = logging.getLogger(__name__)

client = SarvamAI(api_subscription_key=SARVAM_SUBSCRIPTION_KEY)


async def synthesize_mulaw_sarvam_tts(text: str, voice: str = SARVAM_VOICE, lang: str = SARVAM_LANGUAGE):
    try:
        audio_response = client.text_to_speech.convert(
            text=text,
            model=SARVAM_TTS_MODEL,  # type: ignore
            target_language_code=lang,
            speech_sample_rate=8000,
            speaker=voice,
            enable_preprocessing=True,
            pace=SARVAM_PACE,
            pitch=SARVAM_PITCH,
        )

        if not audio_response or not hasattr(audio_response, "audios") or not audio_response.audios:
            logger.warning("[Sarvam TTS]: No audio content received.")
            return b""

        # ✅ Ensure directory exists
        os.makedirs(RESPONSE_AUDIO_CHUNK_DIR, exist_ok=True)

        # Decode the first audio chunk (base64)
        wav_bytes = base64.b64decode(audio_response.audios[0])

        # Save WAV
        wav_path = os.path.join(RESPONSE_AUDIO_CHUNK_DIR, f"sarvam_tts_{uuid.uuid4()}.wav")
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)

        # Convert to μ-law
        mulaw_path = wav_path.replace(".wav", ".raw")

        subprocess.run(
            [
                "ffmpeg", "-i", wav_path,
                "-ar", "8000", "-ac", "1",
                "-acodec", "pcm_mulaw", "-f", "mulaw", "-y", mulaw_path
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        with open(mulaw_path, "rb") as f:
            audio_data = f.read()

        # Clean up
        os.remove(wav_path)
        os.remove(mulaw_path)

        return audio_data

    except subprocess.CalledProcessError as ffmpeg_err:
        logger.error(f"[Sarvam TTS]: FFmpeg conversion failed: {ffmpeg_err}")
        return b""

    except Exception as e:
        logger.error(f"[Sarvam TTS]: Error generating TTS audio: {e}")
        return b""


def detect_text_language(text):
    response = client.text.identify_language(input=text)
    return response.language_code


def transcribe_audio_sarvam(filepath, lang=SARVAM_LANGUAGE):
    logger.info(f"SARVAM_STT_MODEL---: {SARVAM_STT_MODEL}")
    response = client.speech_to_text.transcribe(
        file=open(filepath, "rb"),
        model=SARVAM_STT_MODEL,
        language_code=lang,
    )
    logger.info(f"sarvam stt response--: {response.transcript}")
    return response.transcript


async def transcribe_stream_sarvam(filepath: str, language_code: str = SARVAM_LANGUAGE):
    """
    Streams audio data to SarvamAI's real-time STT service using WebSocket.
    Returns transcription result.
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"[Sarvam STT Stream]: File does not exist: {filepath}")
            return ""

        with open(filepath, "rb") as f:
            audio_bytes = f.read()

        client = AsyncSarvamAI(api_subscription_key=SARVAM_SUBSCRIPTION_KEY)

        async with client.speech_to_text_streaming.connect(language_code=language_code) as ws:
            await ws.transcribe(audio=audio_bytes)  # type: ignore
            logger.info("[Sarvam STT Stream]: Audio data sent for transcription")

            response = await ws.recv()
            logger.info(f"[Sarvam STT Stream]: Received response: {response}")

            return getattr(response, "transcript", "")

    except Exception as e:
        logger.error(f"[Sarvam STT Stream]: Error occurred: {e}")
        return ""


async def synthesize_streaming_sarvam_tts(
    text: str,
    voice: str = SARVAM_VOICE,
    lang: str = SARVAM_LANGUAGE,
):
    """
    ✅ Optimized + Debug-friendly version
    Streams μ-law encoded audio chunks from Sarvam AI TTS API.
    Includes detailed logs and inline comments for step-by-step clarity.
    """

    # Initialize async SarvamAI client
    async_client = AsyncSarvamAI(api_subscription_key=SARVAM_SUBSCRIPTION_KEY)
    logger.info(f"[TTS Streaming] Initialized SarvamAI client for model={SARVAM_TTS_MODEL}")

    try:
        # --- Connect to Sarvam streaming TTS API ---
        async with async_client.text_to_speech_streaming.connect(
            model=SARVAM_TTS_MODEL, # type: ignore
            send_completion_event=True,  # ensures we get a "final" event at the end
        ) as ws:
            logger.info("[TTS Streaming] Connected to Sarvam TTS streaming WebSocket")

            # --- Configure voice + audio parameters ---
            await ws.configure(
                target_language_code=lang,
                speaker=voice,
                pitch=SARVAM_PITCH,
                pace=SARVAM_PACE,
                speech_sample_rate=8000,           # μ-law requires 8kHz mono
                enable_preprocessing=True,
                output_audio_codec="mp3",          # stream compressed mp3
                output_audio_bitrate="64k",
                min_buffer_size=30,
                max_chunk_length=100,
            )
            logger.debug(f"[TTS Streaming] Configured voice={voice}, lang={lang}")

            # --- Send text for synthesis ---
            await ws.convert(text)
            await ws.flush()
            logger.info(f"[TTS Streaming] Sent text for conversion: {text[:60]}{'...' if len(text)>60 else ''}")

            # --- Start receiving streaming messages (audio + events) ---
            async for message in ws:
                logger.debug(f"[TTS Streaming] Received message type={type(message).__name__}")

                # --- AUDIO CHUNK HANDLING ---
                if isinstance(message, AudioOutput):
                    logger.debug(f"[TTS Streaming] Audio chunk meta: {message.data.content_type}, base64_length={len(message.data.audio)}")

                    # Decode base64 → MP3 bytes
                    mp3_bytes = base64.b64decode(message.data.audio)
                    if not mp3_bytes:
                        logger.warning("[TTS Streaming] Empty audio chunk received, skipping...")
                        continue

                    # Create temporary MP3 and RAW (μ-law) paths
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_in:
                        tmp_in.write(mp3_bytes)
                        tmp_in.flush()
                        tmp_out = tmp_in.name.replace(".mp3", ".raw")

                    try:
                        # --- Convert MP3 → μ-law PCM (8kHz mono) using FFmpeg ---
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-i", tmp_in.name,
                                "-ar", "8000",  # resample to 8kHz
                                "-ac", "1",     # mono
                                "-acodec", "pcm_mulaw",  # μ-law encoding
                                "-f", "mulaw",
                                "-y", tmp_out,
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )

                        # Read the μ-law-encoded audio bytes
                        with open(tmp_out, "rb") as f:
                            mulaw_audio = f.read()

                        if mulaw_audio:
                            logger.debug(f"[TTS Streaming] Yielding μ-law chunk ({len(mulaw_audio)} bytes)")
                            yield mulaw_audio
                        else:
                            logger.warning("[TTS Streaming] μ-law file empty after FFmpeg conversion")

                    except subprocess.CalledProcessError as e:
                        logger.error(f"[TTS Streaming] FFmpeg conversion failed: {e}")
                    finally:
                        # Cleanup temporary files
                        for fpath in (tmp_in.name, tmp_out):
                            if os.path.exists(fpath):
                                os.remove(fpath)
                                logger.info(f"[TTS Streaming] Cleaned up temp file: {fpath}")

                # --- EVENT MESSAGE HANDLING ---
                elif isinstance(message, EventResponse):
                    event_type = getattr(message.data, "event_type", "unknown")
                    logger.info(f"[TTS Streaming] Received event: {event_type}")

                    # Sarvam sends a "final" event at the end of streaming
                    if event_type == "final":
                        logger.info("[TTS Streaming] Final event received — TTS stream complete")
                        break

                # --- ERROR MESSAGE HANDLING ---
                elif hasattr(message, "error"):
                    logger.error(f"[TTS Streaming] Error from Sarvam API: {message.error}")
                    break

                # --- FALLBACK: unexpected message types ---
                else:
                    logger.warning(f"[TTS Streaming] Unrecognized message: {message}")
                    continue

            # --- Stream end ---
            logger.info("[TTS Streaming] Streaming session completed successfully.")
            yield None  # 🔔 Signal completion to consumer

    except Exception as e:
        logger.exception(f"[TTS Streaming] Streaming error occurred: {e}")
        yield None
