import base64
import os
import subprocess
import uuid
from sarvamai import AsyncSarvamAI, SarvamAI
from sarvamai.play import save

from aiVoiceAssistant.constants import RESPONSE_AUDIO_CHUNK_DIR
from aiVoiceAssistant.settings import SARVAM_LANGUAGE, SARVAM_PACE, SARVAM_PITCH, SARVAM_STT_MODEL, SARVAM_SUBSCRIPTION_KEY, SARVAM_VOICE, SARVAM_TTS_MODEL


client = SarvamAI(api_subscription_key=SARVAM_SUBSCRIPTION_KEY)


async def synthesize_mulaw_sarvam_tts(text: str, voice: str = SARVAM_VOICE, lang: str = SARVAM_LANGUAGE):
    try:
        audio_response = client.text_to_speech.convert(
            text=text,
            model=SARVAM_TTS_MODEL,    # type: ignore
            target_language_code=lang,
            speech_sample_rate=8000,
            speaker=voice,
            enable_preprocessing=True,
            pace=SARVAM_PACE,
            pitch=SARVAM_PITCH,
        )

        if not audio_response or not hasattr(audio_response, "audios") or not audio_response.audios:
            print("[Sarvam TTS]: No audio content received.")
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

        subprocess.run([
            "ffmpeg", "-i", wav_path,
            "-ar", "8000", "-ac", "1",
            "-acodec", "pcm_mulaw", "-f", "mulaw", "-y", mulaw_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(mulaw_path, "rb") as f:
            audio_data = f.read()

        # Clean up
        os.remove(wav_path)
        os.remove(mulaw_path)

        return audio_data

    except subprocess.CalledProcessError as ffmpeg_err:
        print(f"[Sarvam TTS]: FFmpeg conversion failed: {ffmpeg_err}")
        return b""

    except Exception as e:
        print(f"[Sarvam TTS]: Error generating TTS audio: {e}")
        return b""


def detect_text_language(text):
    response = client.text.identify_language(
        input=text
    )
    return response.language_code


def transcribe_audio_sarvam(filepath, lang=SARVAM_LANGUAGE):
    print(f"SARVAM_STT_MODEL---: {SARVAM_STT_MODEL}")
    response = client.speech_to_text.transcribe(
        file=open(filepath, "rb"),
        model=SARVAM_STT_MODEL,
        language_code=lang
    )
    print(f"sarvam stt response--: {response.transcript}")
    return response.transcript


async def transcribe_stream_sarvam(filepath: str, language_code: str = SARVAM_LANGUAGE):
    """
    Streams audio data to SarvamAI's real-time STT service using WebSocket.
    Returns transcription result.
    """
    try:
        # Load audio file
        if not os.path.exists(filepath):
            print(f"[Sarvam STT Stream]: File does not exist: {filepath}")
            return ""

        with open(filepath, "rb") as f:
            audio_bytes = f.read()

        client = AsyncSarvamAI(api_subscription_key=SARVAM_SUBSCRIPTION_KEY)

        # Establish WebSocket connection for streaming STT
        async with client.speech_to_text_streaming.connect(language_code=language_code) as ws:
            await ws.transcribe(audio=audio_bytes)      # type: ignore
            print("[Sarvam STT Stream]: Audio data sent for transcription")

            response = await ws.recv()
            print(f"[Sarvam STT Stream]: Received response: {response}")

            return getattr(response, "transcript", "")

    except Exception as e:
        print(f"[Sarvam STT Stream]: Error occurred: {e}")
        return ""

 
async def synthesize_streaming_sarvam_tts(text: str, voice: str = SARVAM_VOICE, lang: str = SARVAM_LANGUAGE):
    """
    Stream audio chunks from Sarvam AI's streaming TTS API.
    Yields μ-law encoded audio chunks as they become available.
    """
    try:
        async_client = AsyncSarvamAI(api_subscription_key=SARVAM_SUBSCRIPTION_KEY)

        # Establish WebSocket connection for streaming TTS
        async with async_client.text_to_speech_streaming.connect(model=SARVAM_TTS_MODEL) as ws:
            # Configure the streaming TTS settings
            await ws.configure(
                target_language_code=lang,
                speaker=voice,
                pitch=SARVAM_PITCH,
                pace=SARVAM_PACE,
                min_buffer_size=10,  # Start processing with fewer characters
                max_chunk_length=100,  # Keep chunks small for real-time feel
                output_audio_codec="wav",
                output_audio_bitrate="8000"
            )

            # Send text for conversion
            await ws.convert(text)
            await ws.flush()

            # Process streaming audio chunks
            async for message in ws:
                print(f"[Streaming TTS]: Received message type: {type(message)}")
                if hasattr(message, 'audio') and message.audio:
                    print(f"[Streaming TTS]: Processing audio chunk of size: {len(message.audio)}")
                    # Decode base64 audio data
                    wav_bytes = base64.b64decode(message.audio)

                    # Save WAV temporarily
                    temp_wav_path = os.path.join(RESPONSE_AUDIO_CHUNK_DIR, f"stream_tts_{uuid.uuid4().hex[:8]}.wav")
                    with open(temp_wav_path, "wb") as f:
                        f.write(wav_bytes)

                    # Convert to μ-law
                    mulaw_path = temp_wav_path.replace(".wav", ".raw")

                    try:
                        subprocess.run([
                            "ffmpeg", "-i", temp_wav_path,
                            "-ar", "8000", "-ac", "1",
                            "-acodec", "pcm_mulaw", "-f", "mulaw", "-y", mulaw_path
                        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                        # Read μ-law audio data
                        with open(mulaw_path, "rb") as f:
                            mulaw_audio = f.read()

                        print(f"[Streaming TTS]: Converted to μ-law, size: {len(mulaw_audio)} bytes")

                        # Cleanup temporary files
                        os.remove(temp_wav_path)
                        os.remove(mulaw_path)

                        # Yield the μ-law audio chunk
                        if mulaw_audio:
                            yield mulaw_audio

                    except subprocess.CalledProcessError as ffmpeg_err:
                        print(f"[Streaming TTS]: FFmpeg conversion failed: {ffmpeg_err}")
                        # Cleanup on error
                        if os.path.exists(temp_wav_path):
                            os.remove(temp_wav_path)
                        if os.path.exists(mulaw_path):
                            os.remove(mulaw_path)
                        continue

                elif hasattr(message, 'error'):
                    print(f"[Streaming TTS]: Error from Sarvam: {message.error}")
                    break
                else:
                    print(f"[Streaming TTS]: Received message without audio: {message}")
                    # Check if streaming is complete
                    if hasattr(message, 'is_final') and message.is_final:
                        print(f"[Streaming TTS]: Streaming complete")
                        break

    except Exception as e:
        print(f"[Streaming TTS]: Error in streaming TTS: {e}")
        return
