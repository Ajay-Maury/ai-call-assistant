import asyncio
import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Optional

import webrtcvad
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from aiVoiceAssistant.constants import AUDIO_CACHE_DIR, AUDIO_CHUNK_SIZE, DISENGAGEMENT_BACKCHANNEL_REPEAT_DELAY, DISENGAGEMENT_TRIGGER_SECONDS, ENGAGEMENT_BACKCHANNEL_REPEAT_DELAY, ENGAGEMENT_TRIGGER_SECONDS, INITIAL_GREETING_TEXT, AUDIO_SILENCE_THRESHOLDS, MIN_AUDIO_BYTES, SILENCE_MAX_DURATION, AUDIO_BUFFER_SILENCE, VAD_FRAME_SIZE, VAD_MIN_SILENCE_FRAMES

from core.utils.helper_utils import get_engagement_response
from core.utils.langchain_agent import LangChainAIAgent
from .utils.audio_utils import (convert_mulaw_to_wav, is_silent_mulaw_audio,
                                is_voiced)
from .utils.open_ai_utils import (get_ai_response, is_user_engagement,
                                  transcribe_audio_whisper_groq)
from .utils.redis_utils import get_context, store_context
from .utils.sarvam_utils import synthesize_mulaw_sarvam_tts, synthesize_streaming_sarvam_tts

logger = logging.getLogger(__name__)
vad = webrtcvad.Vad(2)

class VoiceStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        logger.info("✅ [Connect] WebSocket client connected.")

        self.call_sid = None
        self.stream_sid = None
        self.raw_buffer = b""
        self.stop_event = asyncio.Event()
        self.last_audio_time = asyncio.get_event_loop().time()
        self.speech_start_time = 0
        self.engagement_tts_task = None
        self.engagement_tts_stop_event = asyncio.Event()
        self.response_tts_task = None
        self.response_tts_stop_event = asyncio.Event()
        self.engagement_mark_name = None
        self.response_mark_name = None
        self.barge_in_detected = False
        self.silence_task = None
        self.engagement_task = None
        self.engaged_once = False
        self.disengaged_once = False
        self.engagement_state = "unknown"  # new: manage state
        self.last_engaged_time = 0
        self.last_disengaged_time = 0
        logger.info("✅ [Connect] Initial state variables set.")

    async def disconnect(self, close_code):
        logger.info(f"🛑 [Disconnect-{self.call_sid}] Client disconnected with code: {close_code}. Performing cleanup...")
        self.stop_event.set()
        await self._cleanup_tasks()
        logger.info(f"🛑 [Disconnect-{self.call_sid}] Cleanup complete.")

    async def receive(self, text_data):
        if not text_data:
            return

        data = json.loads(text_data)
        event = data.get("event")
        # logger.info(f"➡️ [Receive-{self.call_sid}] Received event: '{event}'")

        if event == "start":
            start_data = data.get("start", {})
            self.call_sid = start_data.get("callSid")
            self.stream_sid = start_data.get("streamSid")
            logger.info(f"➡️ [Start-{self.call_sid}] StreamSid: {self.stream_sid}")
            await self._handle_start_event()

        elif event == "media":
            await self._handle_media_event(data)

        elif event == "mark":
            await self._handle_mark_event(data)

    # 🔽 Event Handlers 🔽

    async def _handle_start_event(self):
        logger.info(f"🚀 [Handler-{self.call_sid}] Handling 'start' event.")
        self.response_tts_task = asyncio.create_task(
            self.stream_cached_or_generate_prompt(
                stop_event=self.response_tts_stop_event,
                greeting_text=INITIAL_GREETING_TEXT,
                cache_subdir="initial_greet_response.ulaw",
                mark_name="initial_greet_response",
                mark_type="response",
            )
        )
        logger.info(f"🚀 [Handler-{self.call_sid}] Initial greeting task created.")
        store_context(self.call_sid, "Hello", INITIAL_GREETING_TEXT)
        logger.info(f"🚀 [Handler-{self.call_sid}] Launching background monitoring tasks...")
        self.silence_task = asyncio.create_task(self.detect_silence_and_respond())
        self.engagement_task = asyncio.create_task(self.monitor_user_engagement())
        self.engagement_task.add_done_callback(self._log_engagement_task_done)

    def _log_engagement_task_done(self, task):
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("💤 Engagement task was cancelled.")
        except Exception as e:
            logger.error(f"💥 Engagement task crashed: {e}", exc_info=True)

    async def _handle_media_event(self, data):
        audio_chunk = base64.b64decode(data["media"]["payload"])
        now = asyncio.get_event_loop().time()
        if not is_silent_mulaw_audio(audio_chunk):
            self.raw_buffer += audio_chunk
            self.last_audio_time = now
            if self.speech_start_time == 0:
                self.speech_start_time = now
        elif len(self.raw_buffer) > 0 and (now - self.last_audio_time) < SILENCE_MAX_DURATION:
            self.raw_buffer += audio_chunk

    async def _handle_mark_event(self, data):
        mark_name = data.get("mark", {}).get("name")
        logger.info(f"📌 [Mark-{self.call_sid}] Received mark: '{mark_name}'")
        if not mark_name:
            return
        self.barge_in_detected = False
        if mark_name.startswith("engagement_response") or mark_name.startswith("disengagement_response"):
            self.engagement_tts_stop_event.set()
            self.engagement_mark_name = mark_name
        elif mark_name.startswith("ai_response"):
            self.response_tts_stop_event.set()
            self.response_mark_name = mark_name
        else:
            self.response_mark_name = self.engagement_mark_name = mark_name
            self.engagement_tts_stop_event.set()
            self.response_tts_stop_event.set()
        logger.info(f"📌 [Mark-{self.call_sid}] State updated. Response Mark: {self.response_mark_name}, Engagement Mark: {self.engagement_mark_name}")

    # 🔽 Ported Logic (now as class methods) 🔽

    async def monitor_user_engagement(self):
        logger.info(f"🧐 [Engagement-{self.call_sid}] Starting user engagement monitor.")
        silent_since = None

        logger.info(f"🧐 [Engagement-{self.call_sid}] Initializing engagement state: {self.engagement_state}, and stop event state:- {self.stop_event.is_set()}")

        while not self.stop_event.is_set():
            await asyncio.sleep(0.5)
            now = asyncio.get_event_loop().time()
            user_is_talking = len(self.raw_buffer) > MIN_AUDIO_BYTES
            tts_active = (self.engagement_tts_task and not self.engagement_tts_task.done()) or \
                         (self.response_tts_task and not self.response_tts_task.done())
            # logger.debug(f"🔁 [Engagement-{self.call_sid}] Loop running. Buffer size: {len(self.raw_buffer)}")
            # logger.debug(f"🔁 [Engagement-{self.call_sid}] User talking: {user_is_talking}, TTS active: {tts_active}")

            if user_is_talking:
                if self.engagement_state != "talking":
                    self.speech_start_time = now
                self.engagement_state = "talking"
                silent_since = None
                if self.speech_start_time == 0:
                    self.speech_start_time = now
                time_speaking = now - self.speech_start_time
                self.last_disengaged_time = 0

                # logger.info(f"🗣️ [Engagement-{self.call_sid}] User is talking. Time speaking: {time_speaking:.2f}s, Buffer size: {len(self.raw_buffer)}")

                if not self.barge_in_detected and not tts_active and \
                   ((not self.engaged_once and time_speaking >= ENGAGEMENT_TRIGGER_SECONDS) or (self.last_engaged_time > 0 and now - self.last_engaged_time >= ENGAGEMENT_BACKCHANNEL_REPEAT_DELAY)):
                    
                    if self.engagement_tts_task and not self.engagement_tts_task.done():
                        self.engagement_tts_task.cancel()
                
                    self.last_engaged_time = now
                    text = get_engagement_response("ENGAGED", "hi")
                    self.engagement_tts_stop_event.clear()
                    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    self.engagement_tts_task = asyncio.create_task(
                        self.stream_cached_or_generate_prompt(
                            stop_event=self.engagement_tts_stop_event,
                            greeting_text=text,
                            cache_subdir=f"engaged_response/{text_hash}.ulaw",
                            mark_name="engagement_response",
                            mark_type="engagement",
                        )
                    )
                    self.engaged_once = True

            elif (self.engagement_mark_name and not self.engagement_mark_name.endswith("tts_complete")) or (self.response_mark_name and not self.response_mark_name.endswith("tts_complete")):
                # logger.info(f"🤫 [Silence-{self.call_sid}] Engagement mark is active, waiting for TTS to complete.")
                # logger.info(f"🤫 [Silence-{self.call_sid}] engagement_mark_name: {self.engagement_mark_name}, response_mark_name: {self.response_mark_name}")
                silent_since = None
                self.last_disengaged_time = 0
                self.engagement_state = "engaged"

            else:
                # logger.info(f"🤫 [Silence-{self.call_sid}] User is silent. Engagement state: {self.engagement_state}"   )
                if self.engagement_state != "silent":
                    silent_since = now
                self.engagement_state = "silent"
                self.speech_start_time = 0
                time_silent = now - silent_since if silent_since else 0
                self.last_engaged_time = 0

                # logger.info(f"🤫 [Silence-{self.call_sid}] User is silent. Time silent: {time_silent:.2f}s, barge_in_detected:- {self.barge_in_detected}, tts_active:- {tts_active}")
                # logger.info(f"disengaged_once:- {self.disengaged_once}, last_disengaged_time:- {self.last_disengaged_time}, now - last_disengaged_time:- {now - self.last_disengaged_time}")

                if not self.barge_in_detected and not tts_active and \
                   ((not self.disengaged_once and time_silent >= DISENGAGEMENT_TRIGGER_SECONDS) or (self.last_disengaged_time > 0 and now - self.last_disengaged_time >= DISENGAGEMENT_BACKCHANNEL_REPEAT_DELAY)):
                    
                    if self.engagement_tts_task and not self.engagement_tts_task.done():
                        self.engagement_tts_task.cancel()
                    
                    text = get_engagement_response("DISENGAGED", "hi")
                    self.engagement_tts_stop_event.clear()
                    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    self.engagement_tts_task = asyncio.create_task(
                        self.stream_cached_or_generate_prompt(
                            stop_event=self.engagement_tts_stop_event,
                            greeting_text=text,
                            cache_subdir=f"disengaged_response/{text_hash}.ulaw",
                            mark_name="disengagement_response",
                            mark_type="engagement",
                        )
                    )
                    self.last_disengaged_time = now
                    self.disengaged_once = True


    async def detect_silence_and_respond(self):
        logger.info(f"🤫 [Silence-{self.call_sid}] Starting silence and barge-in monitor.")
        while not self.stop_event.is_set():
            await asyncio.sleep(0.25)
            logger.debug(f"🤫 [Silence-{self.call_sid}] Checking silence conditions. Raw buffer size: {len(self.raw_buffer)}")
            if len(self.raw_buffer) <= MIN_AUDIO_BYTES:
                continue
            
            now = asyncio.get_event_loop().time()
            elapsed_since_last_audio = now - self.last_audio_time
            tts_active = self.response_tts_task and not self.response_tts_task.done()

            # End of Speech Logic
            frames = [self.raw_buffer[i:i + VAD_FRAME_SIZE] for i in range(0, len(self.raw_buffer), VAD_FRAME_SIZE)]
            silent_frames = sum(1 for frame in frames[-5:] if len(frame) == VAD_FRAME_SIZE and not is_voiced(frame))


            logger.info(f"🤫 [Silence-{self.call_sid}] Raw buffer size: {len(self.raw_buffer)}, Elapsed since last audio: {elapsed_since_last_audio:.2f}s, silent_frames:- {silent_frames}, TTS active: {tts_active}")
            
            # Barge-in Logic
            if tts_active or (self.response_mark_name and not str(self.response_mark_name).endswith("tts_complete")):
                logger.info(f"🤫 [Barge-In-{self.call_sid}] Detected! TTS is active, checking for interruption.")
                self.barge_in_detected = True
                audio_file_path = convert_mulaw_to_wav(self.call_sid, self.raw_buffer)
                user_text = transcribe_audio_whisper_groq(audio_file_path, "hi")
                is_engagement = await is_user_engagement(user_text, self.call_sid, "hi")
                
                if is_engagement:
                    logger.info(f"🤫 [Barge-In-{self.call_sid}] Engagement detected ('{user_text}'), continuing TTS.")
                    self.raw_buffer = b""
                    self.speech_start_time = 0
                    self.barge_in_detected = False
                    continue
                else:
                    logger.info(f"🤫 [Barge-In-{self.call_sid}] User interrupted with '{user_text}'. Stopping TTS.")
                    self.response_tts_stop_event.set()
                    await self.send(text_data=json.dumps({"event": "clear", "streamSid": self.stream_sid}))
                    self.raw_buffer = b""
                    self.speech_start_time = 0
                    self.barge_in_detected = False
                    continue

            if silent_frames >= VAD_MIN_SILENCE_FRAMES or elapsed_since_last_audio > AUDIO_BUFFER_SILENCE:
                logger.info(f"🤫 [Silence-{self.call_sid}] End of speech detected. Transcribing audio of size {len(self.raw_buffer)}...")
                audio_to_process, self.raw_buffer = self.raw_buffer, b""
                
                start_time = time.time()

                logger.info(f"🤫 [Silence-{self.call_sid}] Converting raw audio to WAV format for transcription.")
                audio_file_path = convert_mulaw_to_wav(self.call_sid, audio_to_process)
                
                logger.info(f"[Timing] Received mulaw to wav audio in {time.time() - start_time:.2f}s")
                start_time = time.time()
                
                logger.info(f"🤫 [Silence-{self.call_sid}] Converted audio to WAV at {audio_file_path}.")

                user_text = transcribe_audio_whisper_groq(audio_file_path, "hi")

                logger.info(f"[Timing] Transcribed audio to text: '{user_text}' in {time.time() - start_time:.2f}s")

                self.speech_start_time = 0
                if audio_file_path: os.remove(audio_file_path)

                if user_text:
                    await self.handle_user_transcription(user_text)
                else:
                    logger.info(f"🤫 [Silence-{self.call_sid}] Transcription resulted in empty text.")

    async def handle_user_transcription(self, user_text: str):
        logger.info(f"🤖 [AI-{self.call_sid}] Handling transcription: '{user_text}'")
        try:
            start_time = time.time()

            self.response_mark_name = "ai_response"
            context = get_context(self.call_sid)
            # ai_response = await get_ai_response(user_text, context, self.call_sid)
            # Use streaming AI response with streaming TTS

            logger.info(f"[Timing] Fetched context for response in {time.time() - start_time:.2f}s")
            start_time = time.time()

            ai_response = await self.handle_streaming_ai_response( user_text, context)

            logger.info(f"[Timing] Completed streaming AI response in {time.time() - start_time:.2f}s")

            logger.info(f"🤖 [AI-{self.call_sid}] Generated AI response: '{ai_response}'")
            store_context(self.call_sid, user_text, ai_response)

            if self.response_tts_task and not self.response_tts_task.done():
                logger.info(f"🤖 [AI-{self.call_sid}] Waiting for previous TTS task to complete.")
                await self.wait_for_tts_finish(self.response_tts_task)

            self.response_tts_stop_event.clear()
            # self.response_tts_task = asyncio.create_task(
            #     self.stream_tts_to_client(
            #         text=ai_response,
            #         stop_event=self.response_tts_stop_event,
            #         mark_name="ai_response",
            #         mark_type="response"
            #     )
            # )
        except Exception as e:
            logger.error(f"🤖 [AI-{self.call_sid}] Error in transcription handling: {e}")

    async def stream_cached_or_generate_prompt(self, stop_event, greeting_text, cache_subdir, mark_name, mark_type):
        logger.info(f"🔊 [TTS-{self.call_sid}] Streaming prompt: '{greeting_text}'")
        audio_cache_dir = Path(AUDIO_CACHE_DIR)
        audio_cache_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_cache_dir / cache_subdir
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        if not audio_path.exists():
            logger.info(f"🔊 [TTS-{self.call_sid}] No cache found. Generating new audio.")
            audio_buffer = await self.generate_mulaw_audio_buffer(greeting_text)
            if not audio_buffer: return
            with open(audio_path, "wb") as f:
                f.write(audio_buffer)
        else:
            logger.info(f"🔊 [TTS-{self.call_sid}] Using cached audio from '{audio_path}'.")
            with open(audio_path, "rb") as f:
                audio_buffer = f.read()

        if mark_type == "engagement": self.engagement_mark_name = mark_name
        elif mark_type == "response": self.response_mark_name = mark_name

        await self.play_audio_buffer_to_twilio(audio_buffer, stop_event, mark_name, mark_type)

    # 🔽 Core Helper Methods 🔽

    async def wait_for_tts_finish(self, tts_task):
        try:
            await asyncio.shield(tts_task)
        except asyncio.CancelledError: pass

    async def generate_mulaw_audio_buffer(self, text: str) -> bytes:
        try:
            return await synthesize_mulaw_sarvam_tts(text)
        except Exception as e:
            logger.error(f"⚠️ [TTS Generation Error]: {e}")
            return b""

    async def play_audio_buffer_to_twilio(self, audio_buffer: bytes, stop_event: asyncio.Event, mark_name: Optional[str] = None, mark_type: Optional[str] = None):
        if not audio_buffer: return
        logger.info(f"▶️ [Player-{self.call_sid}] Playing audio buffer of size {len(audio_buffer)}.")
        for i in range(0, len(audio_buffer), AUDIO_CHUNK_SIZE):
            if stop_event.is_set():
                logger.info(f"▶️ [Player-{self.call_sid}] Playback interrupted by stop event.")
                break
            try:
                await self.send(text_data=json.dumps({
                    "event": "media", "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(audio_buffer[i:i+AUDIO_CHUNK_SIZE]).decode()}
                }))
            except Exception as e:
                logger.error(f"⚠️ [Player-{self.call_sid}] WebSocket send error: {e}")
                break
            await asyncio.sleep(0.01)

        if not stop_event.is_set() and mark_name:
            logger.info(f"▶️ [Player-{self.call_sid}] Sending completion mark: '{mark_name}_tts_complete'")
            await self.send(text_data=json.dumps({
                "event": "mark", "streamSid": self.stream_sid,
                "mark": {"name": f"{mark_name}_tts_complete"}
            }))
            # if mark_type == "engagement": self.engagement_mark_name = f"{mark_name}_tts_complete"
            # elif mark_type == "response": self.response_mark_name = f"{mark_name}_tts_complete"

    async def stream_tts_to_client(self, text, stop_event, mark_name=None, mark_type=None):
        audio_buffer = await self.generate_mulaw_audio_buffer(text)
        await self.play_audio_buffer_to_twilio(audio_buffer, stop_event, mark_name, mark_type)


    async def stream_real_time_tts_to_client(self, text, stop_event, mark_name=None):
        """Stream TTS audio using Sarvam AI's streaming API for real-time playback."""
        logger.info(f"[Streaming-TTS-{self.call_sid}]: Starting real-time TTS for: {text}")
        audio_generated = False
        start_time = time.time()

        try:
            tts_stream_start = time.time()
            async for audio_chunk in synthesize_streaming_sarvam_tts(text):
                if stop_event.is_set():
                    logger.info(f"[Streaming-TTS-{self.call_sid}]: Playback interrupted by stop event")
                    break

                logger.info(f"[Streaming-TTS-{self.call_sid}]: Received audio chunk of size {len(audio_chunk)}")
                if not audio_chunk:
                    continue

                audio_generated = True

                for i in range(0, len(audio_chunk), AUDIO_CHUNK_SIZE):
                    if stop_event.is_set():
                        logger.info(f"[Streaming-TTS-{self.call_sid}]: Playback interrupted during chunk")
                        break

                    chunk = audio_chunk[i: i + AUDIO_CHUNK_SIZE]
                    try:
                        await self.send(text_data=json.dumps({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": base64.b64encode(chunk).decode()},
                        }))
                    except Exception as e:
                        logger.error(f"[Streaming-TTS-{self.call_sid}]: Error sending chunk: {e}")
                        return

                    await asyncio.sleep(0.01)

            logger.info(f"[Timing-TTS-{self.call_sid}] Streaming TTS completed in {time.time() - tts_stream_start:.2f}s")

        except Exception as e:
            logger.error(f"[Streaming-TTS-{self.call_sid}]: Error during streaming: {e}")

        if not audio_generated:
            logger.warning(f"[Streaming-TTS-{self.call_sid}]: No audio generated, using fallback")
            try:
                fallback_start = time.time()
                audio_chunk = await synthesize_mulaw_sarvam_tts(text)
                logger.info(f"[Timing-TTS-{self.call_sid}] Fallback TTS took {time.time() - fallback_start:.2f}s")

                if audio_chunk:
                    for i in range(0, len(audio_chunk), AUDIO_CHUNK_SIZE):
                        if stop_event.is_set():
                            logger.info(f"[Fallback-TTS-{self.call_sid}]: Playback interrupted")
                            break
                        chunk = audio_chunk[i: i + AUDIO_CHUNK_SIZE]
                        try:
                            await self.send(text_data=json.dumps({
                                "event": "media",
                                "streamSid": self.stream_sid,
                                "media": {"payload": base64.b64encode(chunk).decode()},
                            }))
                        except Exception as e:
                            logger.error(f"[Fallback-TTS-{self.call_sid}]: Error sending fallback chunk: {e}")
                            return
                        await asyncio.sleep(0.01)
                    audio_generated = True
            except Exception as fallback_e:
                logger.error(f"[Fallback-TTS-{self.call_sid}]: Fallback TTS failed: {fallback_e}")

        if not stop_event.is_set() and audio_generated:
            await self.send(text_data=json.dumps({
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": f"{mark_name}_tts_complete" if mark_name else "streaming_tts_complete"},
            }))
            logger.info(f"[Streaming-TTS-{self.call_sid}]: Streaming complete, sent mark")

        logger.info(f"[Timing-TTS-{self.call_sid}] Total TTS handling time: {time.time() - start_time:.2f}s")


    async def handle_streaming_ai_response(self, user_text, context):
        """Handle AI response with streaming agent and streaming TTS."""
        logger.info(f"[Streaming-AI-{self.call_sid}]: Processing streaming AI response for: {user_text}")
        total_start = time.time()

        try:
            agent = LangChainAIAgent()
            response_chunks = []
            sentence_buffer = ""

            ai_start = time.time()
            async for chunk in agent.process_query_streaming(user_text, self.call_sid, context):
                response_chunks.append(chunk)
                sentence_buffer += chunk

                if sentence_buffer.strip().endswith(('.', '!', '?')) or len(sentence_buffer.strip()) > 50:
                    if sentence_buffer.strip():
                        logger.info(f"[Streaming-AI-{self.call_sid}]: Sending chunk to TTS: {sentence_buffer.strip()}")

                        tts_wait_start = time.time()
                        if self.response_tts_task and not self.response_tts_task.done():
                            await self.wait_for_tts_finish(self.response_tts_task)
                        logger.info(f"[Timing-AI-{self.call_sid}] Waited {time.time() - tts_wait_start:.2f}s for previous TTS")

                        self.response_tts_stop_event.clear()
                        tts_stream_start = time.time()
                        self.response_tts_task = asyncio.create_task(
                            self.stream_real_time_tts_to_client(
                                sentence_buffer.strip(),
                                self.response_tts_stop_event,
                                mark_name="streaming_ai_chunk"
                            )
                        )
                        sentence_buffer = ""
                        await asyncio.sleep(0.1)

            logger.info(f"[Timing-AI-{self.call_sid}] AI streaming response received in {time.time() - ai_start:.2f}s")

            # Handle any remaining buffer
            if sentence_buffer.strip():
                logger.info(f"[Streaming-AI-{self.call_sid}]: Sending final chunk to TTS: {sentence_buffer.strip()}")
                if self.response_tts_task and not self.response_tts_task.done():
                    await self.wait_for_tts_finish(self.response_tts_task)

                self.response_tts_stop_event.clear()
                self.response_tts_task = asyncio.create_task(
                    self.stream_real_time_tts_to_client(
                        sentence_buffer.strip(),
                        self.response_tts_stop_event,
                        mark_name="streaming_ai_final"
                    )
                )

            final_response = "".join(response_chunks)

            context_store_start = time.time()
            store_context(self.call_sid, user_text, final_response)
            logger.info(f"[Timing-AI-{self.call_sid}] Stored context in {time.time() - context_store_start:.2f}s")

            logger.info(f"[Timing-AI-{self.call_sid}] Total streaming AI response time: {time.time() - total_start:.2f}s")
            return final_response

        except Exception as e:
            logger.error(f"[Streaming-AI-{self.call_sid}]: Error in streaming AI response: {e}")
            fallback_ai_start = time.time()
            ai_response = await get_ai_response(user_text, context, self.call_sid)
            logger.info(f"[Timing-AI-{self.call_sid}] Fallback AI response took {time.time() - fallback_ai_start:.2f}s")

            store_context(self.call_sid, user_text, ai_response)
            return ai_response


    async def _cleanup_tasks(self):
        """Helper to cancel and await background tasks."""
        logger.info(f"🧹 [Cleanup-{self.call_sid}] Cancelling background tasks...")
        tasks_to_cancel = [self.silence_task, self.engagement_task, self.response_tts_task, self.engagement_tts_task]
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info(f"🧹 [Cleanup-{self.call_sid}] All tasks cancelled.")
