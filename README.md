# AI Voice Assistant

Real-time conversational AI for phone calls, built with Django, Django Channels, and Twilio Media Streams. The app accepts live call audio over WebSockets, detects pauses and interruptions, transcribes speech with Groq, generates replies with an OpenAI-powered LangChain agent, synthesizes speech with Sarvam AI, and streams audio back to the caller.

## Core stack

| Category | Technology / Service | Purpose |
| --- | --- | --- |
| Backend | Django, Django REST Framework, Django Channels, Daphne | HTTP APIs, ASGI app, WebSocket handling |
| Telephony | Twilio Voice API, Twilio Media Streams | Outbound calls and live PSTN audio streaming |
| LLM / Agent | OpenAI GPT-4o, LangChain, Tavily | Response generation, tools, short-term memory |
| Speech | Groq Whisper, Sarvam AI, FFmpeg | Speech-to-text, text-to-speech, audio transcoding |
| Realtime audio | WebRTC VAD, NumPy | Voice activity detection, silence detection, barge-in handling |
| State | Redis | Per-call conversation context and Channels layer |

## How it works

1. `POST /make-call/` creates an outbound call through Twilio.
2. Twilio requests `POST /voice/`.
3. The server returns TwiML that opens a media stream to `wss://<host>/ws`.
4. `core.consumers.VoiceStreamConsumer` receives mu-law audio frames in real time.
5. The consumer buffers speech, detects silence with WebRTC VAD, and watches for barge-in while TTS is playing.
6. Completed utterances are converted to WAV with FFmpeg and transcribed with Groq Whisper.
7. The transcribed text is passed to a LangChain agent backed by OpenAI, with conversation context loaded from Redis.
8. The AI reply is synthesized with Sarvam AI and streamed back to Twilio as 8 kHz mu-law audio.

## Features

- Real-time Twilio Media Streams integration over WebSockets
- Silence detection and utterance segmentation with WebRTC VAD
- Barge-in detection to stop TTS when the caller interrupts
- Cached greeting and engagement/disengagement prompts
- Redis-backed per-call conversation context
- Streaming AI response flow with streaming TTS playback
- Hinglish-oriented voice agent prompt tuned for outbound business qualification calls

## Repository layout

| Path | Description |
| --- | --- |
| `aiVoiceAssistant/asgi.py` | ASGI entry point for HTTP and WebSocket traffic |
| `aiVoiceAssistant/settings.py` | Environment-driven configuration for Twilio, OpenAI, Groq, Sarvam, and Redis |
| `aiVoiceAssistant/constants.py` | Prompt text, audio thresholds, and cache/input/output directory settings |
| `core/views.py` | Health check, Twilio voice webhook, and outbound call initiation endpoint |
| `core/consumers.py` | Main real-time call orchestration logic |
| `core/routing.py` | WebSocket route definition for `/ws` |
| `core/utils/open_ai_utils.py` | AI response orchestration and Groq transcription helpers |
| `core/utils/langchain_agent.py` | LangChain agent, memory management, and interruption classification |
| `core/utils/sarvam_utils.py` | Sarvam TTS/STT helpers and streaming TTS support |
| `core/utils/audio_utils.py` | FFmpeg transcoding and VAD/audio utility functions |
| `core/utils/redis_utils.py` | Redis conversation context storage |

## Prerequisites

- Python 3.10+ recommended
- Redis running locally or reachable via `REDIS_URL`
- FFmpeg installed and available on `PATH`
- Twilio account with a voice-enabled phone number
- Public HTTPS/WSS endpoint for Twilio webhooks and media streams, typically via ngrok or a deployed host

### 2\. Installation & Configuration

```bash
# Clone the repository
git clone https://github.com/Ajay-Maury/ai-call-assistant.git
cd ai-call-assistant

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create your environment file from the example
# (This step is best practice to avoid missing variables)
# On Linux/macOS:
cp .env.example .env
# On Windows:
# copy .env.example .env

# Now, edit the .env file with your API keys and credentials
```

The code currently reads these variables from `aiVoiceAssistant/settings.py`:

```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
VERIFIED_TEST_NUMBER=+1xxxxxxxxxx

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

GROQ_API_KEY=gsk_...
GROQ_STT_MODEL=whisper-large-v3
GROQ_CHAT_TEMPERATURE=0.4

SARVAM_API_KEY=your_sarvam_subscription_key
SARVAM_SPEAKER=anushka
SARVAM_LANGUAGE=hi-IN
SARVAM_TTS_MODEL=bulbul:v2
SARVAM_STT_MODEL=saarika:v2.5
SARVAM_PACE=1.0
SARVAM_PITCH=0.2

REDIS_URL=redis://localhost:6379/0
WEBSOCKET_URL=wss://<public-host>/ws
VOICE_ROUTE_URL=https://<public-host>/voice/

TAVILY_API_KEY=tvly-...
WHISPER_STT_OFFLINE_MODEL=medium
```

Notes:

- `WEBSOCKET_URL` and `VOICE_ROUTE_URL` are supported in settings, but `core/views.py` currently defines hardcoded module-level defaults. Keep those values aligned with your public host or move the view to read the settings variables directly.
- Twilio requires publicly reachable `https://` and `wss://` endpoints.
- Redis is used both for the Channels layer and for storing call context.

## Local setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Start Redis.
4. Expose the app publicly.
5. Run the Django ASGI server.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

If you use ngrok:

```bash
ngrok http 8000
```

Then set:

- `WEBSOCKET_URL=wss://<your-ngrok-domain>/ws`
- `VOICE_ROUTE_URL=https://<your-ngrok-domain>/voice/`

If you do not update `core/views.py`, also replace the hardcoded `WEB_SOCKET_URL` and `VOICE_ROUTE_URL` constants there.

## API surface

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Health check endpoint |
| `POST` | `/voice/` | Twilio webhook that returns TwiML and starts media streaming |
| `POST` | `/make-call/` | Starts an outbound call. Payload: `{"to": "+1..."}` |
| `WS` | `/ws` or `/ws/` | WebSocket endpoint for Twilio media streaming |

Example outbound call request:

```bash
curl -X POST http://127.0.0.1:8000/make-call/ \
  -H "Content-Type: application/json" \
  -d '{"to":"+15551234567"}'
```

## Runtime behavior and caveats

- The app is optimized for Hindi/Hinglish call flows and the default agent persona is configured for outbound lead qualification.
- Audio is transcoded through FFmpeg, so missing FFmpeg will break both STT preprocessing and TTS playback.

