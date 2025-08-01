# AI Call Assistant

A real-time, low-latency conversational AI for telephony, built on Django and Twilio. This project demonstrates a sophisticated voice bot capable of handling natural conversation with features like **barge-in detection**, **voice activity detection (VAD)**, and **persistent context**.

## 🛠️ Core Technologies

| Category                 | Technology / Service                                       | Purpose                                            |
| ------------------------ | ---------------------------------------------------------- | -------------------------------------------------- |
| **Core Framework** | Django, Django Channels                                    | Backend Server & WebSocket Handling                |
| **AI Services** | OpenAI (GPT-4o), LangChain, Groq, Sarvam AI                | LLM Logic, STT, and TTS                            |
| **Telephony & Streaming**| Twilio Voice API                                           | PSTN Call Gateway & Real-time Audio Stream         |
| **Infrastructure** | Redis, FFmpeg, WebRTC VAD                                  | State/Cache, Audio Processing, Voice Detection     |

## 🌊 Architectural Flow

Here's how a call is processed in real-time:

1.  **📞 Initiate Call**: A `POST` to `/make-call` tells Twilio to start a call.
2.  **🔗 Connect**: Twilio hits `/voice`, which responds with TwiML to open a WebSocket to `/ws`.
3.  **🎤 User Speaks**: Audio is streamed from the user to the server via the WebSocket.
4.  **🤫 VAD & Barge-In**: The server buffers audio, detects when the user stops talking, and listens for interruptions.
5.  **✍️ Transcribe**: The audio segment is sent to **Groq** for ultra-fast Speech-to-Text.
6.  **🧠 Think**: The transcribed text is sent to a **LangChain agent** using **GPT-4o** to generate a response, using context from **Redis**.
7.  **🗣️ Synthesize**: The agent's response is sent to **Sarvam AI** for Text-to-Speech conversion.
8.  **🔊 Respond**: The synthesized audio is streamed back to the user through the WebSocket.

-----

## 🚀 Quick Start (Setup & Run)

### 1\. Prerequisites

  * Python 3.8+
  * [FFmpeg](https://ffmpeg.org/download.html) (available in your system's PATH)
  * A running [Redis](https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/) instance
  * A public URL from a tunneling service like [ngrok](https://ngrok.com/download)

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

**`.env` file contents:**

```env
# Get these keys from their respective service dashboards
OPENAI_API_KEY="sk-..."
GROQ_API_KEY="gsk_..."
SARVAM_SUBSCRIPTION_KEY="..."
TAVILY_API_KEY="tvly-..."
TWILIO_SID="AC..."
TWILIO_TOKEN="..."
TWILIO_NUMBER="+1..."
REDIS_URL="redis://localhost:6379/0"
```

### 3\. Expose Your Local Server

For Twilio to send webhooks to your local machine, you need a public URL.

```bash
# Start ngrok on port 8000
ngrok http 8000
```

Copy the `https` forwarding URL provided by ngrok.

### 4\. Update Code with Your Public URL

Open `core/views.py` and replace the placeholder URLs with your public ngrok URL.

```python
# core/views.py
WEB_SOCKET_URL = "wss://<your-ngrok-url>.ngrok-free.app/ws"
VOICE_ROUTE_URL = "https://<your-ngrok-url>.ngrok-free.app/voice"
```

### 5\. Run the Application

```bash
# Apply migrations (if any) and run the server
python manage.py migrate
python manage.py runserver
```

Your AI Voice Assistant is now live and ready to take calls\!

-----

## 🔌 API Endpoints

| Method | Path         | Description                                        |
| :----- | :----------- | :------------------------------------------------- |
| `POST` | `/make-call/`| Initiates an outbound call. Payload: `{"to": "+1..."}` |
| `POST` | `/voice/`    | Twilio webhook to handle incoming call connections. |
| `GET`  | `/`          | A simple health check endpoint.                    |
| `WS`   | `/ws/`       | WebSocket endpoint for real-time audio streaming.  |

## 📂 Code Structure

| File / Directory       | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| `core/consumer.py`     | **The heart of the app.** Handles WebSocket logic, VAD, and AI orchestration. |
| `core/langchain_agent.py`| Defines the AI agent's personality, tools, and conversation flow.          |
| `core/views.py`        | Handles HTTP requests (`/make-call`, `/voice`).                          |
| `core/utils/`          | Utility modules for audio, AI services, and Redis.                       |
| `requirements.txt`     | All Python project dependencies.                                         |
| `.env.example`         | Template for required environment variables.                             |
