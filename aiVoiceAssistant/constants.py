INITIAL_GREETING_TEXT = "Hi there, this is Manisha from Omen Tech Innovations! Kya aap kisi business ko chalate hain ya kisi aise business ke saath kaam karte hain jo customer calls handle karta hai?"

AI_SYSTEM_PROMPT = """You are Manisha, a friendly and conversational voice agent from Aman Tech Innovations.
... (same as Flask) ...
"""

ENGAGEMENT_WORDS = {...}  # Copy the dict

ENGAGEMENT_RESPONSES = {...}

# Audio settings
AUDIO_CHUNK_DIR = "audio_chunks"
RESPONSE_AUDIO_CHUNK_DIR = "response_audio_chunks"
AUDIO_CACHE_DIR = "static/audio_cache"

AUDIO_CHUNK_SIZE = 160
AUDIO_SAMPLE_RATE = 8000
SILENCE_MAX_DURATION = 0.8
AUDIO_BUFFER_SILENCE = 1.5
MIN_AUDIO_BYTES = 6000
VAD_MIN_SILENCE_FRAMES = 5
VAD_FRAME_SIZE = 320

AUDIO_SILENCE_THRESHOLDS = {
    "MAX_AMPLITUDE": 10000,
    "MIN_RMS_DBFS": -30.0,
}

ENGAGEMENT_TRIGGER_SECONDS = 3.0
ENGAGEMENT_BACKCHANNEL_REPEAT_DELAY = 3.0
DISENGAGEMENT_TRIGGER_SECONDS = 10.0
DISENGAGEMENT_BACKCHANNEL_REPEAT_DELAY = 5.0
