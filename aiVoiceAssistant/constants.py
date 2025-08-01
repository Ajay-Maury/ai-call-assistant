import os


AI_SYSTEM_PROMPT = """You are Manisha, a friendly and conversational voice agent from Aman Tech Innovations.

Your job is to engage the user in a natural, voice-friendly way and introduce our Voice AI bot solution that helps businesses automate outbound calls and customer engagement.

Follow these general instructions for the conversation:

1. **Greet and Introduce Yourself**
   - Always start warmly by greeting the user by name (e.g., "Hi Aman") and saying you're calling from Aman Tech Innovations.
   
2. **State the Purpose Clearly**
   - Briefly explain that you're calling to introduce a Voice AI bot that can help automate customer conversations.

3. **Ask for Permission**
   - Politely ask if it's a good time to share more about how it could help their business.

4. **Qualify the User**
   - Ask if they run a business or work with businesses.
   - If yes, ask if their business involves outbound calls (e.g., sales, support, reminders).
   - Then ask for an idea of the outbound call volume — daily, monthly, or yearly.

5. **Style & Language**
   - Keep responses short, friendly, and suitable for voice (1-2 sentences max).
   - Use a cheerful and conversational tone.
   - If the user speaks Hindi, respond in informal Hinglish unless they ask otherwise.
   - Match the language and tone of the user to maintain a natural flow.

---

**Sample Conversation Opening**:

"Hi Aman, this is Manisha calling from Aman Tech Innovations!  
I'm just calling to quickly introduce a Voice AI bot we've built — it helps automate outbound calls and customer engagement.  
Do you have a quick minute for me to share how it might help your business?"

"""


INITIAL_GREETING_TEXT = "Hi there, this is Manisha from Omen Tech Innovations! Kya aap kisi business ko chalate hain ya kisi aise business ke saath kaam karte hain jo customer calls handle karta hai?"

# Predefined user engagement expressions
ENGAGEMENT_WORDS = {
    # Common Hinglish acknowledgements
    "haan", "ha", "hanji", "haanji", "hmm", "hmm hmm", "hmmhmm", "huh", "huh huh", "uh huh",
    "hum", "ji", "hmm ji", "hmm haan", "hmm haanji", "hmm hanji",
    
    # Variants of "ok", "fine", "alright"
    "ok", "okay", "ok ok", "okay okay", "okey", "alright", "right", "fine", "cool", "done", "perfect", "great", "nice",
    "hmm okay", "hmm ok", "hmm fine", "hmm great", "hmm nice", "hmm perfect", "hmm done", "sounds good", "makes sense",
    
    # Hindi acknowledgements
    "achha", "accha", "acha", "thik hai", "theek hai", "theek", "samjha", "samajh gaya", "hmm samajh gaya", "hmm samjha",
    "sahi", "hmm sahi", "hmm theek", "hmm thik", "hmm achha", "hmm accha", "hmm acha",
    
    # Basic affirmatives
    "yes", "yeah", "yup", "i see", "got it", "i got it", "understood", "noted", "sure", "okay sure", "yes yes",
    
    # Expressions often used in non-intent fillers
    "oh", "ohh", "oh okay", "ohh okay", "ohk", "hmm oh", "hmm ohk", "hmm hmm hmm", "hmm noted", "hmm sure",

    # Polite closures or filler
    "thank you", "thanks", "thanks a lot", "thankyou", "thank u", "thankyou so much", "thanks so much",

    # Connectors (used as filler)
    "and", "or", "yeah yeah", "ya", "yaar", "bas", "thik", "hmm thik hai", "hmm theek hai", "thik thak",

    # Friendly passive tone
    "hmm hmm hmm", "mmm", "mmm hmm", "nice nice", "great great",

    # Random casual fillers
    "Thank you for watching!"
}

ENGAGEMENT_RESPONSES = {
    "ENGAGED": {
        "en": ["hmm", "okay", "got it", "yeah"],
        "hi": ["हां", "अच्छा", "ओके", "ठीक है"]
        # Add more languages here
    },
    "DISENGAGED": {
        "en": ["Are you there?", "Still with me?", "Can you hear me?"],
        "hi": ["क्या आप अभी भी लाइन पर हैं?", "क्या मेरी बात सुन पा रहे हैं?", "आपकी आवाज़ नहीं आ रही है!",],
        # Add more languages here
    }
}
# Audio settings
AUDIO_CHUNK_DIR = "audio_chunks_input"
RESPONSE_AUDIO_CHUNK_DIR = "audio_chunks_response"
AUDIO_CACHE_DIR = "static/audio_cache"

# Ensure audio directories exist
os.makedirs(AUDIO_CHUNK_DIR, exist_ok=True)
os.makedirs(RESPONSE_AUDIO_CHUNK_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Audio processing constants
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
