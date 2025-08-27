import asyncio
import os
import re
import time
import logging
from typing import Optional, Set
import uuid
from groq import Groq
import whisper
from openai import AsyncOpenAI, OpenAI

from aiVoiceAssistant.constants import ENGAGEMENT_WORDS
from aiVoiceAssistant.settings import GROQ_API_KEY, GROQ_CHAT_TEMPERATURE, GROQ_STT_MODEL, OPENAI_API_KEY, OPENAI_MODEL, WHISPER_STT_OFFLINE_MODEL
from core.utils.langchain_agent import LangChainAIAgent

logger = logging.getLogger(__name__)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)
openai_model = OPENAI_MODEL
groq_client = Groq(api_key=GROQ_API_KEY)


# Load once and reuse (recommended for performance)
whisper_model = whisper.load_model(WHISPER_STT_OFFLINE_MODEL)  # Options: tiny, base, small, medium, large

langchain_agent = LangChainAIAgent()

async def get_ai_response(user_input, context=None, call_sid="default"):
    try:
        logger.info(f"User input: {user_input}")

        # Convert the Redis context format to the expected format
        redis_context = context if context else []

        # Use the langchain agent to process the query with Redis context
        response = await langchain_agent.process_query(user_input, call_sid, redis_context)

        return response.strip() if response else ""
    except Exception as e:
        logger.error(f"LangChain agent error: {e}", exc_info=True)
        return "Sorry, something went wrong."


def transcribe_audio_whisper_groq(filepath, lang="hi"):
    try:
        start_time = time.time()
        logger.info(f"Transcribing {filepath} with groq in {lang} language...")

        with open(filepath, "rb") as audio_file:
            transcription = groq_client.audio.translations.create(
                file=(filepath, audio_file.read()),
                model=GROQ_STT_MODEL,
                prompt=(
                    "Transcribe audio from users who often speak in a mix of Hindi/indian-languages and English. "
                    "Preserve meaning, skip background noise, ignore filler sounds like 'umm', 'aaa', 'okay' etc, and if there is no valid response then do not return any random text instead of that return an empty response like ''"
                ),
                temperature=GROQ_CHAT_TEMPERATURE,
            )
        endtime = time.time()

        logger.info(f"Groq Whisper v3-turbo response time: {endtime - start_time:.2f}s")
        logger.info(f"Groq Whisper v3-turbo response: {getattr(transcription, 'text', None)}")

        return transcription.text.strip() if hasattr(transcription, "text") else ""
    except Exception as e:
        logger.error(f"Groq Whisper v3-turbo error: {e}", exc_info=True)
        raise e


async def is_user_engagement(
    user_text: str,
    call_sid: str,
    lang: str = "hi",
) -> bool:
    """
    Determines whether user_text is just engagement.

    Returns:
        True if user input is considered an engagement/backchannel,
        False if it's a true interruption.
    """

    # Lowercase
    text = user_text.lower().strip()

    # Remove punctuation and special characters (except letters, digits, whitespace)
    text = re.sub(r"[^\w\s]", "", text)

    # Collapse multiple spaces to a single space
    normalized = re.sub(r"\s+", " ", text)

    if not normalized:
        return True

    if normalized in ENGAGEMENT_WORDS:
        return True
    # return False

    try:
        return await langchain_agent.classify_user_input_type(user_text, call_sid)
    except Exception as e:
        logger.error(f"[BargeIn-{call_sid}]: LLM classification error: {e}", exc_info=True)
        return False

