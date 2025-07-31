import random

from aiVoiceAssistant.constants import ENGAGEMENT_RESPONSES

def get_engagement_response(status="DISENGAGED", lang_code="en"):
    lang = lang_code.split("-")[0]  # normalize like 'en-US' → 'en'
    responses_by_lang = ENGAGEMENT_RESPONSES.get(status, {})
    return random.choice(responses_by_lang.get(lang, responses_by_lang.get("en", [])))
