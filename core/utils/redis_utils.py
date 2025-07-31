import redis
import json

from aiVoiceAssistant.settings import REDIS_URL

r = redis.from_url(REDIS_URL)

def store_context(call_sid, user_input, ai_reply):
    context = r.get(call_sid)
    if context:
        context = json.loads(context) # type: ignore
    else:
        context = []
    context.append({"role": "user", "content": user_input})
    context.append({"role": "assistant", "content": ai_reply})
    r.set(call_sid, json.dumps(context))

def get_context(call_sid):
    context = r.get(call_sid)
    return json.loads(context) if context else [] # type: ignore
