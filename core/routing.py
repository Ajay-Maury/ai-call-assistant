from django.urls import re_path
from . import consumers

# This is for WebSocket connections only
websocket_urlpatterns = [
    re_path(r'^ws/?$', consumers.VoiceStreamConsumer.as_asgi()),
]