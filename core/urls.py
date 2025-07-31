from django.urls import path
from .views import health_check, voice, make_call

urlpatterns = [
    path('', health_check),
    path('voice/', voice),
    path('make-call/', make_call),
]
