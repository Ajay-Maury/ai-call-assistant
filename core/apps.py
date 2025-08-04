from django.apps import AppConfig
import threading
import asyncio

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        from core.utils.sarvam_utils import warmup_tts

        def _start_warmup():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(warmup_tts())
                loop.close()
            except Exception as e:
                print(f"[TTS Warmup] Error during warmup: {e}")

        threading.Thread(target=_start_warmup, daemon=True).start()
