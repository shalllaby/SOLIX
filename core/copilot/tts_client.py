"""
core/copilot/tts_client.py
==========================
ElevenLabs TTS client implementation with robust fallbacks for SOL.
Uses ElevenLabs (Multilingual v2, Egyptian voice) as primary high-fidelity engine.
Falls back to Edge TTS (ar-EG-SalmaNeural / ar-EG-ShakirNeural) and Google TTS (gTTS) to guarantee audio playback.
"""

import os
import asyncio
import io
from dotenv import load_dotenv

# Load env variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path=env_path, override=True)

try:
    from elevenlabs.client import ElevenLabs
    HAS_ELEVENLABS = True
except ImportError:
    HAS_ELEVENLABS = False

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
MODEL_ID = "eleven_multilingual_v2"
MAX_TEXT_LENGTH = 1000


def _truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text at word boundary to stay within TTS limits."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + " ..."


async def _run_edge_tts(text: str) -> bytes | None:
    """Run edge-tts asynchronously to get audio bytes."""
    try:
        # ar-EG-SalmaNeural is a highly natural Egyptian female/neutral voice. 
        # ar-EG-ShakirNeural is another option. Let's use SalmaNeural or ShakirNeural.
        communicate = edge_tts.Communicate(text, "ar-EG-SalmaNeural")
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        if audio_data:
            return audio_data
    except Exception as e:
        print(f"[TTS Fallback] Edge-TTS error: {e}")
    return None


def text_to_audio(text: str) -> bytes | None:
    """
    Convert a text string to MP3 audio bytes using ElevenLabs, 
    with automatic fallbacks to edge-tts and gTTS.
    """
    if not text or not text.strip():
        return None

    # Load fresh env variables
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
    load_dotenv(dotenv_path=env_path, override=True)
    
    current_api_key = os.environ.get("ELEVENLABS_API_KEY", API_KEY)
    current_voice_id = os.environ.get("ELEVENLABS_VOICE_ID", VOICE_ID)

    text = _truncate(text)

    # 1. Primary Attempt — ElevenLabs
    if HAS_ELEVENLABS and current_api_key and current_api_key != "your_elevenlabs_api_key_here":
        try:
            dyn_client = ElevenLabs(api_key=current_api_key)
            audio_iterator = dyn_client.text_to_speech.convert(
                text=text,
                voice_id=current_voice_id,
                model_id=MODEL_ID,
                output_format="mp3_44100_128"
            )
            if isinstance(audio_iterator, (bytes, bytearray)):
                audio_bytes = audio_iterator
            elif hasattr(audio_iterator, "__iter__") or hasattr(audio_iterator, "__next__"):
                audio_bytes = b"".join(audio_iterator)
            else:
                audio_bytes = audio_iterator
            
            if audio_bytes and len(audio_bytes) > 0:
                print("[TTS] ElevenLabs audio generated successfully.")
                return audio_bytes
        except Exception as e:
            print(f"[TTS] ElevenLabs failed: {e}. Trying legacy/fallbacks...")

    # 2. Secondary Attempt — Edge TTS
    if HAS_EDGE_TTS:
        try:
            print("[TTS] Trying Edge-TTS fallback...")
            # Run the async communicating stream inside a new event loop or using current loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # If loop is already running (e.g. inside FastAPI context), use run_coroutine_threadsafe
                import sys
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(_run_edge_tts(text)))
                    audio_bytes = future.result()
            else:
                audio_bytes = loop.run_until_complete(_run_edge_tts(text))

            if audio_bytes:
                print("[TTS] Edge-TTS audio generated successfully.")
                return audio_bytes
        except Exception as e_edge:
            print(f"[TTS] Edge-TTS failed: {e_edge}")

    # 3. Tertiary Attempt — Google TTS (gTTS)
    if HAS_GTTS:
        try:
            print("[TTS] Trying gTTS fallback...")
            tts = gTTS(text=text, lang='ar', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.read()
            if audio_bytes:
                print("[TTS] gTTS audio generated successfully.")
                return audio_bytes
        except Exception as e_gtts:
            print(f"[TTS] gTTS failed: {e_gtts}")

    print("[TTS] All text-to-speech options failed.")
    return None
