# core/copilot/__init__.py
# Module initialization for the voice-enabled data copilot (SOL).

from core.copilot.sandbox import execute_pandas_code
from core.copilot.data_handler import extract_schema, schema_to_prompt_text
from core.copilot.tts_client import text_to_audio
from core.copilot.audio_handler import transcribe_audio
from core.copilot.llm_client import get_chat_response
