import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar
from pydantic import BaseModel
from groq import AsyncGroq
from backend.tools.dataset_advisor.config import settings

logger = logging.getLogger("advisor.llm")

T = TypeVar("T", bound=BaseModel)

class BaseLLMProvider(ABC):
    """Abstract base class to facilitate swapping LLM providers (e.g. Groq, OpenAI, Anthropic)."""

    @abstractmethod
    async def chat_completion(
        self, 
        prompt: str, 
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        api_key: Optional[str] = None
    ) -> str:
        """Run standard text completion asynchronously."""
        pass

    @abstractmethod
    async def chat_completion_json(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 0.1,
        api_key: Optional[str] = None
    ) -> T:
        """Run structured JSON completions mapped directly to a Pydantic model."""
        pass


class GroqProvider(BaseLLMProvider):
    """Groq Provider implementation utilizing Llama-3.3-70b-versatile for high performance."""

    def __init__(self):
        self.model = settings.GROQ_MODEL

    def _get_client(self, api_key: Optional[str] = None) -> AsyncGroq:
        key = api_key or settings.GROQ_API_KEY
        if not key or key.strip() == "":
            raise ValueError("GROQ_API_KEY is not configured.")
        return AsyncGroq(api_key=key.strip())

    async def chat_completion(
        self, 
        prompt: str, 
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        api_key: Optional[str] = None
    ) -> str:
        try:
            client = self._get_client(api_key)
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=temperature
            )
            try:
                from backend.utils.llm_logger import log_groq_response
                log_groq_response(response, module_name="dataset_advisor")
            except Exception as e_log:
                logger.warning(f"Failed to log token usage: {e_log}")
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq API text completion error: {e}")
            raise e

    async def chat_completion_json(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 0.1,
        api_key: Optional[str] = None
    ) -> T:
        # Enhance instructions to enforce JSON output matching the target schema
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        full_system = (
            f"{system_instruction}\n"
            f"You MUST return a JSON object that strictly complies with the following JSON schema:\n"
            f"{schema_json}\n"
            f"Do NOT wrap the output in markdown codeblocks (e.g. ```json). Output raw valid JSON only."
        )
        
        try:
            client = self._get_client(api_key)
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            try:
                from backend.utils.llm_logger import log_groq_response
                log_groq_response(response, module_name="dataset_advisor")
            except Exception as e_log:
                logger.warning(f"Failed to log token usage: {e_log}")
            raw_content = response.choices[0].message.content or "{}"
            parsed_data = json.loads(raw_content)
            return response_model.model_validate(parsed_data)
        except Exception as e:
            logger.error(f"Groq API structured JSON error: {e}")
            try:
                logger.debug(f"Failed raw payload: {response.choices[0].message.content}")
            except:
                pass
            raise e


def get_llm_provider() -> BaseLLMProvider:
    """Return configured provider based on settings."""
    return GroqProvider()

# Global instantiator
llm_service = get_llm_provider()
