import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar
from pydantic import BaseModel
from groq import AsyncGroq
from backend.app.config import settings

logger = logging.getLogger("advisor.llm")

T = TypeVar("T", bound=BaseModel)

class BaseLLMProvider(ABC):
    """Abstract base class to facilitate swapping LLM providers (e.g. Groq, OpenAI, Anthropic)."""

    @abstractmethod
    async def chat_completion(
        self, 
        prompt: str, 
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 0.2
    ) -> str:
        """Run standard text completion asynchronously."""
        pass

    @abstractmethod
    async def chat_completion_json(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 0.1
    ) -> T:
        """Run structured JSON completions mapped directly to a Pydantic model."""
        pass


class GroqProvider(BaseLLMProvider):
    """Groq Provider implementation utilizing Llama-3.3-70b-versatile for high performance."""

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured in the environment variables.")
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    async def chat_completion(
        self, 
        prompt: str, 
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 0.2
    ) -> str:
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=temperature
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq API text completion error: {e}")
            raise e

    async def chat_completion_json(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: str = "You are a helpful assistant.",
        temperature: float = 0.1
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
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content or "{}"
            parsed_data = json.loads(raw_content)
            return response_model.model_validate(parsed_data)
        except Exception as e:
            logger.error(f"Groq API structured JSON error: {e}")
            # Log the raw response if available
            try:
                logger.debug(f"Failed raw payload: {response.choices[0].message.content}")
            except:
                pass
            raise e

# Factory function to obtain the configured provider
def get_llm_provider() -> BaseLLMProvider:
    """Return configured provider based on settings. Highly extensible."""
    # Currently Groq is default, can easily map keys to OpenAIProvider, AnthropicProvider
    return GroqProvider()

# Global instantiator
llm_service = get_llm_provider()
