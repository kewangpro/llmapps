"""
LLM Provider abstraction for different AI services
"""

import os
import httpx
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("LLMProviders")


class BaseLLM(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def call(self, prompt: str) -> str:
        """Call the LLM API and return response"""
        pass

    def call_with_image(self, prompt: str, image_data: str) -> str:
        """Call the LLM API with image data and return response"""
        # Default implementation returns error - providers should override if they support images
        # Using _ prefix to indicate intentionally unused parameters
        return f"Error: {self.__class__.__name__} does not support image analysis. Please use a vision-capable model."

    def supports_vision(self) -> bool:
        """Check if this LLM provider supports vision/image analysis"""
        return hasattr(self, '_supports_vision') and self._supports_vision


class OllamaLLM(BaseLLM):
    """Ollama LLM client"""

    def __init__(self, model: str = "gemma3:latest", base_url: str = "http://localhost:11434"):
        super().__init__(model)
        self.base_url = base_url
        # Most modern Ollama models support vision
        self._supports_vision = True

    def call(self, prompt: str) -> str:
        """Call Ollama API and return response"""
        import time
        start_time = time.time()
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"🤖 Calling {self.model}: {prompt_preview}")

        try:
            with httpx.Client(base_url=self.base_url, timeout=300.0) as client:
                response = client.post(
                    '/api/chat',
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False
                    }
                )
                result = response.json()
                content = result.get('message', {}).get('content', '')

                duration = time.time() - start_time
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"✅ {self.model} responded in {duration:.2f}s: {content_preview}")

                return content
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {self.model} failed after {duration:.2f}s: {str(e)}")
            return f"Error calling Ollama: {str(e)}"

    def call_with_image(self, prompt: str, image_data: str) -> str:
        """Call Ollama API with image data and return response"""
        import time
        start_time = time.time()
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"🤖🖼️ Calling {self.model} with image: {prompt_preview}")

        try:
            # Extract base64 data from data URL if needed
            if image_data.startswith('data:'):
                base64_data = image_data.split(',')[1]
            else:
                base64_data = image_data

            with httpx.Client(base_url=self.base_url, timeout=300.0) as client:
                response = client.post(
                    '/api/chat',
                    json={
                        "model": self.model,
                        "messages": [{
                            "role": "user",
                            "content": prompt,
                            "images": [base64_data]
                        }],
                        "stream": False
                    }
                )
                result = response.json()
                content = result.get('message', {}).get('content', '')

                duration = time.time() - start_time
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"✅🖼️ {self.model} responded with image in {duration:.2f}s: {content_preview}")

                return content
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌🖼️ {self.model} failed with image after {duration:.2f}s: {str(e)}")
            return f"Error calling Ollama with image: {str(e)}"


class OpenAILLM(BaseLLM):
    """OpenAI GPT client"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        super().__init__(model)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        # GPT-4 and newer models support vision
        self._supports_vision = "gpt-4" in model.lower()

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")

    def call(self, prompt: str) -> str:
        """Call OpenAI API and return response"""
        import time
        start_time = time.time()
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"🤖 Calling {self.model}: {prompt_preview}")

        if not self.api_key:
            logger.error(f"❌ {self.model} failed: API key not configured")
            return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 4000,
                        "temperature": 0.7
                    }
                )

                if response.status_code != 200:
                    duration = time.time() - start_time
                    logger.error(f"❌ {self.model} failed after {duration:.2f}s: HTTP {response.status_code}")
                    return f"Error calling OpenAI: HTTP {response.status_code} - {response.text}"

                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

                duration = time.time() - start_time
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"✅ {self.model} responded in {duration:.2f}s: {content_preview}")

                return content

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {self.model} failed after {duration:.2f}s: {str(e)}")
            return f"Error calling OpenAI: {str(e)}"

    def call_with_image(self, prompt: str, image_data: str) -> str:
        """Call OpenAI API with image data and return response"""
        import time
        start_time = time.time()
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"🤖🖼️ Calling {self.model} with image: {prompt_preview}")

        if not self.api_key:
            logger.error(f"❌🖼️ {self.model} failed: API key not configured")
            return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."

        if not self._supports_vision:
            logger.error(f"❌🖼️ {self.model} failed: No vision support")
            return f"Error: Model {self.model} does not support image analysis. Please use gpt-4o or gpt-4-vision-preview."

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]

            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 4000,
                        "temperature": 0.7
                    }
                )

                if response.status_code != 200:
                    duration = time.time() - start_time
                    logger.error(f"❌🖼️ {self.model} failed after {duration:.2f}s: HTTP {response.status_code}")
                    return f"Error calling OpenAI: HTTP {response.status_code} - {response.text}"

                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

                duration = time.time() - start_time
                content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.info(f"✅🖼️ {self.model} responded with image in {duration:.2f}s: {content_preview}")

                return content

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌🖼️ {self.model} failed with image after {duration:.2f}s: {str(e)}")
            return f"Error calling OpenAI with image: {str(e)}"


class GeminiLLM(BaseLLM):
    """Google Gemini client"""

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(model)
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        # All Gemini 2.5 models support vision
        self._supports_vision = True

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")

    def call(self, prompt: str) -> str:
        """Call Gemini API and return response"""
        import time
        start_time = time.time()
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"🤖 Calling {self.model}: {prompt_preview}")

        if not self.api_key:
            logger.error(f"❌ {self.model} failed: API key not configured")
            return "Error: Gemini API key not configured. Please set GEMINI_API_KEY environment variable."

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }],
                        "generationConfig": {
                            "temperature": 0.7,
                            "topK": 1,
                            "topP": 1,
                            "maxOutputTokens": 4000
                        }
                    }
                )

                if response.status_code != 200:
                    duration = time.time() - start_time
                    logger.error(f"❌ {self.model} failed after {duration:.2f}s: HTTP {response.status_code}")
                    return f"Error calling Gemini: HTTP {response.status_code} - {response.text}"

                result = response.json()
                candidates = result.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        response_text = parts[0].get('text', '')
                        duration = time.time() - start_time
                        content_preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
                        logger.info(f"✅ {self.model} responded in {duration:.2f}s: {content_preview}")
                        return response_text

                duration = time.time() - start_time
                logger.error(f"❌ {self.model} failed after {duration:.2f}s: No response from Gemini API")
                return "No response from Gemini API"

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ {self.model} failed after {duration:.2f}s: {str(e)}")
            return f"Error calling Gemini: {str(e)}"

    def call_with_image(self, prompt: str, image_data: str) -> str:
        """Call Gemini API with image data and return response"""
        import time
        start_time = time.time()
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"🤖🖼️ Calling {self.model} with image: {prompt_preview}")

        if not self.api_key:
            logger.error(f"❌🖼️ {self.model} failed: API key not configured")
            return "Error: Gemini API key not configured. Please set GEMINI_API_KEY environment variable."

        if not self._supports_vision:
            logger.error(f"❌🖼️ {self.model} failed: No vision support")
            return f"Error: Model {self.model} does not support image analysis. Please use gemini-pro-vision."

        try:
            # Extract base64 data from data URL if needed
            if image_data.startswith('data:'):
                mime_type = image_data.split(';')[0].split(':')[1]
                base64_data = image_data.split(',')[1]
            else:
                base64_data = image_data
                mime_type = "image/jpeg"  # Default

            parts = [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": base64_data}}
            ]

            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": parts}],
                        "generationConfig": {
                            "temperature": 0.7,
                            "topK": 1,
                            "topP": 1,
                            "maxOutputTokens": 4000
                        }
                    }
                )

                if response.status_code != 200:
                    duration = time.time() - start_time
                    logger.error(f"❌🖼️ {self.model} failed after {duration:.2f}s: HTTP {response.status_code}")
                    return f"Error calling Gemini: HTTP {response.status_code} - {response.text}"

                result = response.json()
                candidates = result.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        response_text = parts[0].get('text', '')
                        duration = time.time() - start_time
                        content_preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
                        logger.info(f"✅🖼️ {self.model} responded with image in {duration:.2f}s: {content_preview}")
                        return response_text

                duration = time.time() - start_time
                logger.error(f"❌🖼️ {self.model} failed after {duration:.2f}s: No response from Gemini API")
                return "No response from Gemini API"

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌🖼️ {self.model} failed with image after {duration:.2f}s: {str(e)}")
            return f"Error calling Gemini with image: {str(e)}"


def create_llm(provider: str, model: str) -> BaseLLM:
    """Factory function to create LLM instances"""
    if provider.lower() == "openai":
        return OpenAILLM(model)
    elif provider.lower() == "gemini":
        return GeminiLLM(model)
    elif provider.lower() == "ollama":
        return OllamaLLM(model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# All available models
_ALL_MODELS = {
    "openai": {
        "gpt-4o": "gpt-4o",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4-turbo-preview"
    },
    "gemini": {
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite"
    },
    "ollama": {
        "gemma3:latest": "gemma3:latest",
        "llama3.1:latest": "llama3.1:latest",
        "mistral:latest": "mistral:latest",
        "llama3.2-vision:latest": "llama3.2-vision:latest"
    }
}

def get_available_providers():
    """Get available providers based on environment"""
    return _ALL_MODELS

# Dynamic DEFAULT_MODELS based on environment
DEFAULT_MODELS = get_available_providers()