from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from atlas.rag.models import RetrievalResult

from .models import GenerationConfig, LLMOutput


class ChatProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> LLMOutput: ...


class StaticChatProvider:
    """Predictable provider for unit tests."""

    def __init__(self, responses: Sequence[str]) -> None:
        if not responses:
            raise ValueError("StaticChatProvider necesita al menos una respuesta.")
        self._responses = list(responses)
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "static"

    @property
    def model_name(self) -> str:
        return "atlas-static-test-provider"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> LLMOutput:
        del system_prompt, user_prompt, config
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return LLMOutput(
            text=self._responses[index],
            provider=self.provider_name,
            model=self.model_name,
        )


_STOPWORDS = {
    "a", "al", "algo", "como", "con", "cual", "cuando", "cuanto", "de",
    "del", "el", "ella", "en", "es", "esta", "este", "hay", "la", "las",
    "lo", "los", "me", "mi", "para", "por", "que", "se", "si", "su",
    "sus", "tengo", "un", "una", "y",
}


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    )


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _fold(text))
        if len(token) > 2 and token not in _STOPWORDS
    }


def _sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if item.strip()
    ]


class ExtractiveEvidenceProvider:
    """Safe local baseline that returns a sentence copied from retrieved evidence.

    This is intentionally not an LLM. It lets the full RAG and validation pipeline be
    tested without credentials or cloud cost. OCI Generative AI is the production path.
    """

    def __init__(self, retrieval: RetrievalResult) -> None:
        self._retrieval = retrieval

    @property
    def provider_name(self) -> str:
        return "extractive"

    @property
    def model_name(self) -> str:
        return "atlas-extractive-baseline-v1"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> LLMOutput:
        del system_prompt, config
        if not self._retrieval.chunks:
            text = "No encontré información suficiente en los documentos disponibles."
            return LLMOutput(text=text, provider=self.provider_name, model=self.model_name)

        query_match = re.search(
            r"PREGUNTA ORIGINAL:\s*(.*?)\s*(?:EVIDENCIA RECUPERADA:|$)",
            user_prompt,
            flags=re.DOTALL,
        )
        query = query_match.group(1).strip() if query_match else self._retrieval.query
        query_tokens = _tokens(query)

        best: tuple[float, int, str] | None = None
        for chunk in self._retrieval.chunks:
            for sentence in _sentences(chunk.text):
                sentence_tokens = _tokens(sentence)
                overlap = (
                    len(query_tokens & sentence_tokens) / len(query_tokens)
                    if query_tokens
                    else 0.0
                )
                score = 0.75 * overlap + 0.25 * chunk.rerank_score
                candidate = (score, chunk.rank, sentence)
                if best is None or candidate[0] > best[0]:
                    best = candidate

        if best is None:
            text = "No encontré información suficiente en los documentos disponibles."
        else:
            _, rank, sentence = best
            sentence = sentence.rstrip()
            if sentence[-1:] not in ".!?":
                sentence += "."
            text = f"{sentence} [Fuente {rank}]"
        return LLMOutput(text=text, provider=self.provider_name, model=self.model_name)


@dataclass(slots=True)
class OCIProviderSettings:
    compartment_id: str
    model_id: str
    auth_mode: str = "config_file"
    config_file: str = "~/.oci/config"
    profile: str = "DEFAULT"
    region: str | None = None
    service_endpoint: str | None = None

    @classmethod
    def from_environment(cls) -> "OCIProviderSettings":
        compartment_id = os.getenv("OCI_COMPARTMENT_ID", "").strip()
        model_id = os.getenv("OCI_GENAI_MODEL_ID", "").strip()
        if not compartment_id:
            raise ValueError("Falta OCI_COMPARTMENT_ID.")
        if not model_id:
            raise ValueError("Falta OCI_GENAI_MODEL_ID.")
        return cls(
            compartment_id=compartment_id,
            model_id=model_id,
            auth_mode=os.getenv("OCI_AUTH_MODE", "config_file"),
            config_file=os.getenv("OCI_CONFIG_FILE", "~/.oci/config"),
            profile=os.getenv("OCI_CONFIG_PROFILE", "DEFAULT"),
            region=os.getenv("OCI_REGION") or None,
            service_endpoint=os.getenv("OCI_GENAI_ENDPOINT") or None,
        )


class OCIChatProvider:
    """OCI Generative AI chat provider using the official Python SDK."""

    def __init__(self, settings: OCIProviderSettings) -> None:
        self._settings = settings
        self._client: Any | None = None

    @property
    def provider_name(self) -> str:
        return "oci-generative-ai"

    @property
    def model_name(self) -> str:
        return self._settings.model_id

    def _create_client(self) -> Any:
        try:
            import oci
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise RuntimeError(
                "El SDK de OCI no está instalado. Ejecuta: pip install -r requirements.txt"
            ) from error

        mode = self._settings.auth_mode.strip().casefold()
        kwargs: dict[str, Any] = {
            "retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY,
            "timeout": (10, 120),
        }
        if self._settings.service_endpoint:
            kwargs["service_endpoint"] = self._settings.service_endpoint

        if mode == "instance_principal":
            if not self._settings.region:
                raise ValueError("OCI_REGION es obligatorio con instance_principal.")
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            return oci.generative_ai_inference.GenerativeAiInferenceClient(
                {"region": self._settings.region},
                signer=signer,
                **kwargs,
            )
        if mode != "config_file":
            raise ValueError("OCI_AUTH_MODE debe ser config_file o instance_principal.")

        config = oci.config.from_file(
            file_location=str(Path(self._settings.config_file).expanduser()),
            profile_name=self._settings.profile,
        )
        return oci.generative_ai_inference.GenerativeAiInferenceClient(
            config,
            **kwargs,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @staticmethod
    def _extract_text(response: Any) -> tuple[str, dict[str, Any]]:
        try:
            chat_response = response.data.chat_response
            choice = chat_response.choices[0]
            content = choice.message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise RuntimeError("OCI devolvió una respuesta de chat inesperada.") from error

        parts = [
            str(item.text).strip()
            for item in content
            if getattr(item, "text", None)
        ]
        text = "\n".join(part for part in parts if part).strip()
        if not text:
            raise RuntimeError("OCI no devolvió texto en la respuesta.")

        usage: dict[str, Any] = {}
        raw_usage = getattr(chat_response, "usage", None)
        if raw_usage is not None:
            try:
                import oci

                usage = oci.util.to_dict(raw_usage)
            except Exception:  # noqa: BLE001 - usage is optional metadata
                usage = {"raw": str(raw_usage)}
        return text, usage

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        config: GenerationConfig,
    ) -> LLMOutput:
        try:
            import oci
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise RuntimeError(
                "El SDK de OCI no está instalado. Ejecuta: pip install -r requirements.txt"
            ) from error

        models = oci.generative_ai_inference.models
        request = models.GenericChatRequest(
            api_format="GENERIC",
            messages=[
                models.SystemMessage(
                    role="SYSTEM",
                    content=[models.TextContent(type="TEXT", text=system_prompt)],
                ),
                models.UserMessage(
                    role="USER",
                    content=[models.TextContent(type="TEXT", text=user_prompt)],
                ),
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            is_stream=False,
            num_generations=1,
        )
        details = models.ChatDetails(
            compartment_id=self._settings.compartment_id,
            serving_mode=models.OnDemandServingMode(
                model_id=self._settings.model_id,
            ),
            chat_request=request,
        )
        response = self._get_client().chat(chat_details=details)
        text, usage = self._extract_text(response)
        return LLMOutput(
            text=text,
            provider=self.provider_name,
            model=self.model_name,
            usage=usage,
        )
