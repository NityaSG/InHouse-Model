"""GPT-Vision fallback classifier via Azure OpenAI — structured tool-call output."""

import base64
import io
import json
import os
import time

from openai import OpenAI, OpenAIError
from PIL import Image

_MAX_RETRIES = int(os.getenv("GPT_MAX_RETRIES", "3"))
_RETRY_DELAY = float(os.getenv("GPT_RETRY_DELAY_SECONDS", "1.0"))

_SYSTEM_PROMPT = (
    "You are a fashion expert. Classify the image provided by the user "
    "by calling the supplied function with the correct arguments."
)

_USER_PROMPTS = {
    "gender": "Identify the gender of the subject or the intended gender audience for the clothing in this image.",
    "category": "Classify this fashion/retail image into the most appropriate product category.",
}

_TOOL_NAME = "classify_image"


class GPTFallbackError(Exception):
    pass


# ── Image encoding ─────────────────────────────────────────────────────────────
def _pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _resolve_image_url(url: str, preloaded: Image.Image | None) -> str:
    """Return a URL or data-URL for the chat completions vision content block.

    S3 URIs and pre-loaded PIL images are encoded to base64 because they are
    not publicly reachable by the Azure OpenAI service.
    """
    if preloaded is not None:
        return _pil_to_data_url(preloaded)
    if url.startswith("s3://"):
        from classifiers.image_loader import ImageLoadError, load_image
        try:
            return _pil_to_data_url(load_image(url))
        except ImageLoadError as e:
            raise GPTFallbackError(f"Could not load S3 image for GPT: {e}") from e
    return url


# ── Tool schema builder ────────────────────────────────────────────────────────
def _build_tool(classifier_type: str, classes: list[str]) -> dict:
    """Build a chat-completions tool definition with an enum-constrained prediction field."""
    descriptions = {
        "gender": "Classify the gender of the subject or clothing's intended audience.",
        "category": "Classify the fashion/retail product into the appropriate category.",
    }
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": descriptions[classifier_type],
            "parameters": {
                "type": "object",
                "properties": {
                    "prediction": {
                        "type": "string",
                        "enum": classes,
                        "description": "The predicted class. Must be exactly one of the enum values.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in the prediction (0.0 = uncertain, 1.0 = certain).",
                    },
                },
                "required": ["prediction", "confidence"],
            },
        },
    }


# ── Retry wrapper ──────────────────────────────────────────────────────────────
def _create_completion_with_retries(client: OpenAI, **kwargs) -> object:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except OpenAIError as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (2 ** attempt))
    raise GPTFallbackError(f"Azure OpenAI request failed after {_MAX_RETRIES} attempts: {last_exc}") from last_exc


def _parse_tool_arguments(response: object, classes: list[str]) -> tuple[str, float]:
    """Extract prediction + confidence from the first tool call."""
    try:
        args_raw = response.choices[0].message.tool_calls[0].function.arguments
        args = json.loads(args_raw)
    except (AttributeError, IndexError, KeyError, json.JSONDecodeError) as e:
        raise GPTFallbackError(f"Could not parse tool call arguments: {e}") from e

    prediction = str(args.get("prediction", "")).strip()
    if prediction not in classes:
        lower_map = {c.lower(): c for c in classes}
        prediction = lower_map.get(prediction.lower(), classes[0])

    confidence = max(0.0, min(1.0, float(args.get("confidence", 0.9))))
    return prediction, confidence


# ── Main class ─────────────────────────────────────────────────────────────────
class GPTFallback:
    """Azure OpenAI vision classifier using structured tool-call output."""

    def __init__(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise GPTFallbackError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set to use GPT fallback"
            )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.client = OpenAI(base_url=endpoint, api_key=api_key)

    def classify(
        self,
        classifier_type: str,
        classes: list[str],
        url: str,
        preloaded_image: Image.Image | None = None,
    ) -> dict:
        """Classify a single image using GPT vision with structured tool output."""
        image_url = _resolve_image_url(url, preloaded_image)
        tool = _build_tool(classifier_type, classes)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _USER_PROMPTS[classifier_type],
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                ],
            },
        ]

        response = _create_completion_with_retries(
            self.client,
            model=self.deployment,
            temperature=0,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
        )

        prediction, confidence = _parse_tool_arguments(response, classes)
        n = len(classes)
        remainder = (1.0 - confidence) / (n - 1) if n > 1 else 0.0
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                c: confidence if c == prediction else remainder for c in classes
            },
            "source": "gpt",
            "tta": False,
        }
