"""Load PIL images from http(s) URLs or s3:// URIs."""

import io
import os
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image

_REQUEST_TIMEOUT = int(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
_MAX_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))  # 25 MB


class ImageLoadError(Exception):
    """Raised when an image cannot be fetched or decoded."""


_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    return _s3_client


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ImageLoadError(f"Invalid s3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _fetch_s3(uri: str) -> bytes:
    bucket, key = _parse_s3_uri(uri)
    try:
        obj = _get_s3_client().get_object(Bucket=bucket, Key=key)
    except (BotoCoreError, ClientError) as e:
        raise ImageLoadError(f"S3 fetch failed for {uri}: {e}") from e

    body = obj["Body"].read(_MAX_BYTES + 1)
    if len(body) > _MAX_BYTES:
        raise ImageLoadError(f"Image exceeds max size of {_MAX_BYTES} bytes")
    return body


def _fetch_http(url: str) -> bytes:
    try:
        with requests.get(url, stream=True, timeout=_REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            chunks = []
            total = 0
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > _MAX_BYTES:
                    raise ImageLoadError(f"Image exceeds max size of {_MAX_BYTES} bytes")
                chunks.append(chunk)
            return b"".join(chunks)
    except requests.RequestException as e:
        raise ImageLoadError(f"HTTP fetch failed for {url}: {e}") from e


def load_image(url: str) -> Image.Image:
    """Download an image from an http(s) URL or s3:// URI and return a PIL Image."""
    if not url or not isinstance(url, str):
        raise ImageLoadError("url must be a non-empty string")

    scheme = urlparse(url).scheme.lower()
    if scheme == "s3":
        data = _fetch_s3(url)
    elif scheme in ("http", "https"):
        data = _fetch_http(url)
    else:
        raise ImageLoadError(f"Unsupported URL scheme: {scheme!r}")

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (OSError, Image.UnidentifiedImageError) as e:
        raise ImageLoadError(f"Could not decode image from {url}: {e}") from e
