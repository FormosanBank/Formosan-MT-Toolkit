"""DeepL API key rotation, retry policy, usage, and request batching."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import requests
from pivot_types import TranslationJob


class DeepLRuntimeError(RuntimeError):
    """Base class for DeepL runtime errors."""


class DeepLQuotaExceeded(DeepLRuntimeError):
    """Raised when DeepL reports quota exhaustion."""


class DeepLFatalError(DeepLRuntimeError):
    """Raised for non-retryable DeepL API errors."""


@dataclass
class DeepLKey:
    env_name: str
    auth_key: str
    api_base: str


class DeepLClient:
    def __init__(
        self,
        keys: list[DeepLKey],
        timeout: float,
        max_retries: int,
        retry_backoff: float,
    ) -> None:
        if not keys:
            raise ValueError("DeepLClient requires at least one API key.")
        self.keys = keys
        self.key_index = 0
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_backoff = retry_backoff
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "FormosanMT-Pivot/1.0",
            }
        )
        self._apply_current_key()

    @property
    def current_key(self) -> DeepLKey:
        return self.keys[self.key_index]

    @property
    def active_env_name(self) -> str:
        return self.current_key.env_name

    def _apply_current_key(self) -> None:
        self.session.headers["Authorization"] = (
            f"DeepL-Auth-Key {self.current_key.auth_key}"
        )

    def _advance_key(self) -> bool:
        if self.key_index + 1 >= len(self.keys):
            return False
        exhausted_name = self.current_key.env_name
        self.key_index += 1
        self._apply_current_key()
        print(
            f"DeepL key {exhausted_name} exhausted; switching to "
            f"{self.current_key.env_name}.",
            file=sys.stderr,
        )
        return True

    def translate(
        self,
        texts: list[str],
        *,
        source_lang: str,
        target_lang: str,
        split_sentences: str,
        preserve_formatting: bool,
        model_type: Optional[str],
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "text": texts,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "split_sentences": split_sentences,
            "preserve_formatting": preserve_formatting,
        }
        if model_type:
            payload["model_type"] = model_type

        last_quota_error = ""
        while True:
            url = f"{self.current_key.api_base}/v2/translate"
            last_error = ""
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.session.post(
                        url,
                        json=payload,
                        timeout=self.timeout,
                    )
                except requests.RequestException as exc:
                    last_error = str(exc)
                    if attempt == self.max_retries:
                        raise DeepLRuntimeError(
                            f"DeepL request failed: {last_error}"
                        ) from exc
                    time.sleep(self.retry_backoff * attempt)
                    continue

                if response.status_code == 200:
                    data = response.json()
                    translations = data.get("translations", [])
                    if len(translations) != len(texts):
                        raise DeepLRuntimeError(
                            "DeepL returned a different number of translations "
                            f"({len(translations)}) than inputs ({len(texts)})."
                        )
                    return translations

                body = safe_response_text(response)
                if response.status_code == 456:
                    last_quota_error = f"{self.current_key.env_name}: {body}"
                    if self._advance_key():
                        break
                    raise DeepLQuotaExceeded(
                        "All DeepL API keys exhausted. Last error: "
                        f"{last_quota_error}"
                    )
                if response.status_code in {401, 403, 404}:
                    bad_name = self.current_key.env_name
                    print(
                        f"DeepL key {bad_name} is invalid or forbidden "
                        f"(HTTP {response.status_code}); skipping it.",
                        file=sys.stderr,
                    )
                    if self._advance_key():
                        break
                    raise DeepLFatalError(
                        "All DeepL API keys failed. Last error from "
                        f"{bad_name}: HTTP {response.status_code}: {body}"
                    )
                if response.status_code == 400:
                    raise DeepLFatalError(
                        f"DeepL HTTP 400 using {self.current_key.env_name}: {body}"
                    )

                retry_after = response.headers.get("Retry-After")
                if (
                    response.status_code in {408, 409, 429, 500, 502, 503, 504}
                    and attempt < self.max_retries
                ):
                    sleep_for = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else self.retry_backoff * attempt
                    )
                    time.sleep(sleep_for)
                    last_error = f"HTTP {response.status_code}: {body}"
                    continue

                raise DeepLRuntimeError(
                    f"DeepL HTTP {response.status_code} using "
                    f"{self.current_key.env_name}: {body}"
                )
            else:
                raise DeepLRuntimeError(
                    f"DeepL request failed after retries: {last_error}"
                )

    def usage(self) -> Optional[dict[str, Any]]:
        url = f"{self.current_key.api_base}/v2/usage"
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                print(
                    "Warning: could not read DeepL usage for "
                    f"{self.current_key.env_name}: HTTP {response.status_code}",
                    file=sys.stderr,
                )
                return None
            return response.json()
        except requests.RequestException as exc:
            print(
                f"Warning: could not read DeepL usage for "
                f"{self.current_key.env_name}: {exc}",
                file=sys.stderr,
            )
            return None


def safe_response_text(response: requests.Response) -> str:
    text = response.text.strip()
    if len(text) > 500:
        text = text[:500] + "..."
    return text or response.reason


def choose_deepl_api_base(auth_key: str, override: Optional[str]) -> str:
    if override:
        return override.rstrip("/")
    if auth_key.endswith(":fx"):
        return "https://api-free.deepl.com"
    return "https://api.deepl.com"


def discover_api_key_envs(
    environ: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Return configured DEEPL_API_KEY variables in stable numeric order."""
    source = os.environ if environ is None else environ
    names: list[tuple[int, str]] = []
    for env_name, value in source.items():
        match = re.fullmatch(r"DEEPL_API_KEY(?:_(\d+))?", env_name)
        if match and str(value).strip():
            suffix = int(match.group(1) or 1)
            names.append((suffix, env_name))
    return [
        env_name
        for _, env_name in sorted(names, key=lambda item: (item[0], item[1]))
    ]


def parse_api_key_envs(raw: str) -> list[str]:
    if str(raw or "").strip().lower() == "auto":
        return discover_api_key_envs()
    envs = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    output: list[str] = []
    seen: set[str] = set()
    for env_name in envs:
        if env_name not in seen:
            output.append(env_name)
            seen.add(env_name)
    return output


def load_deepl_keys(
    env_names: list[str],
    api_base_override: Optional[str],
) -> list[DeepLKey]:
    keys: list[DeepLKey] = []
    for env_name in env_names:
        auth_key = os.getenv(env_name, "").strip()
        if not auth_key:
            continue
        keys.append(
            DeepLKey(
                env_name=env_name,
                auth_key=auth_key,
                api_base=choose_deepl_api_base(auth_key, api_base_override),
            )
        )
    return keys


def read_deepl_usage_for_key(
    key: DeepLKey,
    timeout: float,
) -> Optional[dict[str, Any]]:
    headers = {"Authorization": f"DeepL-Auth-Key {key.auth_key}"}
    try:
        response = requests.get(
            f"{key.api_base}/v2/usage",
            headers=headers,
            timeout=timeout,
        )
        if response.status_code != 200:
            print(
                f"Warning: could not read DeepL usage for {key.env_name}: "
                f"HTTP {response.status_code}",
                file=sys.stderr,
            )
            return None
        usage = response.json()
        usage["api_key_env"] = key.env_name
        return usage
    except requests.RequestException as exc:
        print(
            f"Warning: could not read DeepL usage for {key.env_name}: {exc}",
            file=sys.stderr,
        )
        return None


def request_body_size(
    texts: list[str],
    *,
    source_lang: str,
    target_lang: str,
    split_sentences: str,
    preserve_formatting: bool,
    model_type: Optional[str],
) -> int:
    payload: dict[str, Any] = {
        "text": texts,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "split_sentences": split_sentences,
        "preserve_formatting": preserve_formatting,
    }
    if model_type:
        payload["model_type"] = model_type
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def batch_jobs(
    jobs: Iterable[TranslationJob],
    *,
    max_texts: int,
    max_request_bytes: int,
    source_lang: str,
    target_lang: str,
    split_sentences: str,
    preserve_formatting: bool,
    model_type: Optional[str],
) -> Iterable[list[TranslationJob]]:
    batch: list[TranslationJob] = []
    for job in jobs:
        one_size = request_body_size(
            [job.text],
            source_lang=source_lang,
            target_lang=target_lang,
            split_sentences=split_sentences,
            preserve_formatting=preserve_formatting,
            model_type=model_type,
        )
        if one_size > max_request_bytes:
            if batch:
                yield batch
                batch = []
            yield [job]
            continue

        candidate = [*batch, job]
        candidate_size = request_body_size(
            [item.text for item in candidate],
            source_lang=source_lang,
            target_lang=target_lang,
            split_sentences=split_sentences,
            preserve_formatting=preserve_formatting,
            model_type=model_type,
        )
        if batch and (
            len(candidate) > max_texts or candidate_size > max_request_bytes
        ):
            yield batch
            batch = [job]
        else:
            batch = candidate

    if batch:
        yield batch
