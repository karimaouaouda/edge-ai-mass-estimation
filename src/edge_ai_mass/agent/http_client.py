"""HTTP client for Laravel edge ingestion endpoints."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class EdgeHttpError(RuntimeError):
    """Raised when the backend rejects or cannot process a request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.status_code is None or self.status_code >= 500


@dataclass(frozen=True, slots=True)
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    body: bytes
    headers: dict[str, str]

    def json(self) -> dict[str, Any]:
        if not self.body:
            return {}
        data = json.loads(self.body.decode("utf-8"))
        if not isinstance(data, dict):
            raise EdgeHttpError("Backend response was not a JSON object")
        return data


Transport = Callable[[HttpRequest, float], HttpResponse]


class UrllibTransport:
    """Small urllib-based transport to avoid a hard runtime dependency."""

    def __call__(self, request: HttpRequest, timeout: float) -> HttpResponse:
        urllib_request = urllib.request.Request(
            request.url,
            data=request.body,
            headers=request.headers,
            method=request.method,
        )
        try:
            with urllib.request.urlopen(urllib_request, timeout=timeout) as response:
                return HttpResponse(
                    status_code=response.status,
                    body=response.read(),
                    headers={key: value for key, value in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                status_code=exc.code,
                body=exc.read(),
                headers={key: value for key, value in exc.headers.items()},
            )


class EdgeHttpClient:
    """Client for `/api/edge/devices/{device_id}` endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        device_id: str,
        device_token: str,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.device_id = device_id
        self.device_token = device_token
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = retry_backoff_seconds
        self.transport = transport or UrllibTransport()
        self.sleep = sleep

    def submit_telemetry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit heartbeat/resource state to Laravel."""

        body = json.dumps(payload).encode("utf-8")
        print(f"Submitting telemetry to {self._url('telemetry')} with payload: {payload}")
        request = HttpRequest(
            method="POST",
            url=self._url('telemetry'),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Device-Token": self.device_token,
            },
            body=body,
        )
        return self._send_json(request)

    def upload_media(
        self,
        path: str | Path,
        *,
        media_type: str,
        request_id: str | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        sha256: str | None = None,
    ) -> dict[str, Any]:
        """Upload an image file as multipart/form-data."""

        file_path = Path(path)
        checksum = sha256 or sha256_file(file_path)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        fields: dict[str, str] = {
            "media_type": media_type,
            "sha256": checksum,
        }
        if request_id:
            fields["request_id"] = request_id
        if correlation_id:
            fields["correlation_id"] = correlation_id
        fields.update(_metadata_fields(metadata or {}))
        body, multipart_type = _multipart_body(
            fields=fields,
            file_field="file",
            file_path=file_path,
            file_content_type=content_type,
        )
        request = HttpRequest(
            method="POST",
            url=self._url("media"),
            headers={
                "Accept": "application/json",
                "Content-Type": multipart_type,
                "X-Device-Token": self.device_token,
            },
            body=body,
        )
        return self._send_json(request)

    def _send_json(self, request: HttpRequest) -> dict[str, Any]:
        last_error: EdgeHttpError | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.transport(request, self.timeout_seconds)
            except (OSError, TimeoutError) as exc:
                last_error = EdgeHttpError(f"Network error while calling backend: {exc}")
            else:
                if 200 <= response.status_code < 300:
                    return response.json()
                message = _response_error_message(response)
                error = EdgeHttpError(message, status_code=response.status_code)
                if not _should_retry_status(response.status_code):
                    raise error
                last_error = error

            if attempt < self.max_retries:
                self.sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))

        if last_error is not None:
            raise last_error
        raise EdgeHttpError("Backend request failed")

    def _url(self, suffix: str) -> str:
        device_id = urllib.parse.quote(self.device_id, safe="")
        print(f"Constructing URL for device_id: {device_id} with suffix: {suffix} : {self.base_url}/api/edge/devices/{device_id}/{suffix}")
        return f"{self.base_url}/api/edge/devices/{device_id}/{suffix}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_retry_status(status_code: int) -> bool:
    return status_code >= 500 or status_code in {408, 429}


def _response_error_message(response: HttpResponse) -> str:
    body = response.body.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = body
    if isinstance(payload, dict):
        detail = payload.get("message") or payload.get("error") or payload
    else:
        detail = payload
    return f"Backend returned HTTP {response.status_code}: {detail}"


def _metadata_fields(metadata: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in metadata.items():
        field_name = f"metadata[{key}]"
        if isinstance(value, (dict, list)):
            fields[field_name] = json.dumps(value, separators=(",", ":"))
        elif value is None:
            fields[field_name] = ""
        else:
            fields[field_name] = str(value)
    return fields


def _multipart_body(
    *,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    file_content_type: str,
) -> tuple[bytes, str]:
    boundary = f"----edge-ai-mass-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    filename = file_path.name
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {file_content_type}\r\n\r\n".encode("ascii"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
