"""Reply previews saved to disk; a send must point at one and match it exactly.

A preview moves through: waiting (prepared, no timer) → armed (the card is
posted and an automatic send is scheduled) → claimed (a send is in flight) →
sent. It can also end cancelled or expired. Every transition happens under one
file lock, so the agent and the automatic sender can never both send it.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import secrets
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PREVIEW_DIR = Path(
    os.environ.get(
        "LINKEDIN_MCP_REPLY_STORE", Path.home() / ".linkedin-mcp" / "reply-store"
    )
)
# Screenshots may live apart from the records, in a folder the chat client is
# allowed to attach files from.
SHOTS_DIR = Path(os.environ.get("LINKEDIN_MCP_SHOTS", PREVIEW_DIR))
PREVIEW_TTL_SECONDS = 24 * 3600
# A re-armed preview never fires sooner than this, so the agent has time to
# read a message from the account owner that arrived just before.
MIN_AUTO_DELAY_SECONDS = 5 * 60
MAX_AUTO_DELAY_SECONDS = 12 * 3600

TRIGGERS = ("owner", "auto")


class PreviewError(ValueError):
    """The preview is missing, expired, already used or does not match."""


def preview_state(data: dict[str, Any], now: float, ttl: int) -> str:
    if data.get("used_at"):
        return "sent"
    if data.get("cancelled_at"):
        return "cancelled"
    if data.get("claimed_at"):
        return "sending"
    if now - data["created_at"] > ttl:
        return "expired"
    if data.get("auto_send_at"):
        return "armed"
    return "waiting"


class PreviewStore:
    def __init__(
        self,
        root: Path = PREVIEW_DIR,
        ttl: int = PREVIEW_TTL_SECONDS,
        shots: Path | None = None,
    ) -> None:
        self.root = root
        self.ttl = ttl
        self.shots = (
            shots if shots is not None else (SHOTS_DIR if root == PREVIEW_DIR else root)
        )

    def _path(self, preview_id: str) -> Path:
        if not preview_id or not preview_id.isalnum():
            raise PreviewError(f"invalid preview_id {preview_id!r}")
        return self.root / f"{preview_id}.json"

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self._ensure_root()
        with open(self.root / ".lock", "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _read(self, preview_id: str) -> dict[str, Any]:
        path = self._path(preview_id)
        if not path.exists():
            raise PreviewError(f"no preview {preview_id}")
        return json.loads(path.read_text())

    def _write(self, data: dict[str, Any]) -> None:
        path = self._path(data["preview_id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        os.replace(tmp, path)

    def screenshot_path(self, preview_id: str) -> Path:
        self._path(preview_id)
        self.shots.mkdir(parents=True, exist_ok=True)
        os.chmod(self.shots, 0o700)
        return self.shots / f"{preview_id}.png"

    def save(self, record: dict[str, Any], now: float | None = None) -> str:
        preview_id = secrets.token_hex(6)
        data = {
            **record,
            "preview_id": preview_id,
            "created_at": now or time.time(),
            "used_at": None,
            "cancelled_at": None,
            "claimed_at": None,
            "auto_send_at": None,
            "armed_at": None,
            "card_message_ids": [],
        }
        with self._locked():
            self._write(data)
        return preview_id

    def _require_open(self, data: dict[str, Any], now: float) -> None:
        state = preview_state(data, now, self.ttl)
        pid = data["preview_id"]
        if state == "sent":
            raise PreviewError(f"preview {pid} was already sent")
        if state == "cancelled":
            raise PreviewError(f"preview {pid} was cancelled")
        if state == "sending":
            raise PreviewError(f"preview {pid} is being sent right now")
        if state == "expired":
            raise PreviewError(f"preview {pid} expired; prepare it again")

    def load(self, preview_id: str, now: float | None = None) -> dict[str, Any]:
        data = self._read(preview_id)
        self._require_open(data, now or time.time())
        return data

    def arm(
        self,
        preview_id: str,
        card_message_ids: list[str],
        delay_seconds: int,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Schedule the automatic send. Re-arming keeps the first deadline, but
        never fires sooner than the minimum delay from now."""
        now = now or time.time()
        if not MIN_AUTO_DELAY_SECONDS <= delay_seconds <= MAX_AUTO_DELAY_SECONDS:
            raise PreviewError(
                f"delay must be between {MIN_AUTO_DELAY_SECONDS // 60} and "
                f"{MAX_AUTO_DELAY_SECONDS // 60} minutes"
            )
        ids = [str(i).strip() for i in card_message_ids if str(i).strip()]
        if not ids:
            raise PreviewError("card_message_ids must name the posted card")
        with self._locked():
            data = self._read(preview_id)
            self._require_open(data, now)
            due = now + delay_seconds
            if data.get("auto_send_at"):
                due = max(data["auto_send_at"], now + MIN_AUTO_DELAY_SECONDS)
            if due - data["created_at"] > self.ttl:
                raise PreviewError(
                    f"preview {preview_id} would expire before the automatic send; "
                    "prepare it again"
                )
            data["auto_send_at"] = due
            data["armed_at"] = now
            known = set(data.get("card_message_ids") or [])
            data["card_message_ids"] = sorted(known | set(ids))
            self._write(data)
            return data

    def cancel(
        self, preview_id: str, reason: str, now: float | None = None
    ) -> dict[str, Any]:
        now = now or time.time()
        with self._locked():
            data = self._read(preview_id)
            self._require_open(data, now)
            data["cancelled_at"] = now
            data["cancel_reason"] = reason.strip()[:300] or "cancelled"
            self._write(data)
            return data

    def claim(
        self, preview_id: str, trigger: str, now: float | None = None
    ) -> dict[str, Any]:
        """Reserve the preview for one send. An automatic send needs a schedule
        that is due; Óscar's yes needs only an open preview."""
        if trigger not in TRIGGERS:
            raise PreviewError(f"trigger must be one of {TRIGGERS}")
        now = now or time.time()
        with self._locked():
            data = self._read(preview_id)
            self._require_open(data, now)
            if trigger == "auto":
                if not data.get("auto_send_at"):
                    raise PreviewError(f"preview {preview_id} has no automatic send")
                if data["auto_send_at"] > now:
                    raise PreviewError(f"preview {preview_id} is not due yet")
            data["claimed_at"] = now
            data["trigger"] = trigger
            self._write(data)
            return data

    def release(self, preview_id: str, error: str) -> None:
        """Undo a claim after a failure that provably sent nothing."""
        with self._locked():
            data = self._read(preview_id)
            data["claimed_at"] = None
            data["last_error"] = error[:500]
            self._write(data)

    def mark_used(self, preview_id: str, result: dict[str, Any]) -> None:
        with self._locked():
            data = self._read(preview_id)
            data["used_at"] = time.time()
            data["result"] = result
            self._write(data)

    def summaries(self, now: float | None = None) -> list[dict[str, Any]]:
        now = now or time.time()
        out = []
        if not self.root.exists():
            return out
        for path in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append(
                {
                    "preview_id": data.get("preview_id"),
                    "kind": data.get("kind"),
                    "state": preview_state(data, now, self.ttl),
                    "target": data.get("thread_id"),
                    "title": data.get("title"),
                    "company": data.get("company"),
                    "created_at": data.get("created_at"),
                    "armed_at": data.get("armed_at"),
                    "auto_send_at": data.get("auto_send_at"),
                    "card_message_ids": data.get("card_message_ids") or [],
                    "trigger": data.get("trigger"),
                    "result": data.get("result"),
                    "last_error": data.get("last_error"),
                }
            )
        return sorted(out, key=lambda s: s["created_at"] or 0)
