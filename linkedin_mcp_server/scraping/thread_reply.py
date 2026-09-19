"""Reply inside an existing LinkedIn messaging thread.

``send_message`` opens LinkedIn's profile-based compose flow, which may start a
separate conversation instead of answering an InMail or an existing thread.
This module answers where the counterpart wrote: it opens
``/messaging/thread/<id>/``, types the reply into the one visible composer of
that page, reads the composer back, and only then submits. Line breaks are
typed as Shift+Enter so the sent message keeps its paragraphs.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.identifiers import (
    messaging_thread_url,
    normalize_thread_id,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

logger = logging.getLogger(__name__)

_COMPOSER_SELECTOR = '[role="textbox"][contenteditable="true"]'
_MAX_MESSAGE_LENGTH = 8000
_COMPOSER_TIMEOUT_MS = 15_000
_SUBMIT_READY_TIMEOUT_MS = 5_000
_CONFIRMATION_TIMEOUT_MS = 20_000
_PREVIEW_DIR = Path.home() / ".linkedin-mcp" / "reply-previews"


def normalize_reply_message(message: str) -> str:
    """Canonical form of a multi-line reply: LF only, no trailing spaces."""
    text = message.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def refuse_an_invalid_reply(thread_id: str, message: str) -> dict[str, Any] | None:
    """Browser-free refusal for a reply that must never reach the composer."""
    reason = None
    normalized = normalize_reply_message(message)
    if not normalized.strip():
        reason = "Message must contain non-whitespace characters."
    elif any(
        (ord(character) < 32 and character != "\n") or ord(character) == 127
        for character in normalized
    ):
        reason = "Message must not contain control characters other than line breaks."
    elif len(normalized) > _MAX_MESSAGE_LENGTH:
        reason = f"Message must be at most {_MAX_MESSAGE_LENGTH} characters."
    if reason is None:
        return None
    return thread_reply_result(
        messaging_thread_url(normalize_thread_id(thread_id), "/"),
        normalize_thread_id(thread_id),
        "invalid_message",
        reason,
    )


def thread_reply_result(
    url: str,
    thread_id: str,
    status: str,
    message: str,
    *,
    composer_text: str | None = None,
    preview_path: str | None = None,
    sent: bool = False,
    retry_safe: bool = True,
) -> dict[str, Any]:
    """Structured response for ``reply_to_thread``.

    ``sent`` is true only after the typed text was seen in the thread's message
    list and the composer went empty. ``retry_safe`` is false from the moment
    the submit button was clicked, whatever happened afterwards.
    """
    result: dict[str, Any] = {
        "url": url,
        "thread_id": thread_id,
        "status": status,
        "message": message,
        "sent": sent,
        "retry_safe": retry_safe,
    }
    if composer_text is not None:
        result["composer_text"] = composer_text
    if preview_path is not None:
        result["preview_path"] = preview_path
    return result


def _words(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def composer_matches(expected: str, composer_text: str) -> bool:
    """True when the composer holds exactly the expected words and lines."""
    expected_lines = _non_empty_lines(expected)
    actual_lines = _non_empty_lines(composer_text)
    return expected_lines == actual_lines and _words(expected) == _words(composer_text)


class ThreadReplier:
    """Type and submit a reply inside one messaging thread page."""

    def __init__(self, session: ScrapingSession, navigator: PageNavigator):
        self._session = session
        self._navigator = navigator

    @property
    def _page(self) -> Any:
        return self._session.page

    async def _composer(self) -> Any | None:
        """The one visible composer of the thread page, or None."""
        locator = self._page.locator(f"{_COMPOSER_SELECTOR}:visible")
        deadline = time.monotonic() + _COMPOSER_TIMEOUT_MS / 1_000
        while True:
            try:
                count = await locator.count()
            except Exception:
                logger.debug("Could not count thread composers", exc_info=True)
                count = 0
            if count == 1:
                return locator.first
            if count > 1 or time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.25)

    @staticmethod
    async def _submit_button(composer: Any) -> Any | None:
        """The submit button of the form that owns the composer."""
        form = composer.locator("xpath=ancestor::form[1]")
        try:
            if await form.count() != 1:
                return None
        except Exception:
            return None
        button = form.locator('button[type="submit"]:visible')
        try:
            if await button.count() != 1:
                return None
        except Exception:
            return None
        return button.first

    async def _clear_composer(self, composer: Any) -> None:
        await composer.click()
        await self._page.keyboard.press("ControlOrMeta+A")
        await self._page.keyboard.press("Delete")

    async def _type_reply(self, composer: Any, message: str) -> None:
        """Type the reply line by line; Shift+Enter keeps it in one message."""
        await self._clear_composer(composer)
        lines = message.split("\n")
        for index, line in enumerate(lines):
            if index:
                await self._page.keyboard.press("Shift+Enter")
            if line:
                await self._page.keyboard.type(line)

    async def _read_composer(self, composer: Any) -> str:
        try:
            text = await composer.inner_text()
        except Exception:
            logger.debug("Could not read the thread composer", exc_info=True)
            return ""
        return normalize_reply_message(text)

    async def _wait_for_submit_ready(self, button: Any) -> bool:
        deadline = time.monotonic() + _SUBMIT_READY_TIMEOUT_MS / 1_000
        while True:
            try:
                if await button.is_enabled():
                    return True
            except Exception:
                return False
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.1)

    async def _save_preview(self, thread_id: str) -> str | None:
        """Screenshot of the filled composer, for a human check before a yes."""
        try:
            _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            _PREVIEW_DIR.chmod(0o700)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", thread_id)[:40]
            path = _PREVIEW_DIR / f"{stamp}-{safe_id}.png"
            await self._page.screenshot(path=str(path), full_page=False)
            return str(path)
        except Exception:
            logger.debug("Could not save the composer preview", exc_info=True)
            return None

    async def _main_text(self) -> str:
        try:
            return await self._page.evaluate(
                "() => (document.querySelector('main') || document.body).innerText"
            )
        except Exception:
            return ""

    async def _wait_for_sent(self, composer: Any, message: str) -> bool:
        """The reply is sent once it shows in the thread and the composer is empty."""
        expected = _words(message)
        deadline = time.monotonic() + _CONFIRMATION_TIMEOUT_MS / 1_000
        while True:
            composer_text = await self._read_composer(composer)
            main_text = _words(await self._main_text())
            if not composer_text.strip() and expected in main_text:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.5)

    async def reply_to_thread(
        self,
        thread_id: str,
        message: str,
        *,
        confirm_send: bool,
    ) -> dict[str, Any]:
        """Type ``message`` into the composer of thread ``thread_id`` and send it.

        With ``confirm_send`` false the text is typed, read back, captured in a
        screenshot and removed again; nothing leaves. With ``confirm_send`` true
        the same text is typed and read back, and submitted only when the
        composer holds exactly the expected lines.
        """
        refusal = refuse_an_invalid_reply(thread_id, message)
        if refusal is not None:
            return refusal
        thread_id = normalize_thread_id(thread_id)
        message = normalize_reply_message(message)
        thread_url = messaging_thread_url(thread_id, "/")

        await self._navigator._navigate_to_page(thread_url)
        await self._session.check_rate_limit()
        try:
            await self._page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("Thread page did not load for %s", thread_id)
        await self._session.dismiss_modal()

        if (
            thread_id not in self._page.url
            and thread_id.rstrip("=") not in self._page.url
        ):
            return thread_reply_result(
                self._page.url,
                thread_id,
                "thread_not_found",
                "LinkedIn did not open the requested thread; the id may be stale. "
                "Read the thread with get_conversation first.",
            )

        composer = await self._composer()
        if composer is None:
            return thread_reply_result(
                self._page.url,
                thread_id,
                "composer_unavailable",
                "The thread page does not expose exactly one message composer. "
                "The conversation may not accept replies.",
            )
        button = await self._submit_button(composer)
        if button is None:
            return thread_reply_result(
                self._page.url,
                thread_id,
                "composer_unavailable",
                "The composer has no single submit button in its form.",
            )

        await self._type_reply(composer, message)
        composer_text = await self._read_composer(composer)
        if not composer_matches(message, composer_text):
            await self._clear_composer(composer)
            return thread_reply_result(
                self._page.url,
                thread_id,
                "composer_mismatch",
                "The composer did not hold the expected text after typing; "
                "nothing was sent.",
                composer_text=composer_text,
            )
        preview_path = await self._save_preview(thread_id)

        if not confirm_send:
            await self._clear_composer(composer)
            return thread_reply_result(
                self._page.url,
                thread_id,
                "dry_run",
                "Typed and verified in the thread composer, then removed; "
                "set confirm_send=True to send.",
                composer_text=composer_text,
                preview_path=preview_path,
            )

        if not await self._wait_for_submit_ready(button):
            await self._clear_composer(composer)
            return thread_reply_result(
                self._page.url,
                thread_id,
                "submit_unavailable",
                "The submit button never became enabled; nothing was sent.",
                composer_text=composer_text,
                preview_path=preview_path,
            )

        await button.click()
        sent = await self._wait_for_sent(composer, message)
        await self._session.check_rate_limit()
        if sent:
            return thread_reply_result(
                self._page.url,
                thread_id,
                "sent",
                "Reply submitted and observed in the thread.",
                composer_text=composer_text,
                preview_path=preview_path,
                sent=True,
                retry_safe=False,
            )
        return thread_reply_result(
            self._page.url,
            thread_id,
            "unconfirmed",
            "Submit was clicked but the reply was not observed in the thread "
            "within the timeout. Read the thread before retrying.",
            composer_text=composer_text,
            preview_path=preview_path,
            sent=False,
            retry_safe=False,
        )
