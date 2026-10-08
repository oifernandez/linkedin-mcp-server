"""Reply inside an existing LinkedIn messaging thread.

``send_message`` opens LinkedIn's profile-based compose flow, which may start a
separate conversation instead of answering an InMail or an existing thread.
This module answers where the counterpart wrote: it opens
``/messaging/thread/<id>/``, types the reply into the one visible composer of
that page, reads the composer back, and only then submits. Line breaks are
typed as Shift+Enter so the sent message keeps its paragraphs.

Files can travel with the reply. They are taken only from configured
directories, attached through the composer's own file inputs, and checked by
name in the composer before and in the thread after the submit.

Each message of the thread carries a ``data-event-urn``. The newest one is
returned with every result, and a caller that passes it back as
``expected_event_urn`` gets ``thread_changed`` instead of a send when anybody
wrote in the thread in between.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Sequence
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
_DIALOG_SELECTOR = '[role="dialog"], [role="alertdialog"], .artdeco-modal'
_CONTACT_PROMPT_PATTERN = re.compile(
    r"informaci[oó]n de contacto|contact (?:info|details)", re.IGNORECASE
)
_DECLINE_PATTERN = re.compile(r"^no\b", re.IGNORECASE)

ATTACHMENT_DIRS_ENV = "LINKEDIN_MCP_ATTACHMENT_DIRS"
_DEFAULT_ATTACHMENT_DIR = Path.home() / ".linkedin-mcp" / "attachments"
_ATTACHMENT_SUFFIXES = frozenset(
    {".pdf", ".doc", ".docx", ".txt", ".ppt", ".pptx", ".xls", ".xlsx", ".png", ".jpg"}
)
_MAX_ATTACHMENTS = 5
_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
_ATTACHMENT_TIMEOUT_MS = 60_000
_ATTACHMENT_NAME_SELECTOR = (
    "[data-test-msg-cross-pillar-uploaded-attachments-list-presenter__attachment-name]"
)
_ATTACHMENT_DONE_SELECTOR = (
    "[data-test-msg-cross-pillar-uploaded-attachments-list__attached]"
)
_ATTACHMENT_REMOVE_SELECTOR = (
    "[data-test-msg-cross-pillar-uploaded-attachment-list-presenter__remove-attachment]"
)
_NEWEST_EVENT_JS = """() => {
  const events = [...document.querySelectorAll('[data-event-urn]')]
    .filter(e => !e.closest('form'));
  return events.length ? events[events.length - 1].getAttribute('data-event-urn') : null;
}"""


def attachment_dirs() -> list[Path]:
    """Directories a reply may attach files from, resolved."""
    raw = os.environ.get(ATTACHMENT_DIRS_ENV, "")
    dirs = [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]
    return [d.resolve() for d in (dirs or [_DEFAULT_ATTACHMENT_DIR])]


def resolve_attachments(paths: Sequence[str] | None) -> list[Path]:
    """The files to attach, or ValueError naming the first one refused.

    Only regular files inside a configured directory, with a document or image
    suffix and under the size limit, so a path from a message can never make
    the browser upload something else from the disk.
    """
    if not paths:
        return []
    if len(paths) > _MAX_ATTACHMENTS:
        raise ValueError(f"At most {_MAX_ATTACHMENTS} attachments per reply.")
    allowed = attachment_dirs()
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not any(path.is_relative_to(d) for d in allowed):
            raise ValueError(f"{raw} is outside the attachment directories.")
        if not path.is_file():
            raise ValueError(f"{raw} is not a file.")
        if path.suffix.lower() not in _ATTACHMENT_SUFFIXES:
            raise ValueError(f"{raw} has a file type that is not allowed.")
        if path.stat().st_size > _MAX_ATTACHMENT_BYTES:
            raise ValueError(f"{raw} is larger than 20 MB.")
        if path.name in {f.name for f in files}:
            raise ValueError(f"{path.name} is attached twice.")
        files.append(path)
    return files


def pick_file_input(accepts: list[str], suffix: str) -> int:
    """Index of the composer file input that accepts ``suffix``.

    The thread form has one input for images and one for documents; the
    document one lists the suffix in its ``accept`` attribute.
    """
    for index, accept in enumerate(accepts):
        parts = [p.strip().lower() for p in accept.split(",")]
        if suffix.lower() in parts:
            return index
    for index, accept in enumerate(accepts):
        if suffix.lower() in {".png", ".jpg"} and "image/*" in accept:
            return index
    return len(accepts) - 1


def normalize_reply_message(message: str) -> str:
    """Canonical form of a multi-line reply: LF only, no trailing spaces."""
    text = message.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def refuse_an_invalid_reply(
    thread_id: str, message: str, attachments: Sequence[str] | None = None
) -> dict[str, Any] | None:
    """Browser-free refusal for a reply that must never reach the composer."""
    reason = None
    status = "invalid_message"
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
    else:
        try:
            resolve_attachments(attachments)
        except ValueError as error:
            reason = str(error)
            status = "invalid_attachment"
    if reason is None:
        return None
    return thread_reply_result(
        messaging_thread_url(normalize_thread_id(thread_id), "/"),
        normalize_thread_id(thread_id),
        status,
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
    diagnostics: dict[str, Any] | None = None,
    thread_event_urn: str | None = None,
    attachments: list[str] | None = None,
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
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    if thread_event_urn is not None:
        result["thread_event_urn"] = thread_event_urn
    if attachments:
        result["attachments"] = attachments
    return result


def _words(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def is_contact_prompt(text: str) -> bool:
    """True for LinkedIn's dialog asking to share email and phone with the sender."""
    return bool(_CONTACT_PROMPT_PATTERN.search(text))


def pick_decline_label(labels: list[str]) -> str | None:
    """The dialog button that answers without sharing anything, or None."""
    for label in labels:
        cleaned = _words(label)
        if _DECLINE_PATTERN.match(cleaned):
            return cleaned
    return None


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

    async def _newest_event_urn(self) -> str | None:
        """URN of the newest message in the thread, outside the composer."""
        try:
            return await self._page.evaluate(_NEWEST_EVENT_JS)
        except Exception:
            logger.debug("Could not read the thread events", exc_info=True)
            return None

    async def _attachment_names(self, form: Any) -> list[str]:
        try:
            names = await form.locator(_ATTACHMENT_NAME_SELECTOR).all_inner_texts()
        except Exception:
            return []
        return [_words(name) for name in names if _words(name)]

    async def _attachments_done(self, form: Any) -> int:
        try:
            return await form.locator(_ATTACHMENT_DONE_SELECTOR).count()
        except Exception:
            return 0

    async def _remove_attachments(self, form: Any) -> None:
        """Take every attached file out of the composer."""
        remove = form.locator(_ATTACHMENT_REMOVE_SELECTOR)
        for _ in range(_MAX_ATTACHMENTS * 2):
            try:
                if await remove.count() == 0:
                    return
                await remove.first.click()
            except Exception:
                logger.debug("Could not remove an attachment", exc_info=True)
                return
            await asyncio.sleep(0.5)

    async def _attach(self, form: Any, files: list[Path]) -> list[str]:
        """Attach ``files`` through the form's file inputs and wait for the upload.

        Returns the names the composer shows once every file reports attached;
        an incomplete list means the wait ran out.
        """
        inputs = form.locator('input[type="file"]')
        accepts = [
            (await inputs.nth(i).get_attribute("accept")) or ""
            for i in range(await inputs.count())
        ]
        if not accepts:
            return []
        for path in files:
            index = pick_file_input(accepts, path.suffix)
            await inputs.nth(index).set_input_files(str(path))
        expected = {path.name for path in files}
        deadline = time.monotonic() + _ATTACHMENT_TIMEOUT_MS / 1_000
        while True:
            names = await self._attachment_names(form)
            if expected <= set(names) and await self._attachments_done(form) >= len(
                files
            ):
                return names
            if time.monotonic() >= deadline:
                return names
            await asyncio.sleep(0.5)

    async def _main_text(self) -> str:
        try:
            return await self._page.evaluate(
                "() => (document.querySelector('main') || document.body).innerText"
            )
        except Exception:
            return ""

    async def _body_text(self) -> str:
        try:
            return await self._page.evaluate("() => document.body.innerText")
        except Exception:
            return ""

    def _contact_prompt_locator(self) -> Any:
        return (
            self._page.locator(_DIALOG_SELECTOR)
            .filter(has_text=_CONTACT_PROMPT_PATTERN)
            .first
        )

    async def _contact_prompt(self) -> dict[str, Any] | None:
        """The visible contact-sharing dialog with its button labels, or None."""
        dialog = self._contact_prompt_locator()
        try:
            if not await dialog.is_visible():
                return None
            text = _words(await dialog.inner_text())
            labels = await dialog.locator("button").all_inner_texts()
        except Exception:
            logger.debug("Could not inspect the contact prompt", exc_info=True)
            return None
        return {
            "text": text[:200],
            "buttons": [_words(label) for label in labels if _words(label)],
        }

    async def _decline_contact_prompt(self, prompt: dict[str, Any]) -> bool:
        """Answer the contact-sharing dialog without sharing; True when clicked."""
        label = pick_decline_label(prompt["buttons"])
        if label is None:
            return False
        dialog = self._contact_prompt_locator()
        try:
            await dialog.get_by_role("button", name=label, exact=True).first.click()
        except Exception:
            logger.debug("Could not decline the contact prompt", exc_info=True)
            return False
        return True

    async def _observe_submission(
        self,
        composer: Any,
        button: Any,
        message: str,
        attachment_names: Sequence[str] = (),
        form: Any = None,
    ) -> dict[str, Any]:
        """Watch the page after the click until the reply shows, or say why not.

        LinkedIn sometimes answers the click with a dialog asking to share the
        sender's email and phone; it is declined and the wait starts again.
        With attachments the reply counts as shown only once the composer has
        let go of them and each file name appears in the thread.
        """
        expected = _words(message)
        deadline = time.monotonic() + _CONFIRMATION_TIMEOUT_MS / 1_000
        outcome: dict[str, Any] = {"sent": False, "contact_prompt": None}
        while True:
            composer_text = await self._read_composer(composer)
            main_text = _words(await self._main_text())
            files_shown = True
            if attachment_names:
                still_attached = await self._attachment_names(form)
                files_shown = not still_attached and all(
                    name in main_text for name in attachment_names
                )
            if not composer_text.strip() and expected in main_text and files_shown:
                outcome["sent"] = True
                return outcome
            if outcome["contact_prompt"] is None:
                prompt = await self._contact_prompt()
                if prompt is not None:
                    declined = await self._decline_contact_prompt(prompt)
                    outcome["contact_prompt"] = {**prompt, "declined": declined}
                    if declined:
                        deadline = time.monotonic() + _CONFIRMATION_TIMEOUT_MS / 1_000
            if time.monotonic() >= deadline:
                outcome["composer_text_after"] = composer_text
                outcome["contact_prompt_in_page"] = is_contact_prompt(
                    await self._body_text()
                )
                try:
                    outcome["submit_enabled"] = await button.is_enabled()
                except Exception:
                    outcome["submit_enabled"] = None
                return outcome
            await asyncio.sleep(0.5)

    async def reply_to_thread(
        self,
        thread_id: str,
        message: str,
        *,
        confirm_send: bool,
        attachments: Sequence[str] | None = None,
        expected_event_urn: str | None = None,
    ) -> dict[str, Any]:
        """Type ``message`` into the composer of thread ``thread_id`` and send it.

        With ``confirm_send`` false the text is typed, the files attached, both
        read back, captured in a screenshot and removed again; nothing leaves.
        With ``confirm_send`` true the same is done and submitted only when the
        composer holds exactly the expected lines and files. A non-None
        ``expected_event_urn`` that no longer names the newest message stops
        the reply before anything is typed.
        """
        refusal = refuse_an_invalid_reply(thread_id, message, attachments)
        if refusal is not None:
            return refusal
        files = resolve_attachments(attachments)
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

        event_urn = await self._newest_event_urn()
        if expected_event_urn is not None and event_urn != expected_event_urn:
            return thread_reply_result(
                self._page.url,
                thread_id,
                "thread_changed",
                "Someone wrote in the thread after the reply was prepared; "
                "nothing was typed or sent. Read the thread and prepare again.",
                thread_event_urn=event_urn,
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
        form = composer.locator("xpath=ancestor::form[1]")

        async def abandon() -> None:
            if files:
                await self._remove_attachments(form)
            await self._clear_composer(composer)

        await self._type_reply(composer, message)
        composer_text = await self._read_composer(composer)
        if not composer_matches(message, composer_text):
            await abandon()
            return thread_reply_result(
                self._page.url,
                thread_id,
                "composer_mismatch",
                "The composer did not hold the expected text after typing; "
                "nothing was sent.",
                composer_text=composer_text,
            )

        attached: list[str] = []
        if files:
            attached = await self._attach(form, files)
            if sorted(attached) != sorted(path.name for path in files):
                await abandon()
                return thread_reply_result(
                    self._page.url,
                    thread_id,
                    "attachment_failed",
                    "The composer did not show every file as attached within the "
                    f"wait; it showed {attached}. Nothing was sent.",
                    composer_text=composer_text,
                    attachments=attached,
                )
        preview_path = await self._save_preview(thread_id)

        if not confirm_send:
            await abandon()
            return thread_reply_result(
                self._page.url,
                thread_id,
                "dry_run",
                "Typed and verified in the thread composer, then removed; "
                "set confirm_send=True to send.",
                composer_text=composer_text,
                preview_path=preview_path,
                thread_event_urn=event_urn,
                attachments=attached,
            )

        if not await self._wait_for_submit_ready(button):
            await abandon()
            return thread_reply_result(
                self._page.url,
                thread_id,
                "submit_unavailable",
                "The submit button never became enabled; nothing was sent.",
                composer_text=composer_text,
                preview_path=preview_path,
                attachments=attached,
            )

        await button.click()
        outcome = await self._observe_submission(
            composer, button, message, attached, form
        )
        await self._session.check_rate_limit()
        prompt = outcome["contact_prompt"]
        diagnostics = {key: value for key, value in outcome.items() if key != "sent"}
        if outcome["sent"]:
            note = (
                " LinkedIn asked to share contact details first; declined."
                if prompt and prompt["declined"]
                else ""
            )
            return thread_reply_result(
                self._page.url,
                thread_id,
                "sent",
                "Reply submitted and observed in the thread." + note,
                composer_text=composer_text,
                preview_path=preview_path,
                sent=True,
                retry_safe=False,
                diagnostics=diagnostics,
                thread_event_urn=await self._newest_event_urn(),
                attachments=attached,
            )
        if prompt and not prompt["declined"]:
            reason = (
                "LinkedIn opened a contact-sharing dialog after submit and no "
                f"decline button was found among {prompt['buttons']}; the reply "
                "is held behind that dialog. Read the thread before retrying."
            )
        elif prompt:
            reason = (
                "LinkedIn opened a contact-sharing dialog after submit; it was "
                "declined but the reply was still not observed in the thread "
                "within the timeout. Read the thread before retrying."
            )
        elif composer_matches(message, outcome.get("composer_text_after", "")):
            await abandon()
            return thread_reply_result(
                self._page.url,
                thread_id,
                "submit_ignored",
                "Submit was clicked but the composer still holds the whole "
                "text and nothing appeared in the thread; the composer was "
                "cleared. Nothing was sent.",
                composer_text=composer_text,
                preview_path=preview_path,
                diagnostics=diagnostics,
                attachments=attached,
            )
        else:
            reason = (
                "Submit was clicked but the reply was not observed in the thread "
                "within the timeout. Read the thread before retrying."
            )
        return thread_reply_result(
            self._page.url,
            thread_id,
            "unconfirmed",
            reason,
            composer_text=composer_text,
            preview_path=preview_path,
            sent=False,
            retry_safe=False,
            diagnostics=diagnostics,
            attachments=attached,
        )
