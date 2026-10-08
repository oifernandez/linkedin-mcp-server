"""
Reply previews: prepare a thread reply as a dry run, schedule it, send it.

A reply that leaves through these tools always matches a saved dry run, word
for word and file for file, and is refused when anybody wrote in the thread
after the dry run. The scheduled send (trigger="auto") fires only for a
preview that was armed and is due; the store's file lock keeps it and a manual
send from both firing.
"""

import asyncio
import hashlib
import json
import logging
import shutil
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.dependencies import get_ready_extractor
from linkedin_mcp_server.outgoing_voice import voice_problems
from linkedin_mcp_server.reply_store import PreviewError, PreviewStore
from linkedin_mcp_server.scraping.identifiers import normalize_thread_id
from linkedin_mcp_server.scraping.thread_reply import (
    normalize_reply_message,
    refuse_an_invalid_reply,
    resolve_attachments,
)

logger = logging.getLogger(__name__)

store = PreviewStore()
_send_lock = asyncio.Lock()

# Results of reply_to_thread that prove nothing left the composer.
NOTHING_SENT = frozenset(
    {
        "invalid_message",
        "invalid_attachment",
        "thread_not_found",
        "thread_changed",
        "composer_unavailable",
        "composer_mismatch",
        "attachment_failed",
        "submit_unavailable",
        "submit_ignored",
    }
)


def reply_digest(thread_id: str, payload: dict[str, Any]) -> str:
    blob = json.dumps(
        {"thread_id": thread_id, "kind": "reply", "payload": payload},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def register_reply_preview_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register the prepare / schedule / send tools for thread replies."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Prepare Reply",
        annotations={"readOnlyHint": False, "openWorldHint": True},
        tags={"messaging", "actions"},
        exclude_args=["extractor"],
    )
    async def prepare_reply(
        thread_id: str,
        message: str,
        ctx: Context,
        attachments: list[str] | None = None,
        counterpart: str | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Dry run of a reply inside a thread, saved as a preview.

        Types the reply and attaches the files in the thread's composer, reads
        both back, takes a screenshot, removes them again and saves the result
        with a preview_id. Nothing is sent. The text must pass the owner's
        writing rules (no colons, long dashes, lists, contact data, unknown
        links, stock phrases or forbidden terms).

        Args:
            thread_id: LinkedIn messaging thread ID
            message: Reply text; multi-line allowed
            ctx: FastMCP context for progress reporting
            attachments: Paths of files to attach, inside the configured
                attachment directories
            counterpart: Name of the person the reply goes to, for listings

        Returns:
            Dict with preview_id, screenshot, thread_id, message, attachments
            and thread_event_urn (the newest message the reply answers).
        """
        refusal = refuse_an_invalid_reply(thread_id, message, attachments)
        if refusal is not None:
            raise ToolError(f"{refusal['status']}: {refusal['message']}")
        problems = voice_problems(normalize_reply_message(message))
        if problems:
            raise ToolError(
                "The reply breaks the owner's writing rules and was not prepared: "
                + "; ".join(problems)
            )
        files = [str(path) for path in resolve_attachments(attachments)]
        extractor = extractor or await get_ready_extractor(
            ctx, tool_name="prepare_reply"
        )
        result = await extractor.reply_to_thread(
            thread_id,
            message,
            confirm_send=False,
            attachments=files,
        )
        if result.get("status") != "dry_run":
            raise ToolError(
                f"Not prepared: {result.get('status')}: {result.get('message')}"
            )
        event_urn = result.get("thread_event_urn")
        if not event_urn:
            raise ToolError(
                "Not prepared: the thread's messages could not be identified, so a "
                "later reply from the counterpart could not be detected."
            )
        normalized_id = normalize_thread_id(thread_id)
        payload = {
            "message": normalize_reply_message(message),
            "attachments": files,
        }
        preview_id = store.save(
            {
                "kind": "reply",
                "thread_id": normalized_id,
                "title": counterpart or normalized_id,
                "payload": payload,
                "digest": reply_digest(normalized_id, payload),
                "thread_event_urn": event_urn,
            }
        )
        shot = store.screenshot_path(preview_id)
        if result.get("preview_path"):
            shutil.copyfile(result["preview_path"], shot)
        return {
            "preview_id": preview_id,
            "screenshot": str(shot) if shot.exists() else None,
            "thread_id": normalized_id,
            "message": payload["message"],
            "attachments": result.get("attachments") or [],
            "thread_event_urn": event_urn,
            "sent": False,
        }

    @mcp.tool(
        title="List Reply Previews",
        annotations={"readOnlyHint": True, "openWorldHint": False},
        tags={"messaging"},
    )
    async def list_previews(include_closed: bool = False) -> dict[str, Any]:
        """
        Every saved reply preview with its state: waiting (no timer), armed
        (automatic send scheduled at auto_send_at), sending, sent, cancelled or
        expired. include_closed=False lists only waiting and armed ones.
        """
        items = store.summaries()
        if not include_closed:
            items = [i for i in items if i["state"] in ("waiting", "armed")]
        return {"previews": items}

    @mcp.tool(
        title="Arm Auto Send",
        annotations={"readOnlyHint": False, "openWorldHint": False},
        tags={"messaging"},
    )
    async def arm_auto_send(
        preview_id: str, card_message_ids: list[str], delay_minutes: int = 30
    ) -> dict[str, Any]:
        """
        Schedule the automatic send of a preview whose card was posted for the
        owner. card_message_ids are the chat message ids of the card, so a reply
        to the card can stop it. Calling it again re-arms: the first deadline is
        kept, and never sooner than 5 minutes from now.
        """
        try:
            data = store.arm(preview_id, card_message_ids, delay_minutes * 60)
        except PreviewError as e:
            raise ToolError(str(e)) from e
        return {
            "preview_id": preview_id,
            "state": "armed",
            "auto_send_at": data["auto_send_at"],
            "card_message_ids": data["card_message_ids"],
        }

    @mcp.tool(
        title="Cancel Reply Preview",
        annotations={"readOnlyHint": False, "openWorldHint": False},
        tags={"messaging"},
    )
    async def cancel_preview(preview_id: str, reason: str) -> dict[str, Any]:
        """Cancel a preview so it can never be sent, by hand or automatically."""
        try:
            store.cancel(preview_id, reason)
        except PreviewError as e:
            raise ToolError(str(e)) from e
        return {"preview_id": preview_id, "state": "cancelled"}

    @mcp.tool(
        timeout=tool_timeout,
        title="Submit Reply Preview",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"messaging", "actions"},
        exclude_args=["extractor"],
    )
    async def submit_preview(
        preview_id: str,
        ctx: Context,
        confirm_send: bool = False,
        trigger: str = "owner",
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Send a prepared reply exactly as previewed.

        trigger="owner" is the owner's yes on the card; trigger="auto" is the
        scheduled send and is refused unless the preview is armed and due.
        Refused when anybody wrote in the thread after the preview. Without
        confirm_send it only re-checks the saved preview.

        Returns:
            Dict with status (sent or unconfirmed), sent, retry_safe and the
            reply_to_thread result. A refusal that sent nothing raises an
            error and leaves the preview usable.
        """
        try:
            rec = store.load(preview_id)
        except PreviewError as e:
            raise ToolError(str(e)) from e
        if reply_digest(rec["thread_id"], rec["payload"]) != rec["digest"]:
            raise ToolError(
                f"preview {preview_id} was altered on disk; prepare it again"
            )
        if not confirm_send:
            return {
                "sent": False,
                "dry_run": True,
                "preview_id": preview_id,
                "payload": rec["payload"],
            }
        extractor = extractor or await get_ready_extractor(
            ctx, tool_name="submit_preview"
        )
        async with _send_lock:
            try:
                rec = store.claim(preview_id, trigger)
            except PreviewError as e:
                raise ToolError(str(e)) from e
            try:
                result = await extractor.reply_to_thread(
                    rec["thread_id"],
                    rec["payload"]["message"],
                    confirm_send=True,
                    attachments=rec["payload"]["attachments"],
                    expected_event_urn=rec["thread_event_urn"],
                )
            except Exception as e:
                # The click may or may not have happened; never send it twice.
                store.mark_used(
                    preview_id,
                    {"sent": False, "status": "unconfirmed", "error": repr(e)},
                )
                raise
            status = result.get("status")
            if status in NOTHING_SENT:
                store.release(preview_id, f"{status}: {result.get('message')}")
                raise ToolError(f"Not sent: {status}: {result.get('message')}")
            outcome = {
                **result,
                "preview_id": preview_id,
                "trigger": trigger,
                "title": rec.get("title"),
                "retry_safe": False,
            }
            store.mark_used(preview_id, outcome)
            return outcome
