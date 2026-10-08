"""Reply previews: attachments, the thread marker, scheduling and the send gate."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Coroutine, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import FunctionTool

from linkedin_mcp_server.outgoing_voice import voice_problems
from linkedin_mcp_server.reply_store import (
    MIN_AUTO_DELAY_SECONDS,
    PreviewError,
    PreviewStore,
)
from linkedin_mcp_server.scraping import thread_reply
from linkedin_mcp_server.scraping.thread_reply import (
    pick_file_input,
    refuse_an_invalid_reply,
    resolve_attachments,
)
from linkedin_mcp_server.tools import reply_previews

BODY = "Hola Paz,\n\nTe paso el CV por aquí.\n\nUn saludo,\nÓscar"
URN = "urn:li:msg_message:(urn:li:fsd_profile:A,2-xyz)"


@pytest.fixture
def attachment_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "attachments"
    folder.mkdir()
    (folder / "cv.pdf").write_bytes(b"%PDF-1.4 test")
    monkeypatch.setenv(thread_reply.ATTACHMENT_DIRS_ENV, str(folder))
    return folder


class TestAttachments:
    def test_files_inside_the_directory_resolve(self, attachment_dir: Path) -> None:
        files = resolve_attachments([str(attachment_dir / "cv.pdf")])
        assert [f.name for f in files] == ["cv.pdf"]

    def test_no_attachments(self) -> None:
        assert resolve_attachments(None) == []

    def test_outside_the_directory_refused(
        self, attachment_dir: Path, tmp_path: Path
    ) -> None:
        stray = tmp_path / "secret.pdf"
        stray.write_bytes(b"x")
        with pytest.raises(ValueError, match="outside"):
            resolve_attachments([str(stray)])

    @pytest.mark.skipif(
        sys.platform == "win32", reason="creating symlinks needs privileges"
    )
    def test_symlink_out_of_the_directory_refused(
        self, attachment_dir: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "private.pdf"
        target.write_bytes(b"x")
        (attachment_dir / "link.pdf").symlink_to(target)
        with pytest.raises(ValueError, match="outside"):
            resolve_attachments([str(attachment_dir / "link.pdf")])

    def test_traversal_refused(self, attachment_dir: Path) -> None:
        with pytest.raises(ValueError, match="outside"):
            resolve_attachments([str(attachment_dir / ".." / "x.pdf")])

    @pytest.mark.parametrize(
        ("name", "size", "match"),
        [
            ("run.sh", 1, "type"),
            ("big.pdf", 20 * 1024 * 1024 + 1, "20 MB"),
        ],
        ids=["script", "oversized"],
    )
    def test_type_and_size(
        self, attachment_dir: Path, name: str, size: int, match: str
    ) -> None:
        with (attachment_dir / name).open("wb") as fh:
            fh.truncate(size)
        with pytest.raises(ValueError, match=match):
            resolve_attachments([str(attachment_dir / name)])

    def test_missing_file(self, attachment_dir: Path) -> None:
        with pytest.raises(ValueError, match="not a file"):
            resolve_attachments([str(attachment_dir / "nope.pdf")])

    def test_duplicates_and_count(self, attachment_dir: Path) -> None:
        cv = str(attachment_dir / "cv.pdf")
        with pytest.raises(ValueError, match="twice"):
            resolve_attachments([cv, cv])
        with pytest.raises(ValueError, match="At most"):
            resolve_attachments([cv] * 6)

    def test_refusal_reports_the_attachment(self, attachment_dir: Path) -> None:
        refusal = refuse_an_invalid_reply("2-abc", BODY, ["/etc/passwd"])
        assert refusal is not None
        assert refusal["status"] == "invalid_attachment"
        assert refusal["sent"] is False

    def test_document_input_is_picked_for_a_pdf(self) -> None:
        accepts = ["image/*", "image/*,.ai,.psd,.pdf,.doc,.docx"]
        assert pick_file_input(accepts, ".pdf") == 1
        assert pick_file_input(accepts, ".PDF") == 1
        assert pick_file_input(accepts, ".png") == 0
        assert pick_file_input(accepts, ".xls") == 1


class TestVoice:
    def check(self, text: str) -> list[str]:
        return voice_problems(
            text, forbidden_terms=["Portugal"], allowed_url_hosts=["koalendar.com"]
        )

    def test_plain_reply_passes(self) -> None:
        assert self.check(BODY) == []

    @pytest.mark.parametrize(
        ("text", "problem"),
        [
            ("Hola Paz: te cuento", "colon"),
            ("Esto — aquello", "long dash"),
            ("Hola\n- uno", "list"),
            ("mi correo es a@b.com", "email"),
            ("Trabajo desde Portugal", "Portugal"),
            ("No dudes en contactarme", "stock phrase"),
            ("Mira https://example.com", "allowed destination"),
        ],
    )
    def test_breaches(self, text: str, problem: str) -> None:
        assert any(problem in p for p in self.check(text))


class TestStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> PreviewStore:
        return PreviewStore(root=tmp_path / "store", ttl=24 * 3600)

    def test_schedule_and_due(self, store: PreviewStore) -> None:
        pid = store.save({"kind": "reply"}, now=1000)
        store.arm(pid, ["4600"], 1800, now=1000)
        with pytest.raises(PreviewError, match="not due"):
            store.claim(pid, "auto", now=2799)
        assert store.claim(pid, "auto", now=2800)["trigger"] == "auto"

    def test_rearm_keeps_deadline_with_floor(self, store: PreviewStore) -> None:
        pid = store.save({"kind": "reply"}, now=1000)
        store.arm(pid, ["1"], 1800, now=1000)
        assert store.arm(pid, ["1"], 1800, now=1500)["auto_send_at"] == 2800
        late = store.arm(pid, ["1"], 1800, now=3000)["auto_send_at"]
        assert late == 3000 + MIN_AUTO_DELAY_SECONDS

    def test_cancelled_never_sends(self, store: PreviewStore) -> None:
        pid = store.save({"kind": "reply"}, now=1000)
        store.cancel(pid, "no", now=1001)
        with pytest.raises(PreviewError, match="cancelled"):
            store.claim(pid, "owner", now=1002)

    def test_one_claim_wins(self, store: PreviewStore) -> None:
        pid = store.save({"kind": "reply"})
        wins: list[int] = []

        def go() -> None:
            try:
                store.claim(pid, "owner")
                wins.append(1)
            except PreviewError:
                pass

        threads = [threading.Thread(target=go) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert wins == [1]


async def tool(mcp: FastMCP, name: str) -> Callable[..., Coroutine[Any, Any, Any]]:
    found = await mcp.get_tool(name)
    assert found is not None
    return cast(FunctionTool, found).fn


@pytest.fixture
def tools_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attachment_dir: Path
) -> tuple[FastMCP, PreviewStore, MagicMock]:
    store = PreviewStore(root=tmp_path / "store", shots=tmp_path / "shots")
    monkeypatch.setattr(reply_previews, "store", store)
    monkeypatch.setenv("OUTGOING_FORBIDDEN_TERMS", "")
    monkeypatch.setenv("OUTGOING_ALLOWED_URL_HOSTS", "")
    shot = tmp_path / "composer.png"
    shot.write_bytes(b"png")
    extractor = MagicMock()
    extractor.reply_to_thread = AsyncMock(
        return_value={
            "status": "dry_run",
            "thread_event_urn": URN,
            "preview_path": str(shot),
            "attachments": ["cv.pdf"],
        }
    )
    mcp = FastMCP("test")
    reply_previews.register_reply_preview_tools(mcp)
    return mcp, store, extractor


async def prepared(env: Any, mock_context: Any) -> str:
    mcp, store, extractor = env
    prepare = await tool(mcp, "prepare_reply")
    out = await prepare(
        "2-xyz",
        BODY,
        mock_context,
        attachments=[str(Path(store.root).parent / "attachments" / "cv.pdf")],
        counterpart="Paz Grau",
        extractor=extractor,
    )
    return out["preview_id"]


class TestTools:
    async def test_prepare_saves_preview_and_screenshot(
        self, tools_env: Any, mock_context: Any
    ) -> None:
        mcp, store, extractor = tools_env
        pid = await prepared(tools_env, mock_context)
        rec = json.loads((store.root / f"{pid}.json").read_text())
        assert rec["thread_event_urn"] == URN
        assert rec["payload"]["attachments"][0].endswith("cv.pdf")
        assert store.screenshot_path(pid).read_bytes() == b"png"
        call = extractor.reply_to_thread.await_args
        assert call.kwargs["confirm_send"] is False

    async def test_prepare_refuses_rule_breaks_before_the_browser(
        self, tools_env: Any, mock_context: Any
    ) -> None:
        mcp, store, extractor = tools_env
        prepare = await tool(mcp, "prepare_reply")
        with pytest.raises(ToolError, match="colon"):
            await prepare("2-xyz", "Hola Paz: hola", mock_context, extractor=extractor)
        extractor.reply_to_thread.assert_not_awaited()

    async def test_prepare_needs_the_thread_marker(
        self, tools_env: Any, mock_context: Any
    ) -> None:
        mcp, store, extractor = tools_env
        extractor.reply_to_thread.return_value = {"status": "dry_run"}
        prepare = await tool(mcp, "prepare_reply")
        with pytest.raises(ToolError, match="could not be identified"):
            await prepare("2-xyz", BODY, mock_context, extractor=extractor)

    async def test_owner_send_passes_marker_and_files(
        self, tools_env: Any, mock_context: Any
    ) -> None:
        mcp, store, extractor = tools_env
        pid = await prepared(tools_env, mock_context)
        extractor.reply_to_thread.return_value = {"status": "sent", "sent": True}
        submit = await tool(mcp, "submit_preview")
        out = await submit(pid, mock_context, confirm_send=True, extractor=extractor)
        assert out["status"] == "sent"
        call = extractor.reply_to_thread.await_args
        assert call.kwargs["confirm_send"] is True
        assert call.kwargs["expected_event_urn"] == URN
        assert call.kwargs["attachments"][0].endswith("cv.pdf")
        with pytest.raises(ToolError, match="already sent"):
            await submit(pid, mock_context, confirm_send=True, extractor=extractor)

    async def test_thread_changed_keeps_preview_usable(
        self, tools_env: Any, mock_context: Any
    ) -> None:
        mcp, store, extractor = tools_env
        pid = await prepared(tools_env, mock_context)
        extractor.reply_to_thread.return_value = {
            "status": "thread_changed",
            "message": "Someone wrote",
        }
        submit = await tool(mcp, "submit_preview")
        with pytest.raises(ToolError, match="thread_changed"):
            await submit(pid, mock_context, confirm_send=True, extractor=extractor)
        assert store.load(pid)["last_error"].startswith("thread_changed")

    async def test_unconfirmed_is_never_retried(
        self, tools_env: Any, mock_context: Any
    ) -> None:
        mcp, store, extractor = tools_env
        pid = await prepared(tools_env, mock_context)
        extractor.reply_to_thread.return_value = {"status": "unconfirmed"}
        submit = await tool(mcp, "submit_preview")
        out = await submit(pid, mock_context, confirm_send=True, extractor=extractor)
        assert out["retry_safe"] is False
        with pytest.raises(ToolError, match="already sent"):
            await submit(pid, mock_context, confirm_send=True, extractor=extractor)

    async def test_crash_mid_send_burns_the_preview(
        self, tools_env: Any, mock_context: Any
    ) -> None:
        mcp, store, extractor = tools_env
        pid = await prepared(tools_env, mock_context)
        extractor.reply_to_thread.side_effect = RuntimeError("browser died")
        submit = await tool(mcp, "submit_preview")
        with pytest.raises(RuntimeError):
            await submit(pid, mock_context, confirm_send=True, extractor=extractor)
        assert {s["state"] for s in store.summaries()} == {"sent"}

    async def test_auto_send_only_when_armed_and_due(
        self, tools_env: Any, mock_context: Any
    ) -> None:
        mcp, store, extractor = tools_env
        pid = await prepared(tools_env, mock_context)
        submit = await tool(mcp, "submit_preview")
        with pytest.raises(ToolError, match="no automatic send"):
            await submit(
                pid,
                mock_context,
                confirm_send=True,
                trigger="auto",
                extractor=extractor,
            )
        arm = await tool(mcp, "arm_auto_send")
        out = await arm(pid, ["4601"], 30)
        assert out["state"] == "armed"
        with pytest.raises(ToolError, match="not due"):
            await submit(
                pid,
                mock_context,
                confirm_send=True,
                trigger="auto",
                extractor=extractor,
            )

    async def test_tampered_preview_refused(
        self, tools_env: Any, mock_context: Any
    ) -> None:
        mcp, store, extractor = tools_env
        pid = await prepared(tools_env, mock_context)
        path = store.root / f"{pid}.json"
        path.write_text(path.read_text().replace("Te paso", "Te mando"))
        submit = await tool(mcp, "submit_preview")
        with pytest.raises(ToolError, match="altered"):
            await submit(pid, mock_context, confirm_send=True, extractor=extractor)

    async def test_cancel_and_list(self, tools_env: Any, mock_context: Any) -> None:
        mcp, store, extractor = tools_env
        pid = await prepared(tools_env, mock_context)
        cancel = await tool(mcp, "cancel_preview")
        await cancel(pid, "Óscar dijo no")
        listing = await tool(mcp, "list_previews")
        assert (await listing())["previews"] == []
        closed = (await listing(include_closed=True))["previews"]
        assert closed[0]["state"] == "cancelled"


class _Names:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    async def all_inner_texts(self) -> list[str]:
        return list(self.names)


class _Form:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def locator(self, selector: str) -> _Names:
        return _Names(self.names)


class _Composer:
    async def inner_text(self) -> str:
        return ""


class _Button:
    async def is_enabled(self) -> bool:
        return True


class _Page:
    url = "https://www.linkedin.com/messaging/thread/2-xyz/"

    def __init__(self, main: str) -> None:
        self.main = main

    async def evaluate(self, script: str) -> str:
        return self.main

    def locator(self, selector: str) -> Any:
        dialog = MagicMock()
        dialog.filter.return_value = dialog
        dialog.first = dialog
        dialog.is_visible = AsyncMock(return_value=False)
        return dialog


class _Session:
    def __init__(self, page: _Page) -> None:
        self.page = page


def _observe(main: str, still_attached: list[str]) -> dict[str, Any]:
    import asyncio

    replier = thread_reply.ThreadReplier(
        cast(Any, _Session(_Page(main))), cast(Any, None)
    )
    return asyncio.run(
        replier._observe_submission(
            _Composer(), _Button(), BODY, ["cv.pdf"], _Form(still_attached)
        )
    )


def test_reply_with_file_counts_once_the_file_is_in_the_thread() -> None:
    main = "Paz Grau Oscar " + BODY.replace("\n", " ") + " cv.pdf 35 KB"
    assert _observe(main, [])["sent"] is True


def test_reply_with_file_still_in_the_composer_is_not_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(thread_reply, "_CONFIRMATION_TIMEOUT_MS", 100)
    main = "Paz Grau Oscar " + BODY.replace("\n", " ") + " cv.pdf 35 KB"
    assert _observe(main, ["cv.pdf"])["sent"] is False


def test_reply_whose_file_never_shows_is_not_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(thread_reply, "_CONFIRMATION_TIMEOUT_MS", 100)
    main = "Paz Grau Oscar " + BODY.replace("\n", " ")
    assert _observe(main, [])["sent"] is False
