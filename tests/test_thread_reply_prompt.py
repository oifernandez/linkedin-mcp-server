"""reply_to_thread after the click: the contact-sharing dialog and the diagnostics."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from linkedin_mcp_server.scraping import thread_reply as module
from linkedin_mcp_server.scraping.thread_reply import (
    ThreadReplier,
    is_contact_prompt,
    pick_decline_label,
)

MESSAGE = "Hi Pedro, sorry for the slow reply.\n\nBest regards,\nÓscar"
PROMPT_TEXT = (
    "¿Quieres compartir tu información de contacto? Pedro Negredo quiere "
    "contactar contigo para obtener más información o programar una entrevista."
)
DECLINE = "No, no quiero compartirlo"
SHARE = "Sí, quiero compartirla"


def test_is_contact_prompt_matches_both_languages() -> None:
    assert is_contact_prompt(PROMPT_TEXT)
    assert is_contact_prompt("Do you want to share your contact info?")
    assert not is_contact_prompt("Escribe un mensaje… Enviar")


def test_pick_decline_label_skips_share_and_unrelated_buttons() -> None:
    assert pick_decline_label([SHARE, DECLINE]) == DECLINE
    assert pick_decline_label(["Notificaciones", "Yes, share"]) is None
    assert pick_decline_label(["No, I don't want to share"]) == (
        "No, I don't want to share"
    )


class FakeDialog:
    """The contact-sharing dialog as the replier sees it through locators."""

    def __init__(self, page: FakePage) -> None:
        self.page = page

    def filter(self, has_text: Any) -> FakeDialog:
        return self

    @property
    def first(self) -> FakeDialog:
        return self

    def locator(self, selector: str) -> FakeDialog:
        return self

    def get_by_role(self, role: str, name: str, exact: bool) -> FakeDialog:
        self.page.clicked.append(name)
        return self

    async def is_visible(self) -> bool:
        return self.page.prompt_open

    async def inner_text(self) -> str:
        return PROMPT_TEXT

    async def all_inner_texts(self) -> list[str]:
        return [DECLINE, SHARE]

    async def click(self) -> None:
        self.page.answer_prompt()


class FakeComposer:
    def __init__(self, page: FakePage) -> None:
        self.page = page

    async def inner_text(self) -> str:
        return self.page.composer_text


class FakeButton:
    async def is_enabled(self) -> bool:
        return True


class FakePage:
    url = "https://www.linkedin.com/messaging/thread/2-YWJjXzEwMA==/"

    def __init__(self, *, prompt_after_click: bool) -> None:
        self.composer_text = "" if prompt_after_click else MESSAGE
        self.thread_text = "Pedro Negredo · 19:13 Hello Oscar,"
        self.prompt_open = prompt_after_click
        self.clicked: list[str] = []

    def locator(self, selector: str) -> FakeDialog:
        return FakeDialog(self)

    async def evaluate(self, script: str) -> str:
        return self.thread_text

    def answer_prompt(self) -> None:
        self.prompt_open = False
        self.thread_text += " " + MESSAGE.replace("\n", " ")


class FakeSession:
    def __init__(self, page: FakePage) -> None:
        self.page = page


def observe(page: FakePage) -> dict[str, Any]:
    replier = ThreadReplier(cast(Any, FakeSession(page)), cast(Any, None))
    return asyncio.run(
        replier._observe_submission(FakeComposer(page), FakeButton(), MESSAGE)
    )


def test_contact_prompt_is_declined_and_the_reply_is_then_observed() -> None:
    page = FakePage(prompt_after_click=True)
    outcome = observe(page)
    assert outcome["sent"] is True
    assert page.clicked == [DECLINE]
    assert outcome["contact_prompt"]["declined"] is True
    assert outcome["contact_prompt"]["buttons"] == [DECLINE, SHARE]


def test_timeout_reports_composer_and_button_state(monkeypatch: Any) -> None:
    monkeypatch.setattr(module, "_CONFIRMATION_TIMEOUT_MS", 100)
    page = FakePage(prompt_after_click=False)
    outcome = observe(page)
    assert outcome["sent"] is False
    assert outcome["contact_prompt"] is None
    assert outcome["composer_text_after"] == MESSAGE
    assert outcome["submit_enabled"] is True
    assert outcome["contact_prompt_in_page"] is False
    assert page.clicked == []
