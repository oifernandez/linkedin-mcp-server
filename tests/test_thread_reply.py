"""Browser-free contracts of reply_to_thread: normalization, refusal, read-back."""

from linkedin_mcp_server.scraping.thread_reply import (
    composer_matches,
    normalize_reply_message,
    refuse_an_invalid_reply,
)

THREAD = "2-YWJjXzEwMA=="


def test_normalize_keeps_paragraphs_and_drops_edges() -> None:
    text = "\r\n Hola Laura,  \r\nGracias.\n\nUn saludo,\nÓscar\n\n"
    assert (
        normalize_reply_message(text) == " Hola Laura,\nGracias.\n\nUn saludo,\nÓscar"
    )


def test_refusal_accepts_line_breaks() -> None:
    assert refuse_an_invalid_reply(THREAD, "Hola\n\nUn saludo,\nÓscar") is None


def test_refusal_rejects_tabs_and_blank_messages() -> None:
    tab = refuse_an_invalid_reply(THREAD, "Hola\tmundo")
    assert tab is not None
    assert tab["status"] == "invalid_message"
    assert tab["sent"] is False
    assert tab["retry_safe"] is True
    blank = refuse_an_invalid_reply(THREAD, " \n \n")
    assert blank is not None
    assert blank["status"] == "invalid_message"


def test_refusal_rejects_oversized_messages() -> None:
    result = refuse_an_invalid_reply(THREAD, "x" * 8001)
    assert result is not None
    assert "8000" in result["message"]


def test_composer_matches_ignores_wrapping_but_not_words_or_lines() -> None:
    expected = "Hola Laura, gracias.\n\nUn saludo,\nÓscar"
    assert composer_matches(expected, "Hola Laura, gracias.\n\n\nUn saludo,\nÓscar")
    assert not composer_matches(expected, "Hola Laura, gracias.\nUn saludo, Óscar")
    assert not composer_matches(expected, "Hola Laura, gracias!\n\nUn saludo,\nÓscar")
