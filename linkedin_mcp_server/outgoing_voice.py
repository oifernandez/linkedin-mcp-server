"""Mechanical checks on any reply that leaves in the account owner's name.

The agent writes the drafts; this module refuses the ones that break the
owner's standing rules in a way a regex can see, so a slip never reaches a client even
when nobody reads the card. Style is judged by people; these are the hard
lines. Account-specific terms (product names, places) come from the
environment so the code carries no personal data.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from urllib.parse import urlparse

FORBIDDEN_TERMS_ENV = "OUTGOING_FORBIDDEN_TERMS"
ALLOWED_URL_HOSTS_ENV = "OUTGOING_ALLOWED_URL_HOSTS"

_URL_RE = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?<![\w€$])\+?\d(?:[ .-]?\d){8,}(?!\w)")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•#>]|\d{1,2}[.)])\s")
_MARKDOWN_RE = re.compile(r"\*\*|__|`")

# Filler that reads as a template, in Spanish and English.
STOCK_PHRASES = (
    "quedo a tu disposición",
    "quedo a vuestra disposición",
    "no dudes en contactarme",
    "no dudéis en contactarme",
    "sería un placer",
    "espero que estés bien",
    "espero que estéis bien",
    "i hope this finds you well",
    "i hope this message finds you well",
    "i'd be thrilled",
    "please don't hesitate",
    "don't hesitate to reach out",
)


def _env_list(name: str) -> list[str]:
    return [t.strip() for t in os.environ.get(name, "").split(",") if t.strip()]


def _host_allowed(host: str, allowed: Iterable[str]) -> bool:
    host = host.lower().split(":")[0]
    return any(host == a or host.endswith("." + a) for a in allowed)


def voice_problems(
    text: str,
    *,
    forbidden_terms: Iterable[str] | None = None,
    allowed_url_hosts: Iterable[str] | None = None,
) -> list[str]:
    """Every hard-rule breach in ``text``; an empty list means it may leave."""
    terms = (
        list(forbidden_terms)
        if forbidden_terms is not None
        else _env_list(FORBIDDEN_TERMS_ENV)
    )
    hosts = [
        h.lower()
        for h in (
            allowed_url_hosts
            if allowed_url_hosts is not None
            else _env_list(ALLOWED_URL_HOSTS_ENV)
        )
    ]
    problems: list[str] = []

    for url in _URL_RE.findall(text):
        host = urlparse(url).netloc
        if not _host_allowed(host, hosts):
            problems.append(f"link to {host or url} is not an allowed destination")
    bare = _URL_RE.sub(" ", text)
    if _EMAIL_RE.search(bare):
        problems.append("contains an email address")
    without_mail = _EMAIL_RE.sub(" ", bare)
    if _PHONE_RE.search(without_mail):
        problems.append("contains a phone number")
    if ":" in _TIME_RE.sub(" ", without_mail):
        problems.append("contains a colon")
    if "—" in text or " – " in text:
        problems.append("contains a long dash")
    if any(_LIST_LINE_RE.match(line) for line in text.splitlines()):
        problems.append("contains a list, heading or quote line")
    if _MARKDOWN_RE.search(text):
        problems.append("contains markdown formatting")
    lowered = text.lower()
    for phrase in STOCK_PHRASES:
        if phrase in lowered:
            problems.append(f"stock phrase {phrase!r}")
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE):
            problems.append(f"names {term!r}, which must not appear")
    return problems
