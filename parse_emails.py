#!/usr/bin/env python3
"""
Parse .eml files from the emails/ folder and convert them to JSON
following the structure defined in the params file:
  message.body
  message.subject
  message.sender.display_name
  message.sender.email
  message.links[i].url
  message.links[i].text
"""

import email
import email.policy
import json
import os
import re
from email.header import decode_header, make_header
from html.parser import HTMLParser
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_mime_words(raw: str) -> str:
    """Decode RFC-2047 encoded header value to a plain string."""
    if not raw:
        return ""
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def parse_sender(from_header: str):
    """Return (display_name, email_address) from a From header value."""
    from email.utils import parseaddr
    display_name, addr = parseaddr(from_header)
    display_name = decode_mime_words(display_name)
    return display_name, addr


class LinkExtractor(HTMLParser):
    """Extract <a href=...> links from an HTML fragment."""

    def __init__(self):
        super().__init__()
        self.links: list[dict] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            # Skip mailto: and empty hrefs
            if href and not href.startswith("mailto:"):
                self._current_href = href
                self._current_text = []

    def handle_endtag(self, tag):
        if tag == "a" and self._current_href is not None:
            text = "".join(self._current_text).strip()
            self.links.append({"url": self._current_href, "text": text})
            self._current_href = None
            self._current_text = []

    def handle_data(self, data):
        if self._current_href is not None:
            self._current_text.append(data)


def extract_links_from_html(html_content: str) -> list[dict]:
    parser = LinkExtractor()
    parser.feed(html_content)
    return parser.links


def extract_links_from_text(text: str) -> list[dict]:
    """Fallback: extract bare URLs from plain text."""
    url_pattern = re.compile(r'https?://[^\s<>"\']+')
    urls = url_pattern.findall(text)
    seen = set()
    links = []
    for url in urls:
        # Strip trailing punctuation
        url = url.rstrip(".,;:!?)")
        if url not in seen:
            seen.add(url)
            links.append({"url": url, "text": ""})
    return links


def get_body_and_links(msg: email.message.Message):
    """
    Walk the MIME tree and return (plain_text_body, links).
    Prefer text/html for link extraction; use text/plain as body fallback.
    """
    plain_body = ""
    html_body = ""

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                decoded = payload.decode(charset, errors="replace")
            except Exception:
                continue

            if ct == "text/plain" and not plain_body:
                plain_body = decoded
            elif ct == "text/html" and not html_body:
                html_body = decoded
    else:
        ct = msg.get_content_type()
        charset = msg.get_content_charset() or "utf-8"
        try:
            payload = msg.get_payload(decode=True)
            decoded = payload.decode(charset, errors="replace") if payload else ""
        except Exception:
            decoded = ""
        if ct == "text/html":
            html_body = decoded
        else:
            plain_body = decoded

    # Use plain text as the body; strip excessive whitespace
    body = plain_body.strip() if plain_body else re.sub(r'<[^>]+>', '', html_body).strip()

    # Extract links: prefer HTML (richer anchor text), fallback to plain text
    if html_body:
        links = extract_links_from_html(html_body)
        # Deduplicate while preserving order
        seen_urls = set()
        unique_links = []
        for lnk in links:
            if lnk["url"] not in seen_urls:
                seen_urls.add(lnk["url"])
                unique_links.append(lnk)
        links = unique_links
    else:
        links = extract_links_from_text(plain_body)

    return body, links


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_eml(eml_path: Path) -> dict:
    raw = eml_path.read_bytes()
    msg = email.message_from_bytes(raw, policy=email.policy.compat32)

    subject = decode_mime_words(msg.get("Subject", ""))
    from_raw = decode_mime_words(msg.get("From", ""))
    display_name, sender_email = parse_sender(from_raw)
    body, links = get_body_and_links(msg)
    date = msg.get("Date", "")

    return {
        "message": {
            "date": date,
            "subject": subject,
            "sender": {
                "display_name": display_name,
                "email": sender_email,
            },
            "body": body,
            "links": links,
        }
    }


def main():
    workspace = Path(__file__).parent
    emails_dir = workspace / "emails-eml"
    output_dir = workspace / "emails-json"
    output_dir.mkdir(exist_ok=True)

    eml_files = sorted(emails_dir.glob("*.eml"))
    if not eml_files:
        print("No .eml files found in emails-eml/")
        return

    for eml_path in eml_files:
        try:
            data = parse_eml(eml_path)
            out_name = eml_path.stem + ".json"
            out_path = output_dir / out_name
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✓ {eml_path.name}")
            # delete file once it has been processed
            os.remove(eml_path)
        except Exception as exc:
            print(f"✗ {eml_path.name}: {exc}")

    print(f"\nDone — {len(eml_files)} file(s) written to {output_dir}/")


if __name__ == "__main__":
    main()
