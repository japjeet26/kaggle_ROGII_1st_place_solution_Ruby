#!/usr/bin/env python3
"""Render docs/feynman_lecture_winning_recipe.md to a print PDF.

Requires the ``markdown`` and ``weasyprint`` packages. From the repository
root::

    python3 docs/render_chapter_pdf.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent / "feynman_lecture_winning_recipe.md"
OUTPUT = Path(__file__).resolve().parent / "feynman_lecture_winning_recipe.pdf"

FENCE_RE = re.compile(
    r"^```(\d+):(\d+):([^\n]+)\n(.*?)^```",
    re.MULTILINE | re.DOTALL,
)
DISPLAY_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)")

LATEX_TO_UNICODE = [
    (r"\mathrm{TVT}", "TVT"),
    (r"\mathrm{GR}", "GR"),
    (r"\mathrm{dTVT}", "dTVT"),
    (r"\mathrm{d}S", "dS"),
    (r"\mathrm{d}x", "dx"),
    (r"\mathrm{d}y", "dy"),
    (r"\mathrm{d}Z", "dZ"),
    (r"\text{well}", "well"),
    (r"\text{typewell}", "typewell"),
    (r"\text{visible prefix}", "visible prefix"),
    (r"\partial", "∂"),
    (r"\nabla", "∇"),
    (r"\sigma", "σ"),
    (r"\times", "×"),
    (r"\cdot", "·"),
    (r"\;", " "),
    (r"\,", " "),
    (r"^{ -4 }", "⁻⁴"),
    (r"^{-4}", "⁻⁴"),
    (r"10^{-4}", "10⁻⁴"),
]


def latex_to_unicode(expr: str) -> str:
    text = " ".join(line.strip() for line in expr.strip().splitlines())
    for src, dst in LATEX_TO_UNICODE:
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rewrite_citation_fences(md: str) -> str:
    def replace(match: re.Match[str]) -> str:
        start, end, path, body = match.group(1), match.group(2), match.group(3), match.group(4)
        caption = f"{path}  ·  lines {start}–{end}"
        lang = "python"
        if path.endswith(".csv"):
            lang = "csv"
        elif path.endswith(".log"):
            lang = "text"
        return f"<p class='code-caption'>{html.escape(caption)}</p>\n\n```{lang}\n{body.rstrip()}\n```\n"

    return FENCE_RE.sub(replace, md)


def rewrite_math(md: str) -> str:
    md = DISPLAY_MATH_RE.sub(
        lambda m: f"\n\n<p class='equation'>{html.escape(latex_to_unicode(m.group(1)))}</p>\n\n",
        md,
    )
    md = INLINE_MATH_RE.sub(
        lambda m: f"<span class='math'>{html.escape(latex_to_unicode(m.group(1)))}</span>",
        md,
    )
    return md


CSS = """
@page {
  size: letter;
  margin: 0.9in 0.95in 1.05in 0.95in;
  @bottom-center {
    content: counter(page);
    font-family: "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
    font-size: 10pt;
    color: #5a5348;
  }
  @bottom-left {
    content: "ROGII 1st-place lecture";
    font-family: "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
    font-size: 8.5pt;
    font-style: italic;
    color: #8a8175;
  }
}
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
html, body {
  background: #f7f1e4;
  color: #1c1610;
}
body {
  font-family: "Palatino Linotype", Palatino, "Book Antiqua", Georgia, "Times New Roman", serif;
  font-size: 11.2pt;
  line-height: 1.48;
  max-width: 6.6in;
  margin: 0 auto;
}
h1 {
  font-size: 22pt;
  font-weight: 700;
  line-height: 1.18;
  margin: 0 0 0.55em;
  letter-spacing: -0.01em;
}
h1 + p {
  font-size: 12pt;
  font-style: italic;
  color: #4a433a;
  margin-top: 0;
}
h2 {
  font-size: 13.5pt;
  font-weight: 700;
  margin: 1.55em 0 0.55em;
  page-break-after: avoid;
  border-bottom: 0.4pt solid #c9b99a;
  padding-bottom: 0.18em;
}
h3 {
  font-size: 11.6pt;
  font-weight: 700;
  font-style: italic;
  margin: 1.15em 0 0.4em;
  page-break-after: avoid;
}
p { margin: 0 0 0.72em; }
a { color: #6b2d12; text-decoration: none; }
strong { font-weight: 700; }
em { font-style: italic; }
code {
  font-family: "Iosevka", "JetBrains Mono", "Source Code Pro", "Consolas", "Courier New", monospace;
  font-size: 0.86em;
  background: #efe6d4;
  padding: 0.05em 0.28em;
  border-radius: 2px;
}
pre {
  font-family: "Iosevka", "JetBrains Mono", "Source Code Pro", "Consolas", "Courier New", monospace;
  font-size: 8pt;
  line-height: 1.35;
  background: #efe8d8;
  border: 0.4pt solid #d4c4a4;
  border-left: 2.5pt solid #8a4a28;
  padding: 0.55em 0.7em;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
  page-break-inside: avoid;
  margin: 0 0 1em;
}
pre code {
  background: transparent;
  padding: 0;
  font-size: inherit;
}
.code-caption {
  font-family: "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 8.5pt;
  font-style: italic;
  color: #6b5e4e;
  margin: 0.9em 0 0.15em;
  page-break-after: avoid;
}
.equation {
  text-align: center;
  font-style: italic;
  margin: 0.85em 0 1em;
  page-break-inside: avoid;
}
.math { font-style: italic; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.7em 0 1.05em;
  font-size: 9.6pt;
  page-break-inside: avoid;
}
th, td {
  border-bottom: 0.4pt solid #cbbca3;
  padding: 0.28em 0.4em;
  text-align: left;
  vertical-align: top;
}
th {
  border-bottom: 1pt solid #8a4a28;
  font-variant: small-caps;
  letter-spacing: 0.03em;
}
tr:nth-child(even) td { background: #f1eadc; }
hr {
  border: 0;
  border-top: 0.4pt solid #c9b99a;
  margin: 1.4em 0;
}
ul, ol { margin: 0 0 0.85em; padding-left: 1.25em; }
li { margin: 0.12em 0; }
blockquote {
  margin: 0.8em 0 1em;
  padding: 0.15em 0 0.15em 0.85em;
  border-left: 2pt solid #8a4a28;
  color: #3a332a;
  font-style: italic;
}
"""


def to_html(md_text: str) -> str:
    md_text = rewrite_citation_fences(md_text)
    md_text = rewrite_math(md_text)
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "smarty"],
        output_format="html5",
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>How to Find a Layer in a Cake You Cannot See</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def print_pdf(html_doc: str, pdf_path: Path) -> None:
    HTML(string=html_doc, base_url=str(SOURCE.parent)).write_pdf(pdf_path)


def main() -> None:
    html_doc = to_html(SOURCE.read_text(encoding="utf-8"))
    print_pdf(html_doc, OUTPUT)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
