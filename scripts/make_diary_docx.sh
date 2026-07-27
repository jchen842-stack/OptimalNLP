#!/usr/bin/env bash
# Render a diary markdown file to landscape .docx for Google Docs.
#
# pandoc emits portrait by default and exposes no page-orientation option, so we rewrite the
# section properties in word/document.xml afterwards. Matches the page setup of summer_d5.docx:
# US Letter landscape (15840 x 12240 twips) with 0.5in (720 twip) margins all round.
#
# Usage: scripts/make_diary_docx.sh diary/summer_d5.1.md
set -euo pipefail

src="${1:?usage: make_diary_docx.sh <diary/*.md>}"
out="${src%.md}.docx"

command -v pandoc >/dev/null || { echo "pandoc not found (brew install pandoc)" >&2; exit 1; }

pandoc "$src" -o "$out"

python3 - "$out" <<'PY'
import re, shutil, sys, zipfile

path = sys.argv[1]
LANDSCAPE = (
    '<w:sectPr>'
    '<w:pgSz w:h="12240" w:orient="landscape" w:w="15840"/>'
    '<w:pgMar w:bottom="720" w:footer="720" w:gutter="0" w:header="720"'
    ' w:left="720" w:right="720" w:top="720"/>'
    '</w:sectPr>'
)

with zipfile.ZipFile(path) as z:
    items = [(i, z.read(i.filename)) for i in z.infolist()]

doc = next(d for i, d in items if i.filename == "word/document.xml").decode("utf-8")
if re.search(r"<w:sectPr\b.*?</w:sectPr>", doc, re.S):
    doc = re.sub(r"<w:sectPr\b.*?</w:sectPr>", LANDSCAPE, doc, count=1, flags=re.S)
else:  # no section properties emitted; insert before the body close
    doc = doc.replace("</w:body>", LANDSCAPE + "</w:body>")

tmp = path + ".tmp"
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
    for info, data in items:
        z.writestr(info, doc.encode("utf-8")
                   if info.filename == "word/document.xml" else data)
shutil.move(tmp, path)
print(f"landscape: {path}")
PY
