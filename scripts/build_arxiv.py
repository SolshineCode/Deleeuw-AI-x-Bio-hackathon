"""
Build arXiv-ready LaTeX and PDF from paper/submission.md.

Steps:
  1. Pre-process markdown: fix image paths, strip inline section numbers,
     extract metadata
  2. Run pandoc -> .tex (standalone, no auto-numbering)
  3. Post-process .tex: fix author block, deduplicate packages,
     convert Abstract section to environment
  4. Generate PDF via pandoc -> HTML -> xhtml2pdf
"""

import re
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
_stem = sys.argv[1] if len(sys.argv) > 1 else "submission"
SRC = PAPER / f"{_stem}.md"
TEX_OUT = PAPER / f"{_stem}.tex"
PDF_OUT = PAPER / f"{_stem}.pdf"
HTML_TEMP = PAPER / f"_temp_{_stem}.html"
MD_TEMP = PAPER / f"_temp_{_stem}_arxiv.md"

# ─── Step 1: Pre-process markdown ────────────────────────────────────────────

text = SRC.read_text(encoding="utf-8")

# Fix image paths: ../demo/X -> figures/X
text = re.sub(r'\.\./demo/([\w_\-\.]+\.png)', r'figures/\1', text)

# Extract metadata from bold block
author_m  = re.search(r'\*\*Author:\*\*\s*(.+)',      text)
contact_m = re.search(r'\*\*Contact:\*\*\s*(.+)',     text)
date_m    = re.search(r'\*\*Date:\*\*\s*(.+)',        text)
venue_m   = re.search(r'\*\*Venue:\*\*\s*(.+)',       text)
affil_m   = re.search(r'\*\*Affiliation:\*\*\s*(.+)', text)

author_str  = (author_m.group(1)  if author_m  else "Caleb DeLeeuw").strip()
date_str    = (date_m.group(1)    if date_m    else "May 2026").strip()
venue_str   = (venue_m.group(1)   if venue_m   else "").strip()
contact_str = (contact_m.group(1) if contact_m else "").strip()
affil_str   = (affil_m.group(1)   if affil_m   else "Independent researcher").strip()

# Remove the H1 title + bold metadata block (pandoc will render from YAML)
text = re.sub(
    r'^# BioRefusalAudit.*?\n\n\*\*Author:\*\*.*?(?=\n---)',
    '',
    text,
    count=1,
    flags=re.DOTALL,
)
# Strip the orphaned "---" separator
text = re.sub(r'^\n*---\n', '', text)

# Strip inline section numbers so pandoc's auto-numbering doesn't double them.
# e.g.  "## 1. Introduction" -> "## Introduction"
#        "### 3.1 The divergence…" -> "### The divergence…"
text = re.sub(r'^(#{1,6} )\d+(\.\d+)*\.? ', r'\1', text, flags=re.MULTILINE)

# Also strip "Finding N:" prefixes in subsection headers (keep the label name)
# "### Finding 1: Gemma…" -> "### Finding 1: Gemma…"  (leave as is — descriptive)

# ─── Build YAML front matter ──────────────────────────────────────────────────
# pandoc 2.12 only supports flat author strings (not name/affiliation dicts)
thanks = f"Independent researcher. Developed at {venue_str}."
if contact_str:
    thanks += f" Contact: {contact_str}."

yaml = f"""---
title: "BioRefusalAudit: Auditing Biosecurity Refusal Depth Using General and Domain-Fine-Tuned Sparse Autoencoders"
author: "{author_str}"
date: "{date_str}"
geometry: "margin=1in"
fontsize: 11pt
colorlinks: true
urlcolor: blue
linkcolor: blue
...

"""

final_md = yaml + text.lstrip()
MD_TEMP.write_text(final_md, encoding="utf-8")
print(f"[1/4] Pre-processed markdown -> {MD_TEMP.name}")

# ─── Step 2: pandoc -> LaTeX ──────────────────────────────────────────────────

latex_cmd = [
    "pandoc",
    str(MD_TEMP),
    "--from=markdown+tex_math_dollars+pipe_tables+raw_tex",
    "--to=latex",
    "--standalone",
    "--listings",
    "--shift-heading-level-by=-1",
    # NO --number-sections: markdown headers already carry numbers in their text
    f"--resource-path={PAPER}",
    "-o", str(TEX_OUT),
]

result = subprocess.run(latex_cmd, capture_output=True, text=True, cwd=str(PAPER))
if result.returncode != 0:
    print("pandoc LaTeX error:", result.stderr)
    sys.exit(1)
print(f"[2/4] LaTeX -> {TEX_OUT.name}")

# ─── Step 3: Post-process .tex ───────────────────────────────────────────────

tex = TEX_OUT.read_text(encoding="utf-8")

# 3a. Fix author block: replace pandoc's simple \author{name} with
#     \author{name \\ \small affiliation}  and add \thanks for venue
thanks_tex = venue_str.replace("&", r"\&").replace("#", r"\#")
tex = re.sub(
    r'\\author\{[^}]*\}',
    (
        r'\\author{' + author_str + r'\\thanks{' + thanks_tex + r'.}'
        r'\\\\\\small Independent researcher}'
    ),
    tex,
    count=1,
)

# 3b. Remove duplicate \usepackage lines that header-includes would add
#     (pandoc already adds these automatically; keep first occurrence only)
seen_packages = set()
clean_lines = []
for line in tex.splitlines():
    m = re.match(r'\\usepackage(\[[^\]]*\])?\{([^}]+)\}', line.strip())
    if m:
        pkg = m.group(2)
        if pkg in seen_packages:
            continue
        seen_packages.add(pkg)
    clean_lines.append(line)
tex = "\n".join(clean_lines)

# 3c. Convert \section{Abstract} + following paragraphs to \begin{abstract}
#     Find the Abstract section and wrap it properly
abstract_pattern = re.compile(
    r'\\hypertarget\{abstract\}\{%\n\\section\{Abstract\}\\label\{abstract\}\}\n\n(.*?)(?=\n\\hypertarget|\Z)',
    re.DOTALL,
)
m = abstract_pattern.search(tex)
if m:
    abstract_body = m.group(1).strip()
    replacement = f"\\begin{{abstract}}\n{abstract_body}\n\\end{{abstract}}\n\n"
    tex = tex[:m.start()] + replacement + tex[m.end():]
    print("     Abstract -> \\begin{abstract}...\\end{abstract}")

# 3d. Suppress section auto-numbering (section titles already have numbers)
if r'\setcounter{secnumdepth}' not in tex:
    tex = tex.replace(
        r'\begin{document}',
        r'\setcounter{secnumdepth}{0}' + '\n' + r'\begin{document}',
    )

TEX_OUT.write_text(tex, encoding="utf-8")
print(f"[3/4] Post-processed .tex saved")

# ─── Step 4: Generate PDF via HTML -> xhtml2pdf ──────────────────────────────

html_cmd = [
    "pandoc",
    str(MD_TEMP),
    "--from=markdown+tex_math_dollars+pipe_tables",
    "--to=html5",
    "--standalone",
    "--mathjax",
    "--shift-heading-level-by=-1",
    f"--resource-path={PAPER}",
    "-o", str(HTML_TEMP),
]
subprocess.run(html_cmd, capture_output=True, cwd=str(PAPER))

html_content = HTML_TEMP.read_text(encoding="utf-8")

# Strip external CSS/JS that xhtml2pdf can't parse (MathJax etc.)
html_content = re.sub(r'<link[^>]+rel=["\']stylesheet["\'][^>]*/>', '', html_content)
html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)

# Render LaTeX-style math as plain text approximations in HTML
# (xhtml2pdf can't render MathJax)
def math_to_text(m):
    inner = m.group(1).strip()
    # Basic readable substitutions
    inner = inner.replace(r'\cos', 'cos').replace(r'\cdot', '·')
    inner = inner.replace(r'\mathbb{R}', 'ℝ').replace(r'\in', '∈')
    inner = inner.replace(r'^T', 'ᵀ').replace(r'^{5 \times 5}', '⁵ˣ⁵')
    inner = inner.replace(r'\times', '×').replace('\\', '')
    return f'<code style="font-size:10pt">{inner}</code>'

html_content = re.sub(r'\\\[(.*?)\\\]', math_to_text, html_content, flags=re.DOTALL)
html_content = re.sub(r'\\\((.*?)\\\)', math_to_text, html_content, flags=re.DOTALL)

css = """
<style>
  @page { margin: 1in; }
  body { font-family: "Times New Roman", Times, serif; font-size: 11pt;
         line-height: 1.5; color: #000; }
  h1 { font-size: 15pt; text-align: center; margin-bottom: 4px; }
  h2 { font-size: 13pt; margin-top: 24px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
  h3 { font-size: 11pt; margin-top: 16px; font-style: italic; }
  h4 { font-size: 11pt; margin-top: 12px; }
  .author, .date { text-align: center; font-size: 11pt; }
  table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 12px 0; }
  th, td { border: 1px solid #999; padding: 4px 8px; text-align: left; }
  th { background: #eee; font-weight: bold; }
  td[align="right"], th[align="right"] { text-align: right; }
  code { font-family: monospace; font-size: 9pt; }
  pre { background: #f5f5f5; padding: 8px; font-size: 9pt; }
  img { max-width: 100%; display: block; margin: 12px auto; }
  blockquote { border-left: 3px solid #ccc; margin: 8px 0; padding-left: 12px; }
  hr { border: none; border-top: 1px solid #ccc; margin: 20px 0; }
  em { font-style: italic; }
  strong { font-weight: bold; }
  p { margin: 6px 0; }
  ul, ol { margin: 6px 0; padding-left: 20px; }
  li { margin: 3px 0; }
</style>
"""
html_content = html_content.replace("</head>", css + "</head>")
HTML_TEMP.write_text(html_content, encoding="utf-8")

try:
    from xhtml2pdf import pisa
    with open(HTML_TEMP, "rb") as src, open(PDF_OUT, "wb") as dest:
        status = pisa.CreatePDF(src, dest=dest, path=str(PAPER) + "/")
    if status.err:
        print(f"     xhtml2pdf: {status.err} warning(s) (PDF still written)")
    size_kb = PDF_OUT.stat().st_size // 1024
    print(f"[4/4] PDF -> {PDF_OUT.name} ({size_kb} KB)")
except Exception as e:
    print(f"[4/4] PDF failed: {e}")

MD_TEMP.unlink(missing_ok=True)
HTML_TEMP.unlink(missing_ok=True)
print("Done. arXiv upload: zip submission.tex + figures/ directory.")
