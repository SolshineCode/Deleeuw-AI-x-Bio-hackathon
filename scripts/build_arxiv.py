"""
Build arXiv-ready LaTeX and PDF from paper/submission.md.

Steps:
  1. Pre-process markdown: fix image paths, extract metadata, inject YAML front matter
  2. Run pandoc -> .tex
  3. Post-process .tex: clean preamble for arXiv compatibility
  4. Generate PDF via pandoc -> HTML -> xhtml2pdf
"""

import re
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SRC = PAPER / "submission.md"
TEX_OUT = PAPER / "submission.tex"
PDF_OUT = PAPER / "submission.pdf"
HTML_TEMP = PAPER / "_temp_submission.html"
MD_TEMP = PAPER / "_temp_submission_arxiv.md"

# ─── Step 1: Pre-process markdown ────────────────────────────────────────────

text = SRC.read_text(encoding="utf-8")

# Fix image paths: ../demo/X -> figures/X
text = re.sub(r'\.\./demo/([\w_\-\.]+\.png)', r'figures/\1', text)

# Extract the bold metadata block just below the H1
# It looks like:  **Author:** Caleb DeLeeuw
author  = re.search(r'\*\*Author:\*\*\s*(.+)', text)
contact = re.search(r'\*\*Contact:\*\*\s*(.+)', text)
date_m  = re.search(r'\*\*Date:\*\*\s*(.+)', text)
venue_m = re.search(r'\*\*Venue:\*\*\s*(.+)', text)
affil_m = re.search(r'\*\*Affiliation:\*\*\s*(.+)', text)

author_str  = author.group(1).strip()  if author  else "Caleb DeLeeuw"
date_str    = date_m.group(1).strip()  if date_m  else "May 2026"
affil_str   = affil_m.group(1).strip() if affil_m else "Independent researcher"
venue_str   = venue_m.group(1).strip() if venue_m else ""
contact_str = contact.group(1).strip() if contact else "caleb.deleeuw@gmail.com"

# Remove the title + bold metadata block from the body
# (pandoc will render it from YAML front matter instead)
text = re.sub(
    r'^# BioRefusalAudit.*?\n\n\*\*Author:\*\*.*?(?=\n---)',
    '',
    text,
    count=1,
    flags=re.DOTALL
)

# Strip the leading "---" separator that would otherwise be left behind
text = re.sub(r'^\n*---\n', '', text)

# Build YAML front matter
yaml = f"""---
title: "BioRefusalAudit: Auditing Biosecurity Refusal Depth Using General and Domain-Fine-Tuned Sparse Autoencoders"
author:
  - name: "{author_str}"
    affiliation: "Independent researcher"
    email: "{contact_str}"
date: "{date_str}"
geometry: "margin=1in"
fontsize: 11pt
numbersections: true
colorlinks: true
urlcolor: blue
linkcolor: blue
header-includes: |
  \\usepackage{{booktabs}}
  \\usepackage{{longtable}}
  \\usepackage{{array}}
  \\usepackage{{multirow}}
  \\usepackage{{graphicx}}
  \\usepackage{{float}}
  \\usepackage{{amsmath}}
  \\usepackage{{amssymb}}
  \\usepackage{{microtype}}
  \\usepackage{{hyperref}}
  \\hypersetup{{breaklinks=true}}
  \\setlength{{\\emergencystretch}}{{3em}}
  \\providecommand{{\\tightlist}}{{\\setlength{{\\itemsep}}{{0pt}}\\setlength{{\\parskip}}{{0pt}}}}
...

"""

final_md = yaml + text.lstrip()
MD_TEMP.write_text(final_md, encoding="utf-8")
print(f"[1/4] Pre-processed markdown written to {MD_TEMP.name}")

# ─── Step 2: pandoc → LaTeX ───────────────────────────────────────────────────

latex_cmd = [
    "pandoc",
    str(MD_TEMP),
    "--from=markdown+tex_math_dollars+pipe_tables+raw_tex",
    "--to=latex",
    "--standalone",
    "--listings",
    "--shift-heading-level-by=-1",   # ## Introduction -> \section, ### Finding -> \subsection
    "--number-sections",
    f"--resource-path={PAPER}",
    "-o", str(TEX_OUT),
]

result = subprocess.run(latex_cmd, capture_output=True, text=True, cwd=str(PAPER))
if result.returncode != 0:
    print("pandoc LaTeX error:", result.stderr)
    sys.exit(1)
print(f"[2/4] LaTeX written to {TEX_OUT.name}")

# ─── Step 3: Post-process .tex for arXiv ────────────────────────────────────

tex = TEX_OUT.read_text(encoding="utf-8")

# arXiv: remove the pandoc-generated abstract filler ("See §Abstract below.")
tex = tex.replace("See §Abstract below.", "")

# Ensure \begin{document} is preceded by \maketitle
# pandoc handles this but make sure it's right
if r"\maketitle" not in tex:
    tex = tex.replace(r"\begin{document}", r"\begin{document}" + "\n" + r"\maketitle")

# Add venue as a \thanks footnote on the author if not already present
if "Apart Research" not in tex[:tex.find(r"\begin{document}")]:
    apart_note = venue_str.replace("&", r"\&").replace("#", r"\#")
    tex = re.sub(
        r'(\\author\{[^}]*)',
        r'\1\\thanks{Developed at ' + apart_note + '.}',
        tex,
        count=1
    )

TEX_OUT.write_text(tex, encoding="utf-8")
print(f"[3/4] Post-processed .tex saved")

# ─── Step 4: Generate PDF via HTML → xhtml2pdf ───────────────────────────────

html_cmd = [
    "pandoc",
    str(MD_TEMP),
    "--from=markdown+tex_math_dollars+pipe_tables",
    "--to=html5",
    "--standalone",
    "--mathjax",
    "--shift-heading-level-by=-1",
    "--number-sections",
    f"--resource-path={PAPER}",
    "-o", str(HTML_TEMP),
]

result = subprocess.run(html_cmd, capture_output=True, text=True, cwd=str(PAPER))
if result.returncode != 0:
    print("pandoc HTML error:", result.stderr)
    sys.exit(1)

# Strip external stylesheets and MathJax script tags that xhtml2pdf can't parse
html_content = HTML_TEMP.read_text(encoding="utf-8")
html_content = re.sub(r'<link[^>]+rel=["\']stylesheet["\'][^>]*/>', '', html_content)
html_content = re.sub(r'<script[^>]*mathjax[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
html_content = re.sub(r'<script[^>]*MathJax[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
# Also strip <style> blocks inserted by pandoc (keep only our own)
html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)

css = """
<style>
  body { font-family: "Times New Roman", Times, serif; font-size: 11pt;
         max-width: 700px; margin: 40px auto; line-height: 1.5; color: #000; }
  h1 { font-size: 16pt; text-align: center; margin-bottom: 4px; }
  h2 { font-size: 13pt; margin-top: 24px; }
  h3 { font-size: 11pt; margin-top: 18px; }
  table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 12px 0; }
  th, td { border: 1px solid #999; padding: 4px 8px; }
  th { background: #eee; }
  code { font-family: monospace; font-size: 9pt; background: #f5f5f5; padding: 1px 3px; }
  pre { background: #f5f5f5; padding: 8px; overflow-x: auto; font-size: 9pt; }
  img { max-width: 100%; height: auto; display: block; margin: 12px auto; }
  blockquote { border-left: 3px solid #ccc; margin: 0; padding-left: 12px; color: #555; }
  .author, .date { text-align: center; font-size: 11pt; }
  hr { border: none; border-top: 1px solid #ccc; margin: 24px 0; }
</style>
"""
html_content = html_content.replace("</head>", css + "</head>")
# Fix image paths in HTML: figures/ -> figures/ (already correct since cwd=PAPER)
HTML_TEMP.write_text(html_content, encoding="utf-8")

# Convert HTML → PDF with xhtml2pdf
try:
    from xhtml2pdf import pisa
    with open(HTML_TEMP, "rb") as src, open(PDF_OUT, "wb") as dest:
        pisa_status = pisa.CreatePDF(src, dest=dest, path=str(PAPER) + "/")
    if pisa_status.err:
        print(f"xhtml2pdf reported {pisa_status.err} error(s) — PDF may be partial")
    else:
        size_kb = PDF_OUT.stat().st_size // 1024
        print(f"[4/4] PDF written to {PDF_OUT.name} ({size_kb} KB)")
except Exception as e:
    print(f"[4/4] PDF generation failed: {e}")
    print("      The .tex is arXiv-ready; compile locally with: pdflatex submission.tex")

# Cleanup temp files
MD_TEMP.unlink(missing_ok=True)
HTML_TEMP.unlink(missing_ok=True)
print("Done.")
