#!/usr/bin/env python3
"""
build_pdf.py — Build a publication-ready HTML from submission.md with figures embedded.
Open the output in Chrome and print to PDF.

Usage: python scripts/build_pdf.py
Output: paper/submission_print.html
"""

import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PAPER = ROOT / "paper" / "submission.md"
DEMO = ROOT / "demo"
OUT = ROOT / "paper" / "submission_print.html"

def b64_img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    ext = path.suffix.lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"

def md_table_to_html(md: str) -> str:
    """Convert markdown tables to HTML tables."""
    lines = md.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect table header row (contains | and next line is separator)
        if "|" in line and i + 1 < len(lines) and re.match(r"\s*\|[\s\-|:]+\|", lines[i+1]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 1  # skip separator
            rows = []
            while i + 1 < len(lines) and "|" in lines[i+1]:
                i += 1
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(row)
            # Build HTML table
            th = "".join(f"<th>{h}</th>" for h in header)
            trs = ""
            for row in rows:
                trs += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            out.append(f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>')
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)

def md_to_html_body(md: str) -> str:
    """Very lightweight markdown to HTML conversion (pandoc handles the heavy lifting)."""
    import subprocess, tempfile, os

    # Write md to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(md)
        tmp_in = f.name

    tmp_out = tmp_in.replace('.md', '.html')
    try:
        subprocess.run(
            ['pandoc', tmp_in, '--from', 'markdown', '--to', 'html5', '-o', tmp_out],
            check=True, capture_output=True
        )
        body = Path(tmp_out).read_text(encoding='utf-8')
    finally:
        os.unlink(tmp_in)
        if Path(tmp_out).exists():
            os.unlink(tmp_out)

    return body

def main():
    md = PAPER.read_text(encoding='utf-8')

    # Convert to HTML body via pandoc
    body = md_to_html_body(md)

    # Embed scaling plot figure after "Figure 1" reference
    scaling_png = DEMO / "scaling_plot_real.png"
    if not scaling_png.exists():
        scaling_png = DEMO / "scaling_plot.png"

    sae_curves = DEMO / "sae_training_curves.png"

    fig1_img = ""
    if scaling_png.exists():
        src = b64_img(scaling_png)
        fig1_img = f'''
<figure class="figure">
  <img src="{src}" alt="Figure 1: Per-tier mean D comparison across models" style="max-width:100%;border:1px solid #ddd;padding:8px;border-radius:4px;">
  <figcaption><strong>Figure 1.</strong> Per-tier mean D comparison across Gemma 2 2B-IT (Gemma Scope 1, 80-tok) and Gemma 4 E2B-IT (author SAE, 150-tok). The Gemma 4 label-split (comply vs. refuse) shows the metric cleanly discriminates posture classes. The Gemma 2 tier separation shows tier-level ordering within the eval set.</figcaption>
</figure>'''

    sae_fig = ""
    if sae_curves.exists():
        src = b64_img(sae_curves)
        sae_fig = f'''
<figure class="figure">
  <img src="{src}" alt="SAE Training Curves" style="max-width:100%;border:1px solid #ddd;padding:8px;border-radius:4px;">
  <figcaption><strong>Figure 2.</strong> Domain-specific SAE training curves (<code>Solshine/gemma4-e2b-bio-sae-v1</code>), 2000 steps on WMDP bio-retain corpus. The contrastive loss plateau reflects shared vocabulary across bio tiers; behavioral activation corpora are needed for stronger separation.</figcaption>
</figure>'''

    # Insert figure 1 after the paragraph that references scaling_plot.png
    body = body.replace(
        '</p>\n<p><strong>Cross-architecture behavioral comparison',
        f'{fig1_img}\n</p>\n<p><strong>Cross-architecture behavioral comparison'
    )

    # Insert SAE figure before the Future work section
    if sae_fig:
        body = body.replace(
            '<h2 id="code-and-data">',
            f'{sae_fig}\n<h2 id="code-and-data">'
        )

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=Source+Code+Pro:wght@400&display=swap');

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #1a1a1a;
        background: white;
        max-width: 720px;
        margin: 0 auto;
        padding: 40px 40px 60px;
    }

    @media print {
        body { max-width: 100%; margin: 0; padding: 20px 30px; }
        a { color: inherit; text-decoration: none; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    }

    h1 {
        font-size: 18pt;
        font-weight: 600;
        line-height: 1.3;
        margin-bottom: 12px;
        color: #111;
    }

    h2 {
        font-size: 13pt;
        font-weight: 600;
        margin-top: 28px;
        margin-bottom: 8px;
        color: #111;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 4px;
    }

    h3 {
        font-size: 11.5pt;
        font-weight: 600;
        margin-top: 18px;
        margin-bottom: 6px;
        color: #222;
    }

    p {
        margin-bottom: 10px;
    }

    .metadata {
        font-size: 10pt;
        color: #555;
        margin-bottom: 16px;
        line-height: 1.8;
    }

    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 20px 0;
    }

    blockquote {
        border-left: 3px solid #ccc;
        margin: 12px 0 12px 20px;
        padding: 4px 16px;
        color: #444;
        font-style: italic;
    }

    code {
        font-family: 'Source Code Pro', 'Courier New', monospace;
        font-size: 9.5pt;
        background: #f5f5f5;
        padding: 1px 4px;
        border-radius: 3px;
        color: #c7254e;
    }

    pre {
        background: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 12px;
        overflow-x: auto;
        margin: 12px 0;
        font-size: 9pt;
        line-height: 1.5;
    }

    pre code {
        background: none;
        padding: 0;
        color: #333;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 9.5pt;
        page-break-inside: avoid;
    }

    th {
        background: #f0f0f0;
        font-weight: 600;
        text-align: left;
        padding: 6px 10px;
        border: 1px solid #ccc;
    }

    td {
        padding: 5px 10px;
        border: 1px solid #ddd;
        vertical-align: top;
    }

    tr:nth-child(even) td { background: #fafafa; }

    ul, ol {
        margin: 8px 0 10px 24px;
    }

    li { margin-bottom: 4px; }

    strong { font-weight: 600; }
    em { font-style: italic; }

    figure.figure {
        margin: 20px 0;
        text-align: center;
        page-break-inside: avoid;
    }

    figure.figure img {
        max-width: 100%;
    }

    figcaption {
        font-size: 9.5pt;
        color: #555;
        margin-top: 8px;
        text-align: left;
        font-style: italic;
    }

    a { color: #1a5276; }

    .abstract-box {
        background: #f9f9f9;
        border-left: 4px solid #2471a3;
        padding: 12px 16px;
        margin: 16px 0;
        font-size: 10.5pt;
    }
    """

    # Extract and reformat the bold metadata lines at the top
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BioRefusalAudit: Auditing Biosecurity Refusal Depth Using General and Domain-Fine-Tuned Sparse Autoencoders</title>
<style>
{css}
</style>
</head>
<body>
{body}
</body>
</html>"""

    OUT.write_text(html, encoding='utf-8')
    print(f"Written: {OUT}")
    print(f"Size: {OUT.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
