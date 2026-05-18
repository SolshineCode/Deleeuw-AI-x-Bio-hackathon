import re, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
tex = open('paper/submission_final.tex', encoding='utf-8').read()

def check(label, ok, detail=''):
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ''))

# Brace balance
check('Brace balance', tex.count('{') == tex.count('}'),
      f"{tex.count('{')} open / {tex.count('}')} close")

# Environment balance
oc = Counter(re.findall(r'\\begin\{(\w+)\}', tex))
cc = Counter(re.findall(r'\\end\{(\w+)\}', tex))
bad = {e:(oc[e],cc[e]) for e in set(oc)|set(cc) if oc[e]!=cc[e]}
check('Env balance', not bad, bad if bad else 'all matched')

# includegraphics without width options
no_w = re.findall(r'\\includegraphics\{', tex)
check('includegraphics without width', not no_w, f"{len(no_w)} found")

# passthrough defined
check('passthrough defined', r'\newcommand{\passthrough}' in tex)

# rule inside abstract
am = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.DOTALL)
if am:
    ab = am.group(1)
    check('rule in abstract', 'rule' not in ab, 'clean' if 'rule' not in ab else 'FOUND — bad')

# double-numbered figure captions
caps = re.findall(r'\\caption\{([^}]+)\}', tex)
bad_caps = [c for c in caps if re.match(r'Figure \d+:', c)]
check('double fig caption', not bad_caps, bad_caps if bad_caps else 'none')

# unicode arrows or daggers remaining
arrows = [(i+1, line.strip()[:60]) for i, line in enumerate(tex.split('\n'))
          if '→' in line or '†' in line]
check('arrow/dagger unicode gone', not arrows, arrows if arrows else 'none')

# Tables with 8+ columns (might overflow) — count l, r, c, and p columns
wide = [m.group(0)[:50] for m in re.finditer(r'\\begin\{longtable\}[^{]*\{([^}]+)\}', tex)
        if sum(m.group(1).count(c) for c in 'lrcp') >= 8]
check('tables <=7 cols', not wide, f"{len(wide)} wide table(s)" if wide else 'OK')

# hyperref setup present
check('hyperref setup', r'\hypersetup{' in tex)

# \tightlist defined
check('tightlist defined', r'\providecommand{\tightlist}' in tex)

# figure placement defined
check('figure placement set', r'\def\fps@figure{htbp}' in tex)

# listings package
check('listings package', r'\usepackage{listings}' in tex)

# amsmath
check('amsmath loaded', r'\usepackage{amsmath' in tex)

# No \bibliography{} call (would need .bib file arXiv doesn't have)
check('no external .bib needed', r'\bibliography{' not in tex)

# No absolute paths
import os
home = os.path.expanduser('~').replace('\\','/')
check('no absolute paths', home not in tex and 'C:/Users' not in tex and '/home/' not in tex)

print('\nDone.')
