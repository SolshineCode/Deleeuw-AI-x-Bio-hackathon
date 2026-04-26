"""One-shot fix for colab_gemma4_sae_training.ipynb bugs identified by Gemini audit."""
import json

nb_path = r"C:\Users\caleb\projects\Deleeuw-AI-x-Bio-hackathon\notebooks\colab_gemma4_sae_training.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))
cells = nb["cells"]

print("Current cells:")
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))[:100]
    print(f"  [{i}] {repr(src)}")

# -----------------------------------------------------------------------
# Fix 1: Insert missing [4/7] cell with TopKSAE class + sae object
#        between index 4 ([3/7]) and index 5 ([5/7])
# -----------------------------------------------------------------------
sae_cell_lines = [
    "# [4/7] TopKSAE Architecture\n",
    "import torch.nn as nn\n",
    "\n",
    "# D_MODEL is set in [3/7] (Gemma 4 E2B hidden size = 1536)\n",
    "D_SAE = 6144   # 4x expansion ratio\n",
    "K     = 32     # TopK sparsity matching Gemma Scope convention\n",
    "\n",
    "class TopKSAE(nn.Module):\n",
    '    """TopK Sparse Autoencoder.\n',
    "\n",
    "    forward(x) -> (x_hat, z, pre_relu):\n",
    "      x_hat    - reconstruction (L_recon)\n",
    "      z        - sparse feature activations (L_contrastive)\n",
    "      pre_relu - pre-activation logits (L_sparsity L1 penalty)\n",
    '    """\n',
    "    def __init__(self, d_model, d_sae, k):\n",
    "        super().__init__()\n",
    "        self.d_model, self.d_sae, self.k = d_model, d_sae, k\n",
    "        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.02)\n",
    "        self.b_enc = nn.Parameter(torch.zeros(d_sae))\n",
    "        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.02)\n",
    "        self.b_dec = nn.Parameter(torch.zeros(d_model))\n",
    "        self.normalize_decoder()\n",
    "\n",
    "    def encode(self, x):\n",
    "        pre = x @ self.W_enc + self.b_enc\n",
    "        pre_relu = torch.relu(pre)\n",
    "        topk_vals, topk_idx = torch.topk(pre_relu, self.k, dim=-1)\n",
    "        out = torch.zeros_like(pre_relu)\n",
    "        out.scatter_(-1, topk_idx, topk_vals)\n",
    "        return out\n",
    "\n",
    "    def decode(self, z):\n",
    "        return z @ self.W_dec + self.b_dec\n",
    "\n",
    "    def forward(self, x):\n",
    "        pre = x @ self.W_enc + self.b_enc\n",
    "        z   = self.encode(x)\n",
    "        x_hat = self.decode(z)\n",
    "        return x_hat, z, pre\n",
    "\n",
    "    @torch.no_grad()\n",
    "    def normalize_decoder(self):\n",
    "        n = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)\n",
    "        self.W_dec.data.div_(n)\n",
    "\n",
    "    @torch.no_grad()\n",
    "    def project_grad(self):\n",
    "        if self.W_dec.grad is None:\n",
    "            return\n",
    "        g, w = self.W_dec.grad, self.W_dec.data\n",
    "        g.sub_((g * w).sum(dim=1, keepdim=True) * w)\n",
    "\n",
    "\n",
    "sae = TopKSAE(D_MODEL, D_SAE, K).to('cuda').float()\n",
    "print(f'SAE: d_model={D_MODEL}, d_sae={D_SAE}, k={K}')\n",
    "print(f'Parameters: {sum(p.numel() for p in sae.parameters()):,}')\n",
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": sae_cell_lines,
}

cells.insert(5, new_cell)
print("Inserted [4/7] TopKSAE cell at index 5")

# -----------------------------------------------------------------------
# Fix 2: Remove duplicate [5/7] cell (now at index 7 after insertion)
#        The one WITHOUT REPO_AVAILABLE guard is the duplicate
# -----------------------------------------------------------------------
dup_src = "".join(cells[7].get("source", []))
if "# [5/7] Configurable Dataset Loading" in dup_src and "REPO_AVAILABLE" not in dup_src:
    cells.pop(7)
    print("Removed duplicate [5/7] cell at index 7")
else:
    print(f"Duplicate check: index 7 = {repr(dup_src[:80])}")

# -----------------------------------------------------------------------
# Fix 3: Add 'import torch.nn.functional as F' to [6/7] cell (now index 7)
# -----------------------------------------------------------------------
cell_67 = cells[7]
src_67 = "".join(cell_67.get("source", []))
if "# [6/7] Training Loop Logic" in src_67 and "import torch.nn.functional as F" not in src_67:
    cell_67["source"] = ["import torch.nn.functional as F\n", "\n"] + list(cell_67["source"])
    print("Added F import to [6/7] cell")
elif "import torch.nn.functional as F" in src_67:
    print("[6/7] already has F import")
else:
    print(f"WARNING: index 7 is not [6/7]: {repr(src_67[:60])}")

# -----------------------------------------------------------------------
# Fix 4: Fix training loop in [7/7] (now index 8)
#        Replace model.generate(32 steps) + model(out) with single forward pass
# -----------------------------------------------------------------------
cell_77 = cells[8]
src_77 = "".join(cell_77.get("source", []))
if "# [7/7] Main Execution Loop" in src_77:
    # Identify the broken generate block
    bad_pattern = (
        "        with torch.no_grad():\n"
        "            # Capture activations during generation (simplified to re-run full sequence)\n"
        "            out = model.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.7)\n"
        "            _ = model(out)"
    )
    good_pattern = (
        "        with torch.no_grad():\n"
        "            # Single forward pass: captures all prompt token activations at once.\n"
        "            # model.generate() fires the hook once per new token (32 times),\n"
        "            # so hook.last would only hold the last generated token's activation.\n"
        "            _ = model(**inputs)"
    )
    if bad_pattern in src_77:
        new_src = src_77.replace(bad_pattern, good_pattern)
        cell_77["source"] = [new_src]
        print("Fixed [7/7]: replaced generate loop with single forward pass")
    else:
        print("Bad pattern not found verbatim; trying substring fix...")
        lines = src_77.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'model.generate(**inputs' in line:
                # Replace this line and the following _ = model(out)
                new_lines.append("        with torch.no_grad():")
                new_lines.append("            # Single forward pass captures all sequence activations.")
                new_lines.append("            _ = model(**inputs)")
                # Skip the model(out) line if it follows
                if i + 1 < len(lines) and '_ = model(out)' in lines[i + 1]:
                    i += 2
                    continue
            elif '_ = model(out)' in line:
                i += 1
                continue
            elif '# Capture activations during generation' in line:
                i += 1
                continue
            elif 'with torch.no_grad():' in line and i + 1 < len(lines) and 'model.generate' in lines[i + 1]:
                # Skip the with block header; the fix above adds it
                i += 1
                continue
            else:
                new_lines.append(line)
            i += 1
        new_src = '\n'.join(new_lines)
        cell_77["source"] = [new_src]
        print("Fixed [7/7] via line-level substitution")
else:
    print(f"WARNING: index 8 not [7/7]: {repr(src_77[:60])}")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\nFinal cell structure:")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))[:90]
    print(f"  [{i}] {repr(src)}")

nb["cells"] = cells
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\nWrote colab_gemma4_sae_training.ipynb")
