"""
Chunk cais/wmdp-bio-forget-corpus into fixed-word-count segments suitable
for SAE contrastive training.

WHY chunking is required:
  Source documents are full academic papers (median ~4K words).
  Eval prompts are ~30 words. Computing SAE activations on 4K-word
  documents produces activation vectors dominated by document-style
  features rather than bio-hazard content features, creating a format
  confound. Chunking to ~200 words narrows the gap (still a mismatch,
  but the signal-to-noise ratio is far better than whole-document).

Output schema (JSONL):
  {
    "text":        str,   # ~chunk_size words
    "tier":        str,   # always "hazard_adjacent" for forget corpus
    "source":      str,   # "cais/wmdp-bio-forget-corpus"
    "title":       str,   # paper title
    "doi":         str,   # DOI for traceability
    "doc_index":   int,   # source document index
    "chunk_index": int    # chunk position within document
  }

Usage:
  python scripts/prepare_bio_forget_corpus.py
  python scripts/prepare_bio_forget_corpus.py --max-docs 500 --chunk-size 150
  python scripts/prepare_bio_forget_corpus.py --out data/bio_forget_chunked --max-chunks 10000
"""

import argparse
import json
import statistics
from pathlib import Path


def chunk_text(text: str, chunk_size: int, overlap: int, min_words: int) -> list[str]:
    words = text.split()
    if len(words) < min_words:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) >= min_words:
            chunks.append(" ".join(chunk_words))
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk WMDP bio-forget corpus for SAE training")
    parser.add_argument("--out", default="data/bio_forget_chunked", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=200, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=20, help="Word overlap between chunks")
    parser.add_argument("--min-chunk-words", type=int, default=50, help="Minimum words to keep a chunk")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit source documents (for testing)")
    parser.add_argument("--max-chunks", type=int, default=None, help="Stop after emitting N chunks")
    args = parser.parse_args()

    from datasets import load_dataset  # noqa: PLC0415

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "bio_forget_chunked.jsonl"

    print(f"Loading cais/wmdp-bio-forget-corpus ...")
    ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
    if args.max_docs:
        ds = ds.select(range(min(args.max_docs, len(ds))))
    print(f"  {len(ds)} source documents")

    n_docs = 0
    n_chunks = 0
    n_skipped_docs = 0
    chunk_lengths = []

    with open(out_file, "w", encoding="utf-8") as f:
        for doc_index, doc in enumerate(ds):
            text = (doc.get("text") or "").strip()
            if not text:
                n_skipped_docs += 1
                continue

            chunks = chunk_text(text, args.chunk_size, args.overlap, args.min_chunk_words)
            for chunk_index, chunk in enumerate(chunks):
                record = {
                    "text": chunk,
                    "tier": "hazard_adjacent",
                    "source": "cais/wmdp-bio-forget-corpus",
                    "title": doc.get("title", ""),
                    "doi": doc.get("doi", ""),
                    "doc_index": doc_index,
                    "chunk_index": chunk_index,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_lengths.append(len(chunk.split()))
                n_chunks += 1
                if args.max_chunks and n_chunks >= args.max_chunks:
                    break

            n_docs += 1
            if args.max_chunks and n_chunks >= args.max_chunks:
                break

    print(f"\nDone.")
    print(f"  Source docs processed : {n_docs}  (skipped empty: {n_skipped_docs})")
    print(f"  Chunks written        : {n_chunks}")
    print(f"  Output                : {out_file}")
    if chunk_lengths:
        print(f"  Chunk word counts     : min={min(chunk_lengths)}, "
              f"median={statistics.median(chunk_lengths):.0f}, "
              f"max={max(chunk_lengths)}")

    # Write a small manifest alongside the JSONL
    manifest = {
        "source_dataset": "cais/wmdp-bio-forget-corpus",
        "tier": "hazard_adjacent",
        "chunk_size_words": args.chunk_size,
        "overlap_words": args.overlap,
        "min_chunk_words": args.min_chunk_words,
        "n_source_docs": n_docs,
        "n_chunks": n_chunks,
        "median_chunk_words": statistics.median(chunk_lengths) if chunk_lengths else 0,
    }
    with open(out_path / "manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"  Manifest              : {out_path / 'manifest.json'}")


if __name__ == "__main__":
    main()
