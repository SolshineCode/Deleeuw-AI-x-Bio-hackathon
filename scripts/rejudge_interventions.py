import json
from pathlib import Path
from biorefusalaudit.judges.regex_classifier import regex_classify
from biorefusalaudit.judges.consortium import aggregate_votes, soft_distribution_vector
from biorefusalaudit.runner.eval_runner import _regex_as_vote

def rejudge(dir_path="runs/interventions"):
    files = sorted(Path(dir_path).glob("*.json"))
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        print(f"--- {f.name} ---")
        for mode in ["baseline", "ablated", "boosted"]:
            if mode not in data: continue
            completion = data[mode]["completion"]
            old_label = data[mode]["label"]
            old_conf = data[mode]["confidence"]
            
            vote = _regex_as_vote(completion)
            verdict = aggregate_votes([vote])
            new_label = verdict.label
            new_conf = float(max(verdict.soft_distribution.values()))
            
            print(f"  {mode:8s}: {old_label} ({old_conf:.2f}) -> {new_label} ({new_conf:.2f})")
            if old_label != new_label:
                print(f"    FIXED! Matched: {vote.raw_response}")

if __name__ == "__main__":
    rejudge()
