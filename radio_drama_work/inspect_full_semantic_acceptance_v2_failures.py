# -*- coding: utf-8 -*-
"""Inspect word-level speaker evidence around the seven v2 review failures."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_full_semantic_candidates_v2 as v2


LABELS = ROOT / "human_full_semantic_acceptance_v2" / "labels_resolved.json"
WORDS = ROOT / "full_source_local_word_scores_v1" / "source_reports"
MASKS = ROOT / "full_semantic_mining_v2" / "hard_exclusion_masks.json"
OUTPUT = ROOT / "human_full_semantic_acceptance_v2" / "failure_word_context.json"


def main() -> None:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["samples"]
    masks = json.loads(MASKS.read_text(encoding="utf-8"))
    result = []
    cache = {}
    for label in labels:
        if label["role"] != "machine_candidate_v2" or label["final_class"] == "target":
            continue
        source_id = str(label["source_id"])
        if source_id not in cache:
            words = json.loads(
                (WORDS / f"{source_id}.json").read_text(encoding="utf-8")
            )["word_rows"]
            states = v2.classify_words(words, masks.get(source_id, []))
            cache[source_id] = (words, states)
        words, states = cache[source_id]
        left = int(label["word_start_index"])
        right = int(label["word_end_index"])
        context_left = max(0, left - 12)
        context_right = min(len(words) - 1, right + 12)
        context = []
        for index in range(context_left, context_right + 1):
            word = words[index]
            state = states[index]
            context.append(
                {
                    "index": index,
                    "inside_candidate": left <= index <= right,
                    "text": str(word.get("text") or ""),
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "state": state["state"],
                    "probability": state["probability"],
                    "strict": state["strict"],
                    "context_target_fraction": state["context"],
                    "cluster_margin": state["cluster"],
                    "anchor_margin": state["anchor"],
                    "target_center_similarity": state["target"],
                    "target_center_delta": state["delta"],
                    "hard_negative": state["hard_negative"],
                }
            )
        result.append(
            {
                "blind_id": label["blind_id"],
                "final_class": label["final_class"],
                "human_note": label["human_note"],
                "source_id": source_id,
                "source": label["source"],
                "candidate_start": label["start"],
                "candidate_end": label["end"],
                "candidate_word_start_index": left,
                "candidate_word_end_index": right,
                "candidate_text": label["text"],
                "word_context": context,
            }
        )
    OUTPUT.write_text(
        json.dumps(
            {
                "version": "full_semantic_acceptance_v2_failure_word_context",
                "samples": result,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "/n",
        encoding="utf-8",
    )
    for item in result:
        print(
            f"[{item['blind_id']}] {item['final_class']} {item['source_id']} "
            f"{item['candidate_start']:.3f}-{item['candidate_end']:.3f} "
            f"words={item['candidate_word_start_index']}-{item['candidate_word_end_index']} "
            f"{item['candidate_text']}",
            flush=True,
        )
        for word in item["word_context"]:
            marker = "*" if word["inside_candidate"] else " "
            print(
                f" {marker}{word['index']:4d} {word['start']:8.3f}-{word['end']:8.3f} "
                f"{word['state']:13s} ctx={word['context_target_fraction']:.2f} "
                f"cl={word['cluster_margin']:+.3f} anc={word['anchor_margin']:+.3f} "
                f"d={word['target_center_delta']:+.3f} {word['text']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
