# -*- coding: utf-8 -*-
"""Complete one omitted checkbox from the user's explicit confirmation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = Path(
    r"C:/Users/lulu/Downloads/anime_semantic_acceptance_feedback_v1_2026-08-02T13-33-19-259Z.json"
)
OUTPUT = ROOT / "anime_semantic_acceptance_v1" / "feedback_completed_user_confirmed.json"


def main() -> None:
    feedback = json.loads(INPUT.read_text(encoding="utf-8"))
    found = False
    for row in feedback.get("samples", []):
        if str(row.get("blind_id")) == "S007":
            row["reviewed"] = True
            row["is_error"] = False
            row["error_type"] = ""
            row["note"] = "用户明确确认：该片段没有问题"
            found = True
    if not found:
        raise RuntimeError("S007 is not present in feedback")
    samples = list(feedback.get("samples", []))
    reviewed = sum(bool(row.get("reviewed")) for row in samples)
    errors = sum(bool(row.get("is_error")) for row in samples)
    feedback["version"] = "anime_semantic_acceptance_feedback_v1_user_confirmed"
    feedback["updated_at"] = datetime.now(timezone.utc).isoformat()
    feedback["complete"] = reviewed == len(samples)
    feedback["summary"] = {
        "reviewed": reviewed,
        "errors": errors,
        "complete": feedback["complete"],
        "completion_reason": "S007 explicitly confirmed by user as no issue",
    }
    if not feedback["complete"]:
        raise RuntimeError(f"Feedback is still incomplete: {reviewed}/{len(samples)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(feedback, ensure_ascii=False, indent=2) + "/n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "reviewed": reviewed, "errors": errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()
