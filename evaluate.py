# Created by Bhumik Tandon
# Scores a held-out synthetic stream. Labels are used here only. Training never sees them.

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timedelta

from transformers import AutoModelForMaskedLM, AutoTokenizer

from savior_common import (
    MODEL_DIR,
    THRESHOLDS_PATH,
    parse_timestamp,
    PseudoPerplexity,
    RepeatWindow,
    build_miner,
    decide,
    failed_publickey,
    invalid_user,
    load_thresholds,
    nginx_access,
    random_benign,
    ssh_failed,
    sudo_denied,
)

REPORT_PATH = MODEL_DIR.parent.parent / "eval_report.json"


def build_stream(seed: int = 19) -> list[dict]:
    rng = random.Random(seed)
    when = datetime(2026, 12, 1, 9, 0, 0)
    events: list[dict] = []

    def advance(gap: int) -> datetime:
        nonlocal when
        when += timedelta(seconds=gap)
        return when

    def push(kind: str, line: str, episode: str | None = None) -> None:
        events.append(
            {"kind": kind, "line": line, "episode": episode, "timestamp": parse_timestamp(line)}
        )

    for _ in range(50):
        push("normal", random_benign(rng, advance(rng.randint(2, 5))))
    for index in range(12):
        push(
            "isolated_fail",
            ssh_failed(advance(90), "root", f"198.51.100.{index + 1}", 3000 + index),
        )
        push("normal", nginx_access(rng, advance(4)))
    for episode in range(8):
        push("normal", random_benign(rng, advance(120)))
        for step in range(8):
            push(
                "brute",
                ssh_failed(advance(3), "root", f"203.0.113.{20 + episode}", 7000 + step),
                episode=f"burst-{episode}",
            )
    for index in range(4):
        push(
            "novel",
            invalid_user(advance(30), "oracle", f"203.0.113.{40 + index}", 8000 + index),
        )
        push("novel", sudo_denied(advance(20), "dev"))
    for index in range(4):
        push(
            "near_variant",
            failed_publickey(advance(30), "root", f"198.51.100.{40 + index}", 9000 + index),
        )
    for _ in range(20):
        push("normal", random_benign(rng, advance(rng.randint(2, 5))))
    return events


def main() -> None:
    if not MODEL_DIR.exists() or not THRESHOLDS_PATH.exists():
        raise SystemExit("Model not found. Run train_savior.py first.")
    thresholds = load_thresholds()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)
    scorer = PseudoPerplexity(model, tokenizer)
    miner = build_miner(load_existing=True)
    window = RepeatWindow(int(thresholds["window_seconds"]))

    counts = defaultdict(int)
    false_alarms = defaultdict(int)
    detected = defaultdict(int)
    episodes = defaultdict(list)
    nll_by_kind = defaultdict(list)
    examples = []
    seen_kinds = set()

    for event in build_stream():
        decision = decide(miner, scorer, window, thresholds, event["line"], event["timestamp"])
        kind = event["kind"]
        counts[kind] += 1
        nll_by_kind[kind].append(decision.nll)
        if kind in {"normal", "isolated_fail"} and decision.alert:
            false_alarms[kind] += 1
        if kind == "novel" and decision.perplexity_alert:
            detected["novel"] += 1
        if kind == "near_variant" and decision.perplexity_alert:
            detected["near_variant"] += 1
        if event["episode"]:
            episodes[event["episode"]].append(decision.brute_alert)
        worth_showing = decision.alert or kind in {"isolated_fail", "near_variant"}
        if worth_showing and kind not in seen_kinds:
            seen_kinds.add(kind)
            examples.append(
                {
                    "kind": kind,
                    "nll": round(decision.nll, 3),
                    "repeats": decision.repeats,
                    "repeat_limit": decision.repeat_limit,
                    "perplexity_alert": decision.perplexity_alert,
                    "brute_alert": decision.brute_alert,
                    "line": event["line"],
                }
            )

    bursts_caught = sum(1 for flags in episodes.values() if any(flags))
    report = {
        "normal_lines": counts["normal"],
        "normal_false_alarms": false_alarms["normal"],
        "isolated_failures": counts["isolated_fail"],
        "isolated_false_alarms": false_alarms["isolated_fail"],
        "brute_episodes": len(episodes),
        "brute_episodes_detected": bursts_caught,
        "novel_lines": counts["novel"],
        "novel_detected": detected["novel"],
        "near_variant_lines": counts["near_variant"],
        "near_variant_detected": detected["near_variant"],
        "nll_threshold": thresholds["nll_threshold"],
        "mean_nll_by_kind": {kind: sum(values) / len(values) for kind, values in nll_by_kind.items()},
        "failed_password_alert_at": next(
            (peak + thresholds["repeat_margin"])
            for template, peak in thresholds["repeat_peak_by_template"].items()
            if "Failed password" in template
        ),
        "window_seconds": thresholds["window_seconds"],
        "parameter_count": thresholds["parameter_count"],
        "examples": examples,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"Normal lines: {report['normal_lines']}, false alarms: {report['normal_false_alarms']}")
    print(
        f"Isolated failed logins: {report['isolated_failures']}, "
        f"false alarms: {report['isolated_false_alarms']}"
    )
    print(
        f"Brute-force episodes: {report['brute_episodes_detected']}/{report['brute_episodes']} "
        f"(alert at {report['failed_password_alert_at']} repeats / {report['window_seconds']}s)"
    )
    print(f"Novel templates: {report['novel_detected']}/{report['novel_lines']}")
    print(
        f"One-word variant (Failed publickey): "
        f"{report['near_variant_detected']}/{report['near_variant_lines']}"
    )
    print("Mean NLL by kind:")
    for kind, score in report["mean_nll_by_kind"].items():
        print(f"  {kind:16s} {score:.3f}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
