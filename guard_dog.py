# Created by Bhumik Tandon
# Tails a log file and alerts on a novel template or a repeated source.

from __future__ import annotations

import time

from transformers import AutoModelForMaskedLM, AutoTokenizer

from savior_common import (
    LOG_FILE,
    MODEL_DIR,
    THRESHOLDS_PATH,
    Decision,
    PseudoPerplexity,
    RepeatWindow,
    build_miner,
    decide,
    load_thresholds,
    parse_timestamp,
    quiet_loggers,
)


def require_training() -> None:
    if not MODEL_DIR.exists() or not THRESHOLDS_PATH.exists():
        raise SystemExit("Model not found. Run train_savior.py first.")


def format_decision(decision: Decision, line: str) -> str:
    limit = "-" if decision.repeat_limit is None else str(decision.repeat_limit)
    detail = f"nll={decision.nll:.2f}  repeats={decision.repeats}  limit={limit}"
    if decision.brute_alert and decision.perplexity_alert:
        label = "BRUTE+NOVEL"
    elif decision.brute_alert:
        label = "BRUTE FORCE"
    elif decision.perplexity_alert:
        label = "NOVEL"
    else:
        label = "ok"
    ip = f"  ip={decision.ip}" if decision.alert and decision.ip else ""
    return f"[{label:<11}] {detail}{ip} | {line.strip()}"


def follow(path):
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(0, 2)
        while True:
            line = handle.readline()
            if line:
                yield line
                continue
            time.sleep(0.2)


def main() -> None:
    quiet_loggers()
    require_training()
    LOG_FILE.touch(exist_ok=True)
    thresholds = load_thresholds()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)
    model.eval()
    scorer = PseudoPerplexity(model, tokenizer)
    miner = build_miner(load_existing=True)
    window = RepeatWindow(int(thresholds["window_seconds"]))

    print(f"Watching {LOG_FILE.name}")
    print(
        f"Novel template when mean NLL >= {thresholds['nll_threshold']:.2f}. "
        f"Brute force when one IP repeats a known template "
        f"{thresholds['repeat_margin']} above its normal peak inside "
        f"{thresholds['window_seconds']}s."
    )
    print("Start attack_simulator.py in another terminal. Ctrl+C stops this watcher.")
    try:
        for line in follow(LOG_FILE):
            if not line.strip():
                continue
            decision = decide(miner, scorer, window, thresholds, line, parse_timestamp(line))
            print(format_decision(decision, line), flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
