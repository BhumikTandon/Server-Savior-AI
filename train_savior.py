# Created by Bhumik Tandon
# Trains a small DistilBERT from scratch on normal log templates and
# calibrates both detectors on a later normal stream. Attack lines are not used.

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertConfig, DistilBertForMaskedLM

from savior_common import (
    ARTIFACTS,
    DRAIN_STATE,
    MAX_LENGTH,
    MODEL_DIR,
    NLL_MARGIN,
    REPEAT_MARGIN,
    THRESHOLDS_PATH,
    WINDOW_SECONDS,
    PseudoPerplexity,
    RepeatWindow,
    build_miner,
    generate_normal_corpus,
    invalid_user,
    parse_line,
    parse_timestamp,
    quiet_loggers,
    sudo_denied,
)

SEED = 7
TRAIN_LINES = 2400
VAL_LINES = 600
EPOCHS = 8
BATCH_SIZE = 64
LEARNING_RATE = 5e-4


class TemplateDataset(Dataset):
    def __init__(self, token_rows: list[list[int]]) -> None:
        self.token_rows = token_rows

    def __len__(self) -> int:
        return len(self.token_rows)

    def __getitem__(self, index: int) -> list[int]:
        return self.token_rows[index]


def mask_tokens(batch: torch.Tensor, attention: torch.Tensor, tokenizer, probability: float = 0.15):
    """Mask 15% of real tokens. Inference scores the same way: one [MASK] at a time."""
    labels = batch.clone()
    special = torch.zeros_like(batch, dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        special |= batch.eq(token_id)
    odds = torch.full(batch.shape, probability)
    odds.masked_fill_(special | attention.eq(0), 0.0)
    chosen = torch.bernoulli(odds).bool()
    for row in range(chosen.size(0)):
        if chosen[row].any():
            continue
        eligible = (~special[row] & attention[row].bool()).nonzero(as_tuple=False)
        if len(eligible):
            chosen[row, eligible[0, 0]] = True
    labels[~chosen] = -100
    inputs = batch.clone()
    inputs[chosen] = tokenizer.mask_token_id
    return inputs, labels


def make_collate(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(rows: list[list[int]]) -> dict[str, torch.Tensor]:
        tensors = [torch.tensor(row, dtype=torch.long) for row in rows]
        batch = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_id)
        attention = batch.ne(pad_id).long()
        inputs, labels = mask_tokens(batch, attention, tokenizer)
        return {"input_ids": inputs, "attention_mask": attention, "labels": labels}

    return collate


def templates_for(miner, lines: list[str]) -> list[str]:
    templates = []
    for line in lines:
        template, known, _ips = parse_line(miner, line)
        if not known:
            raise RuntimeError(f"Normal line did not match a trained template:\n{line}\n{template}")
        templates.append(template)
    return templates


def require_separate_failure_template(templates: list[str]) -> None:
    failed = [template for template in set(templates) if "Failed password" in template]
    accepted = [template for template in set(templates) if "Accepted" in template]
    if len(failed) != 1 or not accepted:
        rendered = "\n".join(sorted(set(templates)))
        raise RuntimeError(
            "Drain merged success and failure lines into one template, so a brute-force "
            f"burst would look like normal logins.\n{rendered}"
        )


def main() -> None:
    quiet_loggers()
    random.seed(SEED)
    torch.manual_seed(SEED)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    if DRAIN_STATE.exists():
        DRAIN_STATE.unlink()

    train_lines = generate_normal_corpus(TRAIN_LINES, seed=SEED, start=datetime(2026, 11, 1, 8, 0, 0))
    val_lines = generate_normal_corpus(VAL_LINES, seed=SEED + 1, start=datetime(2026, 11, 8, 8, 0, 0))

    miner = build_miner(load_existing=False)
    for line in train_lines:
        miner.add_log_message(line)
    train_templates = templates_for(miner, train_lines)
    require_separate_failure_template(train_templates)
    miner.save_state("training corpus mined")

    print("Templates learned from normal traffic:")
    for template, count in Counter(train_templates).most_common():
        print(f"  {count:4d}  {template}")

    print("Downloading the DistilBERT tokenizer (weights are not used; the model starts random)...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    token_rows = tokenizer(train_templates, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        n_layers=2,
        n_heads=4,
        hidden_dim=512,
        max_position_embeddings=MAX_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = DistilBertForMaskedLM(config)
    parameter_count = sum(item.numel() for item in model.parameters())
    print(f"Training a {parameter_count:,}-parameter DistilBERT for {EPOCHS} epochs on CPU...")

    loader = DataLoader(
        TemplateDataset(token_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate(tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        steps = 0
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
            if steps % 10 == 0:
                print(f"  epoch {epoch:02d}  step {steps:03d}  loss {total / steps:.4f}", flush=True)
        print(f"  epoch {epoch:02d}  mean MLM loss {total / steps:.4f}", flush=True)

    model.eval()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    scorer = PseudoPerplexity(model, tokenizer)
    val_templates = templates_for(miner, val_lines)
    val_nll = [scorer.mean_nll(template) for template in val_templates]
    nll_threshold = max(val_nll) + NLL_MARGIN

    window = RepeatWindow(WINDOW_SECONDS)
    peaks: dict[str, int] = defaultdict(int)
    for line in val_lines:
        template, _known, ips = parse_line(miner, line)
        peaks.setdefault(template, 0)
        when = parse_timestamp(line)
        for ip in ips:
            peaks[template] = max(peaks[template], window.observe(template, ip, when))

    thresholds = {
        "nll_threshold": nll_threshold,
        "nll_margin": NLL_MARGIN,
        "validation_nll_max": max(val_nll),
        "validation_nll_mean": sum(val_nll) / len(val_nll),
        "window_seconds": WINDOW_SECONDS,
        "repeat_margin": REPEAT_MARGIN,
        "repeat_peak_by_template": dict(peaks),
        "parameter_count": parameter_count,
        "train_lines": TRAIN_LINES,
        "validation_lines": VAL_LINES,
    }
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))

    failed = next(template for template in peaks if "Failed password" in template)
    novel_time = datetime(2026, 11, 9, 12, 0, 0)
    novel_nll = scorer.mean_nll(parse_line(miner, invalid_user(novel_time, "oracle", "203.0.113.20", 2500))[0])
    sudo_nll = scorer.mean_nll(parse_line(miner, sudo_denied(novel_time, "dev"))[0])
    print(f"Validation mean NLL {thresholds['validation_nll_mean']:.3f}, max {max(val_nll):.3f}")
    print(f"Novel-template threshold {nll_threshold:.3f}")
    print(f"Unseen 'Invalid user' NLL {novel_nll:.3f}; unseen sudo denial NLL {sudo_nll:.3f}")
    print(
        f"Failed-password repeat peak on validation {peaks[failed]}; "
        f"brute-force alert at {peaks[failed] + REPEAT_MARGIN} in {WINDOW_SECONDS}s"
    )
    print(f"Saved model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
