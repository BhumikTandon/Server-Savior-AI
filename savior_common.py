# Shared log synthesis, Drain3 parsing, and the two detectors.
# Created by Bhumik Tandon

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import torch
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "savior_brain"
DRAIN_STATE = ARTIFACTS / "drain3_state.bin"
THRESHOLDS_PATH = ARTIFACTS / "thresholds.json"
DRAIN_CONFIG = ROOT / "drain3.ini"
LOG_FILE = ROOT / "server_activity_logs.txt"

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
WINDOW_SECONDS = 60
# Alert when one IP repeats a known template this many times above the
# busiest example of that template in the normal validation stream.
REPEAT_MARGIN = 4
# Pseudo-perplexity uses mean NLL. Alert this far above the worst normal line.
NLL_MARGIN = 0.5
MAX_LENGTH = 64

USERS = ["admin", "deploy", "dev", "git", "ci", "backup", "metrics", "ops"]
INTERNAL_IPS = [f"10.0.{octet}.{host}" for octet in range(4) for host in range(1, 31)]
PATHS = ["/index.html", "/health", "/api/status", "/static/app.js", "/login", "/robots.txt"]
COMMANDS = ["/usr/bin/backup.sh", "/usr/sbin/logrotate", "/usr/local/bin/cert-renew"]


def quiet_loggers() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("drain3").setLevel(logging.ERROR)


def stamp(ts: datetime) -> str:
    return ts.strftime(TIMESTAMP_FMT)


def parse_timestamp(line: str) -> datetime:
    try:
        return datetime.strptime(line[:19], TIMESTAMP_FMT)
    except ValueError:
        return datetime.now().replace(microsecond=0)


def ssh_accepted(rng: random.Random, ts: datetime) -> str:
    method = rng.choice(["publickey", "password"])
    user = rng.choice(USERS)
    ip = rng.choice(INTERNAL_IPS)
    pid = rng.randint(1000, 9999)
    return f"{stamp(ts)} sshd[{pid}]: Accepted {method} for {user} from {ip} port 22"


def ssh_failed(ts: datetime, user: str, ip: str, pid: int) -> str:
    return f"{stamp(ts)} sshd[{pid}]: Failed password for {user} from {ip} port 22"


def nginx_access(rng: random.Random, ts: datetime) -> str:
    ip = rng.choice(INTERNAL_IPS)
    path = rng.choice(PATHS)
    status = rng.choice([200, 200, 200, 304])
    size = rng.randint(120, 4000) if status == 200 else 0
    return f'{stamp(ts)} nginx: {ip} - - "GET {path} HTTP/1.1" {status} {size}'


def cron_job(rng: random.Random, ts: datetime) -> str:
    user = rng.choice(USERS)
    pid = rng.randint(1000, 9999)
    command = rng.choice(COMMANDS)
    return f"{stamp(ts)} CRON[{pid}]: ({user}) CMD ({command})"


def systemd_session(rng: random.Random, ts: datetime) -> str:
    user = rng.choice(USERS)
    session = rng.randint(10, 900)
    return f"{stamp(ts)} systemd: Started session {session} of user {user}."


def invalid_user(ts: datetime, user: str, ip: str, pid: int) -> str:
    return f"{stamp(ts)} sshd[{pid}]: Invalid user {user} from {ip} port 22"


def failed_publickey(ts: datetime, user: str, ip: str, pid: int) -> str:
    return f"{stamp(ts)} sshd[{pid}]: Failed publickey for {user} from {ip} port 22"


def sudo_denied(ts: datetime, user: str) -> str:
    return (
        f"{stamp(ts)} sudo: {user} : user NOT in sudoers ; "
        f"TTY=pts/1 ; PWD=/home/{user} ; USER=root ; COMMAND=/usr/bin/passwd"
    )


def random_benign(rng: random.Random, ts: datetime) -> str:
    roll = rng.random()
    if roll < 0.46:
        return ssh_accepted(rng, ts)
    if roll < 0.78:
        return nginx_access(rng, ts)
    if roll < 0.90:
        return cron_job(rng, ts)
    return systemd_session(rng, ts)


def generate_normal_corpus(count: int, seed: int, start: datetime) -> list[str]:
    """Normal traffic only. A failed password is a typo, at most once per IP per 3 minutes."""
    rng = random.Random(seed)
    ts = start
    last_fail: dict[str, datetime] = {}
    lines: list[str] = []
    while len(lines) < count:
        ts += timedelta(seconds=rng.randint(1, 4))
        if rng.random() < 0.08:
            ip = rng.choice(INTERNAL_IPS)
            previous = last_fail.get(ip)
            if previous is None or (ts - previous).total_seconds() >= 180:
                last_fail[ip] = ts
                lines.append(ssh_failed(ts, rng.choice(USERS), ip, rng.randint(1000, 9999)))
                continue
        lines.append(random_benign(rng, ts))
    return lines


def build_miner(load_existing: bool) -> TemplateMiner:
    quiet_loggers()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if not load_existing and DRAIN_STATE.exists():
        DRAIN_STATE.unlink()
    config = TemplateMinerConfig()
    config.load(str(DRAIN_CONFIG))
    return TemplateMiner(FilePersistence(str(DRAIN_STATE)), config)


def parse_line(miner: TemplateMiner, line: str) -> tuple[str, bool, list[str]]:
    """Return the template, whether it matched a trained cluster, and IP parameters."""
    cleaned = line.strip()
    cluster = miner.match(cleaned, full_search_strategy="fallback")
    if cluster is not None:
        template = cluster.get_template()
        known = True
    else:
        template = miner.masker.mask(cleaned)
        known = False
    extracted = miner.extract_parameters(template, cleaned, exact_matching=True)
    ips = [item.value for item in extracted or [] if item.mask_name == "IP"]
    return template, known, ips


class RepeatWindow:
    """Counts how often one IP produces one template inside a time window."""

    def __init__(self, window_seconds: int) -> None:
        self.window = timedelta(seconds=window_seconds)
        self._events: dict[tuple[str, str], deque[datetime]] = defaultdict(deque)

    def observe(self, template: str, ip: str, when: datetime) -> int:
        queue = self._events[(template, ip)]
        queue.append(when)
        cutoff = when - self.window
        while queue and queue[0] < cutoff:
            queue.popleft()
        return len(queue)


class PseudoPerplexity:
    """Mean negative log-likelihood with one token masked at a time.

    DistilBERT is bidirectional, so the token being scored has to be replaced
    with [MASK]. Leaving it visible is not perplexity.
    """

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self._cache: dict[str, float] = {}

    @torch.inference_mode()
    def mean_nll(self, text: str) -> float:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        token_ids = encoded["input_ids"][0]
        special = set(self.tokenizer.all_special_ids)
        positions = [index for index, token in enumerate(token_ids.tolist()) if token not in special]
        if not positions:
            self._cache[text] = 0.0
            return 0.0
        masked = token_ids.unsqueeze(0).repeat(len(positions), 1)
        for row, position in enumerate(positions):
            masked[row, position] = self.tokenizer.mask_token_id
        attention = torch.ones_like(masked)
        logits = self.model(input_ids=masked, attention_mask=attention).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        index_rows = torch.arange(len(positions))
        index_cols = torch.tensor(positions)
        targets = token_ids[index_cols]
        score = float((-log_probs[index_rows, index_cols, targets]).mean())
        self._cache[text] = score
        return score


def load_thresholds() -> dict:
    return json.loads(THRESHOLDS_PATH.read_text())


def brute_limit(thresholds: dict, template: str) -> int | None:
    peaks = thresholds["repeat_peak_by_template"]
    if template not in peaks:
        return None
    return int(peaks[template]) + int(thresholds["repeat_margin"])


@dataclass
class Decision:
    template: str
    known_template: bool
    nll: float
    repeats: int
    repeat_limit: int | None
    ip: str | None
    perplexity_alert: bool
    brute_alert: bool

    @property
    def alert(self) -> bool:
        return self.perplexity_alert or self.brute_alert


def decide(miner, scorer: PseudoPerplexity, window: RepeatWindow, thresholds: dict, line: str, when: datetime) -> Decision:
    template, known, ips = parse_line(miner, line)
    repeats = 0
    ip_hit = None
    for ip in ips:
        count = window.observe(template, ip, when)
        if count >= repeats:
            repeats = count
            ip_hit = ip
    limit = brute_limit(thresholds, template)
    nll = scorer.mean_nll(template)
    return Decision(
        template=template,
        known_template=known,
        nll=nll,
        repeats=repeats,
        repeat_limit=limit,
        ip=ip_hit,
        perplexity_alert=nll >= float(thresholds["nll_threshold"]),
        brute_alert=limit is not None and repeats >= limit,
    )
