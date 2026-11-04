# Server-Savior-AI

Unsupervised intrusion detection for server logs.

A small DistilBERT learns the wording of normal log lines. A separate counter watches how often one IP repeats a line. Perplexity catches event types the model has never seen. The counter catches brute force, which reuses a failure line the model already considers normal. A single failed login is normal traffic.

**Created by:** Bhumik Tandon

## How detection works

Drain3 masks the timestamp, IP, user, path, and counts, then clusters what remains into a template. The model never sees the raw IP. Training uses normal traffic only, including occasional typos, and does not use attack labels.

The model is a 2-layer DistilBERT (9.0M parameters) initialized at random. The DistilBERT tokenizer is the pretrained one. The weights are not. At scoring time each token is replaced with `[MASK]` in turn, and the score is the mean negative log-likelihood. That is pseudo-perplexity. Feeding the unmasked line in as the label is not, because a bidirectional model can see the token it is supposed to predict.

The IP Drain3 removed is kept for a 60-second window. Each template's alert limit is the busiest stretch of that template in a normal validation stream, plus 4. Failed passwords peaked at 1, so the 5th failure from the same IP alerts. The first four do not.

## Results

Thresholds were fixed on the normal validation stream, then applied to a later synthetic stream the trainer never saw.

| Held-out stream | Result |
| --- | --- |
| Normal lines | 90 lines, 0 false alarms |
| One-off failed logins, different IPs | 12 lines, 0 false alarms |
| Brute-force bursts (8 failures, 3 seconds apart) | 8/8 detected |
| Novel events (`Invalid user`, sudo denial) | 8/8 detected |
| One-word remix, `Failed publickey` | 0/4 detected |

Mean negative log-likelihood was 0.012 on normal lines, 0.035 on the known failure template, and 6.41 on novel events. The alert threshold is 0.54. `Failed publickey` scored 0.10: both words already appear in normal logs, so a one-token edit stays under the margin. Brute force is not caught by that score. It is caught because the same known failure template repeats.

`evaluate.py` is the only file that uses labels. `eval_report.json` is the run above.

## Run

Python 3.12. Training the checked-in configuration took about 90 seconds on CPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_savior.py
python evaluate.py
```

Then, in two terminals:

```bash
python guard_dog.py
python attack_simulator.py
```

The simulator only appends lines to `server_activity_logs.txt`. It does not open a socket. A live pass looks like this:

```text
[ok         ] nll=0.04  repeats=1  limit=5 | Failed password for root from 198.51.100.23 port 22
[ok         ] nll=0.04  repeats=4  limit=5 | Failed password for root from 203.0.113.66 port 22
[BRUTE FORCE] nll=0.04  repeats=5  limit=5  ip=203.0.113.66 | Failed password for root from 203.0.113.66 port 22
[NOVEL      ] nll=3.28  repeats=1  limit=-  ip=203.0.113.80 | Invalid user oracle from 203.0.113.80 port 22
[NOVEL      ] nll=9.55  repeats=0  limit=- | sudo: dev : user NOT in sudoers ; ... COMMAND=/usr/bin/passwd
```

## Limits

This tails a local file of synthetic logs. It is not a host agent, and a benign service the model has never seen is also a novel template. The language-model score cannot tell which IP is attacking, because that field is masked before scoring. The window is what puts the IP back.

## Layout

| File | Role |
| --- | --- |
| `train_savior.py` | Mine normal templates, train the masked LM, save both thresholds |
| `guard_dog.py` | Tail `server_activity_logs.txt` and print alerts |
| `attack_simulator.py` | Append one normal failure, one burst, and two unseen events |
| `evaluate.py` | Score the held-out stream and write `eval_report.json` |
| `savior_common.py` | Log synthesis, parsing, pseudo-perplexity, and the repeat window |
| `drain3.ini` | Masking rules. Similarity is high enough that success and failure stay different templates |

Model weights and the Drain3 snapshot are written under `artifacts/` and are not committed. Run `train_savior.py` to create them.
