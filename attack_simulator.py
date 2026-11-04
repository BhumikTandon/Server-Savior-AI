# Created by Bhumik Tandon
# Appends a scripted log scenario. This writes lines to a local file; it does not
# open sockets or try passwords.

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta

from savior_common import (
    LOG_FILE,
    cron_job,
    invalid_user,
    nginx_access,
    ssh_accepted,
    ssh_failed,
    sudo_denied,
)

PAUSE_SECONDS = 0.6


def main() -> None:
    rng = random.Random(4)
    when = datetime.now().replace(microsecond=0)
    script: list[tuple[str, str]] = []

    def add(note: str, line: str, gap: int) -> None:
        nonlocal when
        when += timedelta(seconds=gap)
        script.append((note, line))

    for _ in range(4):
        add("normal login or request; expect ok", ssh_accepted(rng, when) if rng.random() < 0.5 else nginx_access(rng, when), 3)
    add(
        "one failed password from a new address; expect ok",
        ssh_failed(when, "root", "198.51.100.23", 4410),
        20,
    )
    add("normal cron; expect ok", cron_job(rng, when), 5)
    for index in range(8):
        add(
            "brute-force failure; expect BRUTE FORCE once the same IP passes the limit",
            ssh_failed(when, "root", "203.0.113.66", 5500 + index),
            3,
        )
    add(
        "unseen ssh event; expect NOVEL",
        invalid_user(when, "oracle", "203.0.113.80", 5601),
        15,
    )
    add("unseen sudo denial; expect NOVEL", sudo_denied(when, "dev"), 8)
    add("normal login after the burst; expect ok", ssh_accepted(rng, when), 8)

    print(f"Appending {len(script)} lines to {LOG_FILE.name}. Start guard_dog.py first.")
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        for note, line in script:
            handle.write(line + "\n")
            handle.flush()
            print(f"{note}\n  {line}")
            time.sleep(PAUSE_SECONDS)
    print("Scenario finished.")


if __name__ == "__main__":
    main()
