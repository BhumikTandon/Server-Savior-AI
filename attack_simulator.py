# Created by Bhumik Tandon
# File Name: attack_simulator.py
# Purpose: Injects logs into the file to test if guard_dog.py detects them.

import time
import random

LOG_FILE = "server_activity_logs.txt"

print(f"Injecting logs into {LOG_FILE}...")
print("Press Ctrl+C to stop.")

while True:
    time.sleep(2) # Wait 2 seconds
    
    # 90% chance of Normal, 10% chance of Attack
    if random.random() > 0.1:
        log = "12:00:00 sshd[123]: Accepted publickey for admin from 192.168.1.5 port 22"
        print(f"Adding NORMAL log: {log}")
    else:
        log = "12:00:00 sshd[123]: Failed password for root from 203.0.113.66 port 22"
        print(f"Adding ATTACK log: {log}")
        
    with open(LOG_FILE, "a") as f:
        f.write(log + "\n")