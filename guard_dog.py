# Created by Bhumik Tandon
# File Name: guard_dog.py
# Purpose: Loads the trained model to monitor logs in real-time and alert on anomalies.

import time
import torch
import os
from drain3 import TemplateMiner
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_PATH = "./savior_brain"
LOG_FILE = "server_activity_logs.txt"
THRESHOLD = 6.0

if not os.path.exists(MODEL_PATH):
    print("Error: Model not found. Run train_savior.py first!")
    exit()

print("Loading Server-Savior-AI...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
template_miner = TemplateMiner()

def check_log(line):
    template = template_miner.add_log_message(line.strip())["template_mined"]
    inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
    return loss

def follow(file):
    file.seek(0, 2)
    while True:
        line = file.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line

print(f"Server-Savior is watching {LOG_FILE}...")
try:
    with open(LOG_FILE, "r") as f:
        for line in follow(f):
            score = check_log(line)
            if score > THRESHOLD:
                print(f"THREAT DETECTED [Score: {score:.2f}]: {line.strip()}")
            else:
                print(f"[Safe: {score:.2f}]", end="\r")
except FileNotFoundError:
    print(f"Make sure {LOG_FILE} exists!")