# Created by Bhumik Tandon
# File Name: train_savior.py
# Purpose: Generates synthetic data and trains the SLM from scratch to learn normal server patterns.

import torch
import random
import os
from datetime import datetime, timedelta
from drain3 import TemplateMiner
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DistilBertConfig,
    DistilBertForMaskedLM
)
from torch.utils.data import Dataset

LOG_FILENAME = "server_activity_logs.txt"
SAVE_PATH = "./savior_brain"

def generate_synthetic_logs():
    users = ["admin", "root", "dev", "deploy"]
    ips = ["192.168.1.5", "10.0.0.2", "172.16.0.5"]
    attack_ip = "203.0.113.66"
    
    with open(LOG_FILENAME, "w") as f:
        for index in range(300):
            timestamp = (datetime.now() + timedelta(seconds=index)).strftime("%H:%M:%S")
            if 250 <= index < 280:
                f.write(f"{timestamp} sshd[123]: Failed password for root from {attack_ip} port 22\n")
            else:
                user = random.choice(users)
                ip = random.choice(ips)
                if random.random() > 0.5:
                    f.write(f"{timestamp} sshd[123]: Accepted publickey for {user} from {ip} port 22\n")
                else:
                    f.write(f"{timestamp} nginx: GET /index.html HTTP/1.1 200 512\n")

if not os.path.exists(LOG_FILENAME):
    generate_synthetic_logs()

template_miner = TemplateMiner()
clean_templates = []
with open(LOG_FILENAME, "r") as f:
    for line in f:
        clean_templates.append(template_miner.add_log_message(line.strip())["template_mined"])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class LogDataset(Dataset):
    def __init__(self, texts):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].detach().clone()
        return item
    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = LogDataset(clean_templates[:200])

config = DistilBertConfig(vocab_size=30522, hidden_size=128, num_hidden_layers=2, num_attention_heads=4)
model = DistilBertForMaskedLM(config)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=50,
    per_device_train_batch_size=16,
    learning_rate=1e-3,
    save_steps=500,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
)

print("Training Server-Savior-AI...")
trainer.train()

print(f"Saving model to {SAVE_PATH}...")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("Done! You can now run guard_dog.py")