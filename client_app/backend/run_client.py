# backend/run_client.py
import json, time, sys

print("Starting fake Flower client...")

for epoch in range(1, 6):
    time.sleep(1)
    print(json.dumps({
        "type": "metrics",
        "epoch": epoch,
        "loss": 1/epoch,
        "acc": 0.5 + 0.1*epoch
    }))

print("Client done.")
