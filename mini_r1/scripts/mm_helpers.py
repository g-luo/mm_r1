import pandas as pd
import wandb
import json
import torch
import PIL

def check_for_csv(v):
    try:
        return [json.dumps(_v) for _v in v]
    except Exception as e:
        return None

def log_completion(completions, **kwargs):
    # if random.random() < 0.1:  # 1% chance to write samples into a file
    results = {"completions": completions}
    results = {**results, **kwargs}
    results = {k: check_for_csv(v) for k, v in results.items()}
    results = {k: v for k, v in results.items() if v is not None}
    print(results)
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False, encoding='utf-8')
    if wandb.run is not None:
        wandb.log({"completion": wandb.Table(dataframe=df)})