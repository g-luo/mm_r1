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

def log_completion(completions, log_path="results.csv", log_interval=1, **kwargs):
    if not hasattr(log_completion, "idx"):
        log_completion.idx = 0
    log_completion.idx += 1

    if log_completion.idx % log_interval == 0:
        try:
            results = {"idx": [log_completion.idx] * len(completions), "completions": completions}
            results = {**results, **kwargs}
            results = {k: check_for_csv(v) for k, v in results.items()}
            results = {k: v for k, v in results.items() if v is not None}
            df = pd.DataFrame(results)
            df.to_csv(log_path, mode="a", index=False, encoding='utf-8')
            if wandb.run is not None:
                df = pd.read_csv(log_path)
                wandb.log({"results": wandb.Table(dataframe=df)})
        except Exception as e:
            print(e)
            pass