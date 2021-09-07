import os
import torch

import pandas as pd
from tabulate import tabulate

checkpoint_files = [f for f in os.listdir(".") if f.endswith(".t7")]

df = []

for checkpoint_file in checkpoint_files:
    sd = torch.load(checkpoint_file)

    epoch = sd["epoch"]
    errs = sd["error_history"][-1]

    df.append([checkpoint_file, epoch, errs])

df = pd.DataFrame(df, columns=["Filename", "Epoch", "Latest Error"])

print(tabulate(df, headers="keys", tablefmt="psql"))

df.to_pickle("latest_results.pd")
