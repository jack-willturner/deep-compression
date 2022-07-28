import os
import torch

import pandas as pd
from tabulate import tabulate

def get_no_params(net, verbose=False, mask=False):
    params = net
    tot = 0
    for p in params:
        no = torch.sum(params[p] != 0)
        if "conv" in p:
            tot += no
    return tot
    
checkpoint_files = [f for f in os.listdir(".") if f.endswith(".t7")]

df = []

for checkpoint_file in checkpoint_files:
    sd = torch.load(checkpoint_file)

    epoch = sd["epoch"]
    errs = sd["error_history"][-1]
    num = get_no_params(sd["net"])


    df.append([checkpoint_file, epoch, errs, num])

df = pd.DataFrame(df, columns=["Filename", "Epoch", "Latest Error","Para Num"])

print(tabulate(df, headers="keys", tablefmt="psql"))

df.to_pickle("latest_results.pd")
