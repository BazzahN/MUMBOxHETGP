'''
Includes the functions for plotting and data
handeling in the anlysis
'''
import os
import re
import torch
import pandas as pd
import json
from pathlib import Path
from exp_utils import TKWARGS

def extract_hyperparameters(indir):
    # directory containing json files
    

    # containers
    scalar_rows = []
    vector_rows = []

    # regex to extract m and t from filename
    pattern = re.compile(f"^data_(\d+)\.json$")
    for filename in os.listdir(indir):
        if not filename.endswith(".json"):
            continue

        match = pattern.match(filename)
        if not match:
            continue

        m = int(match.group(1))

        filepath = os.path.join(indir, filename)

        with open(filepath, "r") as f:
            data = json.load(f)

        # -----------------------
        # Scalar parameters
        # -----------------------
        scalar_row = {
            "m": m,
            "tau_1": data.get("tau_1"),
            "tau_2": data.get("tau_2"),
            "l_1": data.get("l_1"),
            "l_2": data.get("l_2"),
            "mu_1": data.get("mu_1"),
            "mu_2": data.get("mu_2"),
            "llhood":data.get("llhood")
        }
        scalar_rows.append(scalar_row)

        # -----------------------
        # Vector parameters
        # -----------------------
        mu_u_1 = data.get("mu_u_1", [])
        mu_u_2 = data.get("mu_u_2", [])
        u = data.get("u", [])

        # ensure equal length
        n = min(len(mu_u_1), len(mu_u_2), len(u))

        for i in range(n):
            vector_rows.append({
                "m": m,
                "index": i,
                "mu_u_1": mu_u_1[i],
                "mu_u_2": mu_u_2[i],
                "u": u[i],
            })

    # -----------------------
    # Create DataFrames
    # -----------------------
    df_GP_params = pd.DataFrame(scalar_rows)
    df_inducing_points = pd.DataFrame(vector_rows)

    # optional: sort
    df_GP_params = df_GP_params.sort_values(["m"]).reset_index(drop=True)
    df_inducing_points  = df_inducing_points .sort_values(["m", "index"]).reset_index(drop=True)

    return df_GP_params,df_inducing_points


def calculate_SE(preds,test_out):
	'''
	SE(x) = \(\hat{f}(x) - f(x))^2
	'''

	return (preds - test_out)**2


def get_files(indir,file_names,suffix=None):

	data = {}
	
	for file_name in file_names:
		if suffix is not None:
			get_name = file_name + suffix
		else:
			get_name = file_name
		load_in = torch.load(indir /  f"{get_name}.pt").to(**TKWARGS)
		data[file_name] = load_in
	return data