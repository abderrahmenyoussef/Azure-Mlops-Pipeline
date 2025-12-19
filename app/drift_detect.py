import matplotlib
matplotlib.use("Agg")  # OBLIGATOIRE pour Docker / Azure

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import os
from datetime import datetime


def detect_drift(
    reference_file,
    production_file,
    threshold=0.05,
    output_dir="drift_reports",
):
    """
    Détecte le data drift entre un jeu de référence et un jeu de production
    en utilisant le test KS (Kolmogorov-Smirnov)
    """
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"Reference file not found: {reference_file}")

    if not os.path.exists(production_file):
        raise FileNotFoundError(f"Production file not found: {production_file}")

    os.makedirs(output_dir, exist_ok=True)

    ref = pd.read_csv(reference_file)
    prod = pd.read_csv(production_file)

    results = {}

    for col in ref.columns:
        if col != "Exited" and col in prod.columns:
            ref_col = ref[col].dropna()
            prod_col = prod[col].dropna()

            if ref_col.empty or prod_col.empty:
                continue

            stat, p_value = ks_2samp(ref_col, prod_col)

            results[col] = {
                "p_value": float(p_value),
                "statistic": float(stat),
                "drift_detected": bool(p_value < threshold),
            }

    report_path = os.path.join(
        output_dir,
        f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
