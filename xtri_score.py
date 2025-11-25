#!/usr/bin/env python3
"""
XTRI Scoring Tool (CLI)
-----------------------
Compute Cross‑Tenant Exposure Risk (CTER) and subscores for AI-in-Cloud case studies.

Usage:
  python xtri_score.py --input case.yaml --out-prefix OpenAI_Mar2023

The input YAML defines:
  - posture: 0–3 scores for 12 dimensions
  - controls: C,E,Q,M,R in [0,1] per dimension (to compute m_j)
  - optional params: weights, gamma emphasis, propagation edges, beta, kappa

Outputs:
  - <out-prefix>_table.csv   (per-dimension posture, m_j, residual z_j)
  - <out-prefix>_summary.json (CTER raw, overall 0–100, subscores, hotspots)
"""

import argparse, json, math, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

try:
    import yaml  # pyyaml
except Exception as e:
    print("Missing dependency: pyyaml. Install with: pip install pyyaml", file=sys.stderr)
    raise

# --- Dimensions and roles ---
DIMS: Dict[int, Tuple[str, str]] = {
    1: ("Public Network Exposure", "Amplifier"),
    2: ("Tenant Isolation", "Primary cause"),
    3: ("Minimum Privilege to Reach Data", "Primary cause"),
    4: ("Authentication Strength", "Primary cause"),
    5: ("Data Sensitivity", "Magnitude maker"),
    6: ("Encryption (At Rest/In Transit)", "Mitigation/detector"),
    7: ("Access Policy Hygiene", "Primary cause"),
    8: ("Temp Links / SAS Discipline", "Amplifier"),
    9: ("Cross-Account/Project Sharing", "Amplifier"),
    10: ("Snapshot/Backup Exposure", "Magnitude maker"),
    11: ("Logging Privacy & Minimization", "Primary cause"),
    12: ("Config Drift Guardrails", "Mitigation/detector"),
}

DEFAULT_WEIGHTS = {
    "m": {"wC":0.30, "wE":0.25, "wQ":0.15, "wM":0.15, "wR":0.15},
    "S": {2:0.30, 3:0.20, 4:0.15, 7:0.20, 11:0.15},
    "A": {1:0.50, 8:0.30, 9:0.20},
    "M": {6:0.30, 12:0.30, 11:0.40},
    "beta": 0.8,
    "kappa": 0.5,
}

DEFAULT_GAMMA = {j: 1.2 for j in [1,2,3,4,7,8,9,10,11]}
for j in [5,6,12]:
    DEFAULT_GAMMA[j] = 1.0

DEFAULT_PROP_EDGES = [
    # (from_dim, to_dim, weight)
    (8, 1, 0.10),   # long-lived links behave like public exposure
    (2, 3, 0.10),   # weak isolation stresses privilege boundaries
    (2, 11, 0.10),  # weak isolation increases log privacy risk under faults
]

def m_from_components(C, E, Q, M, R, caps=True, w=None):
    if w is None:
        w = DEFAULT_WEIGHTS["m"]
    m = w["wC"]*C + w["wE"]*E + w["wQ"]*Q + w["wM"]*M + w["wR"]*R
    # Safety caps
    if caps:
        # Low-coverage cap
        if C < 0.50:
            m = min(m, 0.60)
    return m

def compute_scores(posture: Dict[int,int],
                   controls: Dict[int,Dict[str,float]],
                   weights: Dict=None,
                   gamma: Dict[int,float]=None,
                   prop_edges: List[Tuple[int,int,float]]=None):
    weights = weights or DEFAULT_WEIGHTS
    gamma = gamma or DEFAULT_GAMMA
    prop_edges = prop_edges or DEFAULT_PROP_EDGES

    # 1) Normalize posture
    x_norm = {j: posture[j]/3.0 for j in DIMS.keys()}

    # 2) Control effectiveness m_j
    m_eff = {j: m_from_components(**controls[j], w=weights["m"]) for j in DIMS.keys()}

    # 3) Residuals
    z = {j: x_norm[j]*(1-m_eff[j]) for j in DIMS.keys()}

    # 4) Emphasis
    z_tilt = {j: (z[j] ** (gamma.get(j,1.0))) for j in DIMS.keys()}

    # 5) Propagation over CTE drivers
    cte_dims = [1,2,3,4,7,8,9,10,11]
    A = pd.DataFrame(0.0, index=cte_dims, columns=cte_dims)
    for a,b,w in prop_edges:
        if a in cte_dims and b in cte_dims:
            A.loc[a,b] = float(w)
    z_vec = pd.Series({j: z_tilt[j] for j in cte_dims})
    I = np.eye(len(cte_dims))
    p_cte = pd.Series(((I + A.values + (A.values @ A.values)) @ z_vec.values),
                      index=cte_dims).to_dict()

    def r(j):
        return p_cte[j] if j in cte_dims else z[j]

    # 6) Aggregation
    S = list(weights["S"].keys())
    Agrp = list(weights["A"].keys())
    M = list(weights["M"].keys())
    beta = float(weights.get("beta", 0.8))
    kappa = float(weights.get("kappa", 0.5))

    Base = sum(weights["S"][j]*r(j) for j in S)
    Amp = sum(weights["A"][j]*r(j) for j in Agrp)
    Impact = 1 + beta*(z_tilt[5]) + 0.5*(z_tilt[10])
    Mit = sum(weights["M"][j]*z[j] for j in M)

    CTER = Base * Impact * (1 + Amp) * (1 - kappa * Mit)
    overall = round(100 * (1 - math.exp(-CTER)), 1)

    # Subscores (readable mapping)
    to_0_100 = lambda v: round(100 * (1 - math.exp(-3*v)), 1)
    subs = {
        "Isolation": (z[2] + z[1]/2)/ (1 + 0.5),
        "Secrets/Dependencies": (z[7] + z[3]) / 2,
        "Data Handling": (z[5] + z[11] + z[10]) / 3,
        "Detect/Respond": (z[12] + (1 - m_eff[2])) / 2,
        "AI Surface": (z[4])
    }
    subscores = {k: to_0_100(v) for k,v in subs.items()}

    # Table
    rows = []
    for j in DIMS:
        name, role = DIMS[j]
        rows.append({
            "Dim": j,
            "Dimension": name,
            "Role": role,
            "Posture (0-3)": posture[j],
            "x_norm": round(x_norm[j],3),
            "Control m_j": round(m_eff[j],3),
            "Residual z_j": round(z[j],3)
        })
    df = pd.DataFrame(rows).sort_values("Dim")

    summary = {
        "CTER_raw": round(CTER, 4),
        "Overall_0_100": overall,
        "Subscores_0_100": subscores,
        "Top_Residuals": sorted([(int(j), DIMS[j][0], round(z[j],3)) for j in z],
                                key=lambda t: t[2], reverse=True)[:5]
    }
    return df, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="YAML with posture/controls/optional params")
    ap.add_argument("--out-prefix", required=True, help="Output file prefix")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        cfg = yaml.safe_load(f)

    posture = {int(k): int(v) for k,v in cfg["posture"].items()}
    controls = {int(k): {kk: float(vv) for kk,vv in cfg["controls"][str(k)].items()} for k in posture.keys()}

    weights = cfg.get("weights", DEFAULT_WEIGHTS)
    # normalize numeric keys back to int for S/A/M if provided
    for key in ["S","A","M"]:
        if key in weights:
            weights[key] = {int(k): float(v) for k,v in weights[key].items()}

    gamma = {int(k): float(v) for k,v in cfg.get("gamma", DEFAULT_GAMMA).items()}
    prop_edges = [(int(a), int(b), float(w)) for (a,b,w) in cfg.get("prop_edges", DEFAULT_PROP_EDGES)]

    df, summary = compute_scores(posture, controls, weights, gamma, prop_edges)

    csv_path = f"{args.out_prefix}_table.csv"
    json_path = f"{args.out_prefix}_summary.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
