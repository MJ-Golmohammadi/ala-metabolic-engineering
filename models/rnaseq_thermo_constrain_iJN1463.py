#!/usr/bin/env python3
"""
Condition-specific GEM construction using RNA-seq CSV + local ΔG file.

Requirements:
    pip install cobra pandas

Input files:
- Expression CSV (semicolon-separated): Synonym;Expression_Glu;Expression_Cit;Expression_Fer;Expression_Ser
- DeltaG CSV (semicolon-separated): reaction_id;deltaG
"""

import os
import re
import logging
import cobra
import pandas as pd
import math

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = "models/iJN1463.xml"
EXPR_FILE = "expression_txt_files/merged_expression.csv"     # semicolon-separated CSV
DG_FILE   = "models/kegg_reactions_CC_ph7.0.csv"             # CSV with reaction_id, deltaG
CSV_SEPARATOR = ";"
OUTPUT_DIR = "models/final_constrained_rnaseq_thermo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDITIONS = {
    "Glu": "Expression_Glu",
    "Cit": "Expression_Cit",
    "Fer": "Expression_Fer",
    "Ser": "Expression_Ser",
}

FLUX_CAP = 6000
DG_SCALE1 = 100      # scaling factor for bounds
DG_SCALE2 = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_flux_bound_from_tpm(tpm: float) -> float:
    try:
        val = float(tpm)
    except Exception:
        return 0.0
    if val <= 0.0:
        return 0.0
    return float(min(FLUX_CAP, val))


def evaluate_gene_rule(rule: str, expr_map: dict) -> float:
    def _eval_recursive(s: str) -> float:
        if s is None:
            return 0.0
        s = str(s).strip()
        if s == "" or s.lower() in {"none", "nan"}:
            return 0.0
        while True:
            m = re.search(r"\(([^()]+)\)", s)
            if not m:
                break
            inner_val = _eval_recursive(m.group(1))
            s = s[:m.start()] + str(inner_val) + s[m.end():]
        parts_or = [p.strip() for p in re.split(r"\s+or\s+", s, flags=re.IGNORECASE)]
        if len(parts_or) > 1:
            return max(_eval_recursive(p) for p in parts_or)
        parts_and = [p.strip() for p in re.split(r"\s+and\s+", s, flags=re.IGNORECASE)]
        if len(parts_and) > 1:
            return min(_eval_recursive(p) for p in parts_and)
        token = s.strip().strip('"').strip("'")
        try:
            return float(token)
        except Exception:
            return float(expr_map.get(token.lower(), 0.0))
    try:
        return _eval_recursive(rule)
    except Exception:
        return 0.0


def get_bounds_from_dg_file(rxn_id: str, dg_map: dict):
    """
    Look up ΔG from precomputed CSV file.
    """
    dg = dg_map.get(rxn_id, None)
    if dg is None or pd.isna(dg):
        return 0.0, FLUX_CAP
    dg = float(dg)
    if dg < -5.0:
        ub = min(FLUX_CAP, int(abs(dg) * DG_SCALE1))
        return 0.0, float(ub)
    elif -5.0 <= dg <= -1.0:
        ub = int(abs(dg) * DG_SCALE2)
        return 0.0, float(ub)
    elif -1.0 < dg < 1.0:
        ub = 50
        return 0.0, float(ub)
    elif 1.0 < dg <= 20.0:
        ub = int(50 / math.sqrt(abs(dg)))
        return 0.0, float(ub)
    else:
        ub = 10.0
        lb = -10.0
        return float(lb), float(ub)

# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------
def constrain_model(condition: str, expr_column: str, dg_map: dict):
    logging.info("Condition: %s | Expression column: %s", condition, expr_column)

    # Load expression CSV
    df_expr = pd.read_csv(EXPR_FILE, sep=CSV_SEPARATOR)
    grouped = df_expr.groupby("Synonym")[expr_column].mean().dropna()
    expr_map = {str(k).lower(): float(v) for k, v in grouped.to_dict().items()}

    model = cobra.io.read_sbml_model(MODEL_PATH)
    logging.info("Model loaded: %d reactions", len(model.reactions))

    summary_rows = []

    for rxn in model.reactions:

        # ---------------------------------------------------------
        # ✅ TPM-based expression bound
        # ---------------------------------------------------------
        tpm_value = evaluate_gene_rule(rxn.gene_reaction_rule, expr_map)
        expr_bound = get_flux_bound_from_tpm(tpm_value)
        tpm_lb, tpm_ub = -expr_bound, expr_bound

        # ---------------------------------------------------------
        # ✅ CASE 1 — EXCHANGE REACTIONS (must NOT be constrained)
        # ---------------------------------------------------------
        if rxn.id.startswith("EX_"):

            if expr_bound == 0:
                final_lb = -1e-9
                final_ub =  1e-9
            else:
                final_lb = tpm_lb
                final_ub = tpm_ub

            rxn.lower_bound = float(final_lb)
            rxn.upper_bound = float(final_ub)

            summary_rows.append({
                "reaction_id": rxn.id,
                "reaction_name": rxn.name,
                "kegg_id": "",
                "gpr_rule": rxn.gene_reaction_rule or "",
                "DeltaG": None,
                "tpm_value": float(tpm_value),
                "expr_bound": float(expr_bound),
                "final_lower_bound": float(final_lb),
                "final_upper_bound": float(final_ub),
            })
            continue  # ✅ VERY IMPORTANT

        # ---------------------------------------------------------
        # ✅ CASE 2 — BIOMASS (must always stay open)
        # ---------------------------------------------------------
        if rxn.id == "BIOMASS_KT2440_WT3":
            final_lb = 0.0
            final_ub = 1000.0

            rxn.lower_bound = final_lb
            rxn.upper_bound = final_ub

            summary_rows.append({
                "reaction_id": rxn.id,
                "reaction_name": rxn.name,
                "kegg_id": "",
                "gpr_rule": rxn.gene_reaction_rule or "",
                "DeltaG": None,
                "tpm_value": float(tpm_value),
                "expr_bound": float(expr_bound),
                "final_lower_bound": float(final_lb),
                "final_upper_bound": float(final_ub),
            })
            continue  # ✅ VERY IMPORTANT

        # ---------------------------------------------------------
        # ✅ CASE 3 — TPM = 0 → reaction fully off
        # ---------------------------------------------------------
        if expr_bound == 0.0:
            final_lb, final_ub = 0.0, 0.0
            rid = None

            rxn.lower_bound = final_lb
            rxn.upper_bound = final_ub

            summary_rows.append({
                "reaction_id": rxn.id,
                "reaction_name": rxn.name,
                "kegg_id": "",
                "gpr_rule": rxn.gene_reaction_rule or "",
                "DeltaG": None,
                "tpm_value": float(tpm_value),
                "expr_bound": float(expr_bound),
                "final_lower_bound": float(final_lb),
                "final_upper_bound": float(final_ub),
            })
            continue

        # ---------------------------------------------------------
        # ✅ CASE 4 — NORMAL REACTIONS (TPM + ΔG constraints)
        # ---------------------------------------------------------
        rid = None
        if "kegg.reaction" in rxn.annotation:
            rid = rxn.annotation["kegg.reaction"]
            if isinstance(rid, list):
                rid = rid[0]

        if rid:
            dg = dg_map.get(rid, None)
            lb_dg, ub_dg = get_bounds_from_dg_file(rid, dg_map)
        else:
            dg = None
            lb_dg, ub_dg = 0.0, 50.0

        if rxn.reversibility:
            final_lb = max(lb_dg, tpm_lb)
            final_ub = min(ub_dg, tpm_ub)

        elif dg is None or pd.isna(dg):
            final_lb = 0.0
            final_ub = min(ub_dg, (tpm_ub * 0.6))

        else:
            final_lb = 0.0
            final_ub = min(ub_dg, tpm_ub)

        if final_ub < final_lb:
            final_lb, final_ub = -10.0, 10.0

        # ---------------------------------------------------------
        # ✅ APPLY FINAL BOUNDS
        # ---------------------------------------------------------
        rxn.lower_bound = float(final_lb)
        rxn.upper_bound = float(final_ub)

        summary_rows.append({
            "reaction_id": rxn.id,
            "reaction_name": rxn.name,
            "kegg_id": rid or "",
            "gpr_rule": rxn.gene_reaction_rule or "",
            "DeltaG": dg_map.get(rid, None),
            "tpm_value": float(tpm_value),
            "expr_bound": float(expr_bound),
            "final_lower_bound": float(final_lb),
            "final_upper_bound": float(final_ub),
        })

    out_sbml = os.path.join(OUTPUT_DIR, f"iJN1463_{condition}_ExprThermoConstrainedFile.xml")
    cobra.io.write_sbml_model(model, out_sbml)

    out_csv = os.path.join(OUTPUT_DIR, f"reaction_bounds_summary_{condition}.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

    logging.info("Saved model and summary for %s", condition)


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Starting rnaseq+thermo constrain with ΔG file")
    dg_df = pd.read_csv(DG_FILE, sep=";")
    dg_map = dict(zip(dg_df['reaction_id'], dg_df['deltaG']))
    for cond, col in CONDITIONS.items():
        constrain_model(cond, col, dg_map)
    print("[INFO] All conditions processed")
