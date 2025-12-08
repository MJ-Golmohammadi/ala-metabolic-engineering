#!/usr/bin/env python3
"""
Condition-specific GEM construction using RNA-seq CSV + local ΔG file.

This script constrains a base genome-scale metabolic model (GEM) per condition
using RNA-seq expression data and precomputed reaction ΔG values. Essential
reactions and their optional bounds are read from an external CSV file so that
the list of essential reactions is not hard-coded.

Requirements:
    pip install cobra pandas

Inputs:
    - MODEL_PATH: SBML model file
    - EXPR_FILE: semicolon-separated expression CSV (Synonym;Expression_Glu;...)
    - DG_FILE: semicolon-separated ΔG CSV (reaction_id;deltaG)
    - ESSENTIAL_CSV: CSV with column `ids` containing reaction ids (single id,
      Python set literal like "{'R1','R2'}", or delimited list). Optional
      columns: lb, ub to specify bounds for listed reactions.

Outputs:
    - Constrained SBML models per condition in OUTPUT_DIR
    - reaction_bounds_summary_{condition}.csv per condition
"""

import os
import re
import logging
import cobra
import pandas as pd
import math
import ast

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = "models/iJN1463.xml"
EXPR_FILE = "expression_txt_files/merged_expression.csv"     # semicolon-separated CSV
DG_FILE = "models/kegg_reactions_CC_ph7.0.csv"               # CSV with reaction_id, deltaG
ESSENTIAL_CSV = "config/essential_rxns.csv"                 # CSV listing essential reaction ids and optional bounds
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
GENERAL_SCALE = 10
DG_SCALE1 = 100      # scaling factor for bounds
DG_SCALE2 = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# Helper functions for parsing and loading essential reactions CSV
# -----------------------------------------------------------------------------
def parse_ids_field(val):
    """
    Parse an `ids` field from the CSV. Supported formats:
      - Single id: "ATPM"
      - Python set/list literal: "{'ATPM','PDH'}" or "['ATPM','PDH']"
      - Delimited list: "ATPM;PDH" or "ATPM,PDH"
    Returns a list of cleaned reaction ids.
    """
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s == "":
        return []

    # Try to parse Python literal (set/list/tuple/dict)
    try:
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                return [str(k).strip() for k in parsed.keys()]
            if isinstance(parsed, (set, list, tuple)):
                return [str(x).strip() for x in parsed]
    except Exception:
        # Fall through to delimiter parsing
        pass

    # Delimiter-based parsing
    if ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
        return parts
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts

    # Single token
    return [s]


def load_essential_map(csv_path):
    """
    Load essential reactions and optional bounds from CSV.

    Returns:
        dict: { reaction_id: (lb_or_None, ub_or_None) }
    If CSV is missing or empty, returns an empty dict.
    """
    essential_map = {}
    if not os.path.exists(csv_path):
        logging.warning("Essential CSV not found: %s. No essential reactions will be forced.", csv_path)
        return essential_map

    try:
        df_e = pd.read_csv(csv_path, sep=";")
    except Exception as e:
        logging.warning("Failed to read essential CSV %s: %s", csv_path, e)
        return essential_map

    if "ids" not in df_e.columns:
        logging.warning("Essential CSV does not contain 'ids' column. No essential reactions loaded.")
        return essential_map

    # Determine possible column names for lb/ub
    lb_cols = [c for c in ("lb", "lower_bound", "final_lower_bound") if c in df_e.columns]
    ub_cols = [c for c in ("ub", "upper_bound", "final_upper_bound") if c in df_e.columns]

    for _, row in df_e.iterrows():
        ids_field = row.get("ids", None)
        ids = parse_ids_field(ids_field)
        # Read lb/ub if present
        lb = None
        ub = None
        if lb_cols:
            try:
                val = row[lb_cols[0]]
                lb = float(val) if not pd.isna(val) else None
            except Exception:
                lb = None
        if ub_cols:
            try:
                val = row[ub_cols[0]]
                ub = float(val) if not pd.isna(val) else None
            except Exception:
                ub = None
        for rid in ids:
            if rid:
                essential_map[str(rid).strip()] = (lb, ub)
    logging.info("Loaded %d essential reaction entries from %s", len(essential_map), csv_path)
    return essential_map

# -----------------------------------------------------------------------------
# Thermodynamic helper
# -----------------------------------------------------------------------------
def get_bounds_from_dg_file(rxn_id: str, dg_map: dict):
    """
    Determine ΔG-based bounds for a reaction id using heuristic scaling.
    Returns (lb, ub).
    """
    dg = dg_map.get(rxn_id, None)
    if dg is None or pd.isna(dg):
        return 0.0, FLUX_CAP
    dg = float(dg)
    if dg < -5.0:
        ub = min(FLUX_CAP, int(abs(dg) * DG_SCALE1 * GENERAL_SCALE))
        return 0.0, float(ub)
    elif -5.0 <= dg <= -1.0:
        ub = int(abs(dg) * DG_SCALE2 * GENERAL_SCALE)
        return 0.0, float(ub)
    elif -1.0 < dg < 1.0:
        ub = 50
        return 0.0, float(ub)
    elif 1.0 < dg <= 20.0:
        ub = int((50 / math.sqrt(abs(dg))) * GENERAL_SCALE)
        return 0.0, float(ub)
    else:
        ub = 100.0
        lb = -100.0
        return float(lb), float(ub)

# -----------------------------------------------------------------------------
# Expression and gene-rule helpers
# -----------------------------------------------------------------------------
def get_flux_bound_from_tpm(tpm: float) -> float:
    """
    Convert TPM-like expression value to a flux bound (heuristic).
    Zero or negative TPM yields zero bound.
    """
    try:
        val = float(tpm)
    except Exception:
        return 0.0
    if val <= 0.0:
        return 0.0
    return float(min(FLUX_CAP, val))


def evaluate_gene_rule(rule: str, expr_map: dict) -> float:
    """
    Evaluate a gene-reaction rule string against an expression map.
    Supports 'and'/'or' logic and parentheses. Returns a numeric TPM-derived value.
    """
    def _eval_recursive(s: str) -> float:
        if s is None:
            return 0.0
        s = str(s).strip()
        if s == "" or s.lower() in {"none", "nan"}:
            return 0.0
        # Resolve parentheses first
        while True:
            m = re.search(r"\(([^()]+)\)", s)
            if not m:
                break
            inner_val = _eval_recursive(m.group(1))
            s = s[:m.start()] + str(inner_val) + s[m.end():]
        # OR logic
        parts_or = [p.strip() for p in re.split(r"\s+or\s+", s, flags=re.IGNORECASE)]
        if len(parts_or) > 1:
            return max(_eval_recursive(p) for p in parts_or)
        # AND logic
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

# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------
def constrain_model(condition: str, expr_column: str, dg_map: dict, essential_map: dict):
    """
    Constrain the base model for a given condition using expression and ΔG maps.
    Essential reactions are enforced according to `essential_map`.
    """
    logging.info("Condition: %s | Expression column: %s", condition, expr_column)

    # Load expression CSV and build expression map (mean per Synonym)
    df_expr = pd.read_csv(EXPR_FILE, sep=CSV_SEPARATOR)
    grouped = df_expr.groupby("Synonym")[expr_column].mean().dropna()
    expr_map = {str(k).lower(): float(v) for k, v in grouped.to_dict().items()}

    # Load base model
    model = cobra.io.read_sbml_model(MODEL_PATH)
    logging.info("Model loaded: %d reactions", len(model.reactions))

    summary_rows = []
    EPS = 1e-5

    for rxn in model.reactions:

        # -----------------------------------------------------------------
        # (A) If reaction is listed as essential in CSV, apply provided bounds
        # -----------------------------------------------------------------
        if rxn.id in essential_map:
            csv_lb, csv_ub = essential_map.get(rxn.id, (None, None))
            if csv_lb is not None or csv_ub is not None:
                final_lb = float(csv_lb) if csv_lb is not None else (-1000.0 * GENERAL_SCALE if rxn.reversibility else 0.0)
                final_ub = float(csv_ub) if csv_ub is not None else (1000.0 * GENERAL_SCALE)
            else:
                final_lb = -1000.0 * GENERAL_SCALE if rxn.reversibility else 0.0
                final_ub = 1000.0 * GENERAL_SCALE

            rxn.lower_bound = float(final_lb)
            rxn.upper_bound = float(final_ub)

            summary_rows.append({
                "reaction_id": rxn.id,
                "reaction_name": rxn.name,
                "kegg_id": "",
                "gpr_rule": rxn.gene_reaction_rule or "",
                "DeltaG": None,
                "tpm_value": 0.0,
                "expr_bound": 0.0,
                "final_lower_bound": float(final_lb),
                "final_upper_bound": float(final_ub),
            })
            continue

        # ---------------------------------------------------------
        # (B) Compute TPM-derived expression bound
        # ---------------------------------------------------------
        tpm_value = evaluate_gene_rule(rxn.gene_reaction_rule, expr_map)
        expr_bound = get_flux_bound_from_tpm(tpm_value)
        tpm_lb, tpm_ub = -expr_bound, expr_bound

        # ---------------------------------------------------------
        # (C) Exchange reactions: do not apply thermodynamic or TPM shutdown
        #     These reactions define environmental availability and must remain open.
        # ---------------------------------------------------------
        if rxn.id.startswith("EX_"):
            # Keep a small epsilon to avoid strict zero blocking in some solvers
            final_lb = -EPS
            final_ub = 999_999
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
            continue

        # ---------------------------------------------------------
        # (D) Biomass reaction: keep fully open for growth simulation
        # ---------------------------------------------------------
        if rxn.id == "BIOMASS_KT2440_WT3":
            final_lb = 0.0
            final_ub = 999_999
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
        # (E) TPM == 0 -> apply soft-floor constraints to avoid pathway collapse
        # ---------------------------------------------------------
        if expr_bound == 0.0:
            if rxn.reversibility:
                final_lb = -100 * GENERAL_SCALE
                final_ub = 100 * GENERAL_SCALE
            else:
                final_lb = 0.0
                final_ub = 1000.0

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
            continue

        # ---------------------------------------------------------
        # (F) Apply thermodynamic ΔG constraints for reactions with KEGG IDs
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
            lb_dg, ub_dg = 0.0, 50.0 * GENERAL_SCALE

        # ---------------------------------------------------------
        # (G) Combine TPM and ΔG constraints
        # ---------------------------------------------------------
        if rxn.reversibility:
            final_lb = max(lb_dg, tpm_lb * GENERAL_SCALE)
            final_ub = min(ub_dg, tpm_ub * GENERAL_SCALE)
        elif dg is None or pd.isna(dg):
            final_lb = 0.0
            final_ub = min(ub_dg, (tpm_ub * 0.6 * GENERAL_SCALE))
        else:
            final_lb = 0.0
            final_ub = min(ub_dg, tpm_ub * GENERAL_SCALE)

        # Safety check: ensure feasible bounds
        if final_ub < final_lb:
            final_lb, final_ub = -100.0, 100.0

        # ---------------------------------------------------------
        # (H) Apply final bounds and record summary
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

    # Save constrained model and summary table
    out_sbml = os.path.join(OUTPUT_DIR, f"iJN1463_{condition}_ExprThermoConstrainedFile.xml")
    cobra.io.write_sbml_model(model, out_sbml)

    out_csv = os.path.join(OUTPUT_DIR, f"reaction_bounds_summary_{condition}.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

    logging.info("Saved model and summary for %s", condition)


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Starting rnaseq+thermo constrain with ΔG file and external essential list")

    # Load ΔG map
    try:
        dg_df = pd.read_csv(DG_FILE, sep=";")
        dg_map = dict(zip(dg_df['reaction_id'], dg_df['deltaG']))
    except Exception as e:
        logging.warning("Failed to load ΔG file %s: %s. Proceeding with empty ΔG map.", DG_FILE, e)
        dg_map = {}

    # Load essential reactions map from CSV (may be empty)
    essential_map = load_essential_map(ESSENTIAL_CSV)

    # Process each condition
    for cond, col in CONDITIONS.items():
        constrain_model(cond, col, dg_map, essential_map)

    print("[INFO] All conditions processed")
