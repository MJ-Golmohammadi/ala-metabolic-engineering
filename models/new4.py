#!/usr/bin/env python3
"""
Condition-specific GEM construction using RNA-seq CSV + local ΔG file.

Modifications requested:
- Do NOT globally replace reaction bounds for all reactions.
  * For reactions not targeted by engineering, preserve the model's original
    lower/upper bounds but cap them to +/-6000 (upper cap = +6000, lower cap = -6000).
- Apply the RNA-derived and ΔG-derived bound adjustments only to a specific
  list of reactions (ALA production targets) provided in a CSV:
    ala_production_rxns.csv
  This CSV uses the same 'ids' column format as the other config CSVs.
- All CSV parsing uses CSV_SEPARATOR.
- `parse_ids_field` robustly parses inputs like "{'OCBT'}" -> "OCBT".
- English comments and logging added for clarity.
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
DG_FILE = "models/kegg_reactions_CC_ph7.0.csv"               # CSV with reaction_id;deltaG
ALA_RXNS_CSV = "config/ala_production_rxns.csv"             # CSV listing reactions to apply engineering to
CSV_SEPARATOR = ";"                                         # separator used in CSV files
OUTPUT_DIR = "models/final_constrained_rnaseq_thermo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDITIONS = {
    "Glu": "Expression_Glu",
    "Cit": "Expression_Cit",
    "Fer": "Expression_Fer",
    "Ser": "Expression_Ser",
}

# Global caps requested by user
UPPER_CAP = 6000.0
LOWER_CAP = -6000.0

FLUX_CAP = 6000
GENERAL_SCALE = 10
DG_SCALE1 = 100      # scaling factor for bounds
DG_SCALE2 = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# Helper functions for parsing and loading CSV lists
# -----------------------------------------------------------------------------
def parse_ids_field(val):
    """
    Parse an `ids` field from a CSV cell and return a list of cleaned reaction ids.

    Supported input formats:
      - Single id: "ATPM"
      - Python set/list literal: "{'ATPM','PDH'}" or "['ATPM','PDH']"
      - Delimited list: "ATPM;PDH" or "ATPM,PDH"
      - Bare braces with single token: "{OCBT}" or "{'OCBT'}"

    Returns:
        list of strings (each stripped of whitespace and quotes)
    """
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s == "":
        return []

    # Try to parse Python literal safely
    try:
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                return [str(k).strip() for k in parsed.keys()]
            if isinstance(parsed, (set, list, tuple)):
                return [str(x).strip() for x in parsed]
            return [str(parsed).strip()]
    except Exception:
        # Fall through to delimiter parsing
        pass

    # Delimiter-based parsing
    if ";" in s:
        parts = [p.strip().strip("'\"") for p in s.split(";") if p.strip()]
        return parts
    if "," in s:
        parts = [p.strip().strip("'\"") for p in s.split(",") if p.strip()]
        return parts

    # Remove surrounding braces if present (e.g., "{OCBT}" or "{'OCBT'}")
    m = re.match(r"^\{(.*)\}$", s)
    if m:
        inner = m.group(1).strip()
        if "," in inner or ";" in inner:
            sep = "," if "," in inner else ";"
            parts = [p.strip().strip("'\"") for p in inner.split(sep) if p.strip()]
            return parts
        return [inner.strip().strip("'\"")]

    # Single token fallback
    return [s.strip().strip("'\"")]


def load_essential_map(csv_path):
    """
    Load essential reactions and optional bounds from CSV.

    Returns:
        dict: { reaction_id: (lb_or_None, ub_or_None) }

    The CSV must contain a column named 'ids'. Optional columns for bounds:
    - lower bound: one of ('lb', 'lower_bound', 'final_lower_bound')
    - upper bound: one of ('ub', 'upper_bound', 'final_upper_bound')
    """
    essential_map = {}
    if not os.path.exists(csv_path):
        logging.warning("Essential CSV not found: %s. No essential reactions will be forced.", csv_path)
        return essential_map

    try:
        df_e = pd.read_csv(csv_path, sep=CSV_SEPARATOR)
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


def load_ala_set(csv_path):
    """
    Load a set of reaction ids from ALA_RXNS_CSV that should receive the
    engineered RNA/thermo bound adjustments. The CSV must contain an 'ids' column.
    Returns a set of reaction ids (strings).
    """
    ala_set = set()
    if not os.path.exists(csv_path):
        logging.info("ALA reactions CSV not found: %s. No engineering targets loaded.", csv_path)
        return ala_set

    try:
        df_a = pd.read_csv(csv_path, sep=CSV_SEPARATOR)
    except Exception as e:
        logging.warning("Failed to read ALA CSV %s: %s", csv_path, e)
        return ala_set

    if "ids" not in df_a.columns:
        logging.warning("ALA CSV does not contain 'ids' column. No ALA reactions loaded.")
        return ala_set

    for _, row in df_a.iterrows():
        ids_field = row.get("ids", None)
        ids = parse_ids_field(ids_field)
        for rid in ids:
            if rid:
                ala_set.add(str(rid).strip())
    logging.info("Loaded %d ALA engineering target ids from %s", len(ala_set), csv_path)
    return ala_set

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
def cap_bounds(lb: float, ub: float) -> (float, float):
    """
    Cap provided bounds to the global LOWER_CAP and UPPER_CAP.
    Ensures lower <= upper after capping.
    """
    capped_lb = max(lb, LOWER_CAP)
    capped_ub = min(ub, UPPER_CAP)
    if capped_ub < capped_lb:
        # If capping produced an infeasible interval, fall back to wide but capped interval
        capped_lb, capped_ub = LOWER_CAP, UPPER_CAP
    return float(capped_lb), float(capped_ub)


def constrain_model(condition: str, expr_column: str, dg_map: dict, essential_map: dict, ala_set: set):
    """
    Constrain the base model for a given condition using expression and ΔG maps.

    Behavior summary:
    - If reaction is in essential_map: force bounds from CSV (same as before).
    - Else if reaction is in ala_set: apply the engineered RNA/ΔG logic (TPM + ΔG).
    - Else: preserve original model bounds but cap them to +/-6000.
    - Exchange reactions and biomass are handled explicitly (kept open but capped).
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

    for rxn in model.reactions:

        # Save original model bounds to preserve if not targeted
        orig_lb = float(rxn.lower_bound)
        orig_ub = float(rxn.upper_bound)

        # -----------------------------------------------------------------
        # (2) If reaction is in ALA engineering set -> apply RNA/thermo logic
        # -----------------------------------------------------------------
        if rxn.id in ala_set:
            # Compute TPM-derived expression bound
            tpm_value = evaluate_gene_rule(rxn.gene_reaction_rule, expr_map)
            expr_bound = get_flux_bound_from_tpm(tpm_value)
            tpm_lb, tpm_ub = -expr_bound, expr_bound

            # Exchange reactions: keep open but cap
            if rxn.id.startswith("EX_"):
                final_lb = -5
                final_ub = UPPER_CAP
                final_lb, final_ub = cap_bounds(final_lb, final_ub)
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
                    "note": "ala_target_exchange"
                })
                continue

            # Biomass reaction: keep open but capped
            if rxn.id == "BIOMASS_KT2440_WT3":
                final_lb = 0.0
                final_ub = UPPER_CAP
                final_lb, final_ub = cap_bounds(final_lb, final_ub)
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
                    "note": "biomass_capped"
                })
                continue

            # If TPM == 0 -> apply soft-floor constraints (only for ALA targets)
            if expr_bound == 0.0:
                if rxn.reversibility:
                    final_lb = -100 * GENERAL_SCALE
                    final_ub = 100 * GENERAL_SCALE
                else:
                    final_lb = 0.0
                    final_ub = 1000.0
                final_lb, final_ub = cap_bounds(final_lb, final_ub)
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
                    "note": "ala_target_soft_floor"
                })
                continue

            # Thermodynamic ΔG constraints (if KEGG annotation present)
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

            # Combine TPM and ΔG constraints (only for ALA targets)
            if rxn.reversibility:
                final_lb = max(lb_dg, tpm_lb * GENERAL_SCALE)
                final_ub = min(ub_dg, tpm_ub * GENERAL_SCALE)
            elif dg is None or pd.isna(dg):
                final_lb = 0.0
                final_ub = min(ub_dg, (tpm_ub * 0.6 * GENERAL_SCALE))
            else:
                final_lb = 0.0
                final_ub = min(ub_dg, tpm_ub * GENERAL_SCALE)

            # Safety check and cap
            if final_ub < final_lb:
                final_lb, final_ub = -100.0, 100.0
            final_lb, final_ub = cap_bounds(final_lb, final_ub)

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
                "note": "ala_target_engineered"
            })
            continue

        # -----------------------------------------------------------------
        # (3) Default: preserve original bounds but cap them to +/-6000
        # -----------------------------------------------------------------
        final_lb, final_ub = cap_bounds(orig_lb, orig_ub)
        rxn.lower_bound = float(final_lb)
        rxn.upper_bound = float(final_ub)

        # For reporting, compute TPM and DeltaG info for completeness (optional)
        tpm_value = evaluate_gene_rule(rxn.gene_reaction_rule, {})  # empty map; not used for default
        summary_rows.append({
            "reaction_id": rxn.id,
            "reaction_name": rxn.name,
            "kegg_id": "",
            "gpr_rule": rxn.gene_reaction_rule or "",
            "DeltaG": None,
            "tpm_value": float(0.0),
            "expr_bound": float(0.0),
            "final_lower_bound": float(final_lb),
            "final_upper_bound": float(final_ub),
            "note": "preserved_capped"
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
    print("[INFO] Starting rnaseq+thermo constrain with ALA-targeted engineering")

    # Load ΔG map
    try:
        dg_df = pd.read_csv(DG_FILE, sep=CSV_SEPARATOR)
        dg_map = dict(zip(dg_df['reaction_id'], dg_df['deltaG']))
    except Exception as e:
        logging.warning("Failed to load ΔG file %s: %s. Proceeding with empty ΔG map.", DG_FILE, e)
        dg_map = {}


    # Load ALA engineering target set (only these reactions will receive RNA/thermo changes)
    ala_set = load_ala_set(ALA_RXNS_CSV)

    # Process each condition
    for cond, col in CONDITIONS.items():
        constrain_model(cond, col, dg_map, essential_map, ala_set)

    print("[INFO] All conditions processed")
