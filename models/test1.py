#!/usr/bin/env python3
"""
Export reaction IDs, names, and bounds from a COBRA SBML model to CSV.
"""

import cobra
import pandas as pd

# مسیر فایل مدل SBML ساخته‌شده‌ات
MODEL_PATH = "models/final_constrained_rnaseq_thermo/iJN1463_Glu_ExprThermoConstrainedFile.xml"

# مسیر خروجی CSV
OUTPUT_CSV = "reaction_bounds_export.csv"

def export_reaction_bounds(model_path, output_csv):
    # بارگذاری مدل
    try:
        model = cobra.io.read_sbml_model(model_path)
        print(f"[INFO] Model loaded: {len(model.reactions)} reactions")
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        return

    # استخراج اطلاعات واکنش‌ها
    rows = []
    for rxn in model.reactions:
        rows.append({
            "reaction_id": rxn.id,
            "reaction_name": rxn.name,
            "lower_bound": rxn.lower_bound,
            "upper_bound": rxn.upper_bound
        })

    # ذخیره به CSV
    try:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Reaction bounds exported to {output_csv}")
    except Exception as exc:
        print(f"[ERROR] Failed to save CSV: {exc}")

if __name__ == "__main__":
    export_reaction_bounds(MODEL_PATH, OUTPUT_CSV)
