#!/usr/bin/env python3
"""
Export all reaction IDs from a COBRA model with their bounds to a CSV file.

Requirements:
    pip install cobra pandas
"""

import cobra
import pandas as pd

# مسیر فایل مدل SBML
MODEL_PATH = "models/iJN1463.xml"   # مسیر مدل خودت را اینجا بگذار
OUTPUT_FILE = "reaction_bounds.csv" # فایل خروجی CSV

def main():
    # بارگذاری مدل
    model = cobra.io.read_sbml_model(MODEL_PATH)
    print(f"✅ Model loaded: {len(model.reactions)} reactions")

    # استخراج اطلاعات واکنش‌ها
    data = []
    for rxn in model.reactions:
        data.append({
            "reaction_id": rxn.id,
            "reaction_name": rxn.name,
            "lower_bound": rxn.lower_bound,
            "upper_bound": rxn.upper_bound
        })

    # ذخیره در CSV
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"💾 Reaction list with bounds saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
