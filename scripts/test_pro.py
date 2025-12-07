# find_essential_reactions.py
import cobra
import pandas as pd
import numpy as np
from cobra.flux_analysis import single_reaction_deletion, single_gene_deletion, find_blocked_reactions

# ---------- تنظیمات کاربر ----------
MODEL_PATH = "models/iJN1463.xml"  # مسیر مدل constrained تو
BIOMASS_RXN = "BIOMASS_KT2440_WT3"
GROWTH_DROP_THRESHOLD = 0.01   # اگر رشد بعد حذف < 1% از رشد WT => واکنش ضروری در نظر گرفته شود
TOP_N = 50                     # برای خروجی پیشنهادی
OUTPUT_PREFIX = "essential_analysis"
# ------------------------------------

# بارگذاری مدل و محاسبه رشد پایه (wild-type constrained)
model = cobra.io.read_sbml_model(MODEL_PATH)
model.objective = BIOMASS_RXN
wt_sol = model.optimize()
wt_status = wt_sol.status
wt_growth = wt_sol.objective_value if wt_status == 'optimal' else 0.0
print("WT status:", wt_status, "WT growth:", wt_growth)

# اگر مدل infeasible یا growth≈0 است، هشدار بده و متوقف کن (نیاز به relax اولیه)
if wt_status != 'optimal' or wt_growth <= 1e-9:
    print("Warning: WT model infeasible or growth≈0. Run preliminary relaxations (exchanges, ATPM, floors) before essentiality analysis.")
else:
    # 1) حذف تک-واکنش (fast, vectorized)
    print("Running single_reaction_deletion (this may take time for large models)...")
    rxn_del_df = single_reaction_deletion(model, model.reactions.list_attr('id'), method='fba')
    # rxn_del_df has column 'growth' with biomass after deletion
    rxn_del_df = rxn_del_df.reset_index().rename(columns={'index':'reaction_id','growth':'growth_after_deletion'})
    rxn_del_df['wt_growth'] = wt_growth
    rxn_del_df['growth_fraction'] = rxn_del_df['growth_after_deletion'] / wt_growth
    rxn_del_df['growth_drop_pct'] = (1.0 - rxn_del_df['growth_fraction']) * 100.0
    # Mark essential by threshold or infeasible
    rxn_del_df['status'] = rxn_del_df['growth_after_deletion'].apply(lambda x: 'infeasible' if pd.isna(x) else 'ok')
    rxn_del_df['is_essential'] = rxn_del_df['growth_fraction'] <= GROWTH_DROP_THRESHOLD

    # 2) مرتب‌سازی و ذخیره نتایج کامل
    rxn_del_df = rxn_del_df.sort_values(['is_essential','growth_fraction'], ascending=[False, True])
    rxn_del_df.to_csv(f"{OUTPUT_PREFIX}_single_reaction_deletion.csv", index=False)
    print("Wrote:", f"{OUTPUT_PREFIX}_single_reaction_deletion.csv")

    # 3) خلاصه واکنش‌های ضروری
    essentials = rxn_del_df[rxn_del_df['is_essential']].copy()
    print("Number of essential reactions (threshold {:.2%}): {}".format(GROWTH_DROP_THRESHOLD, len(essentials)))
    essentials.to_csv(f"{OUTPUT_PREFIX}_essential_reactions.csv", index=False)

    # 4) پیشنهادات هدفمند برای relax (اولویت: انرژی، گلیکولیز، TCA، exchange)
    # بارگذاری metadata از مدل برای نام و reversibility
    meta = []
    for _, row in essentials.head(TOP_N).iterrows():
        rid = row['reaction_id']
        if rid in model.reactions:
            r = model.reactions.get_by_id(rid)
            meta.append({
                'reaction_id': rid,
                'name': r.name,
                'reversibility': r.reversibility,
                'current_lb': r.lower_bound,
                'current_ub': r.upper_bound,
                'growth_after_deletion': row['growth_after_deletion'],
                'growth_fraction': row['growth_fraction'],
                'suggested_relax_ub': max(1.0, abs(r.upper_bound)*2.0) if abs(r.upper_bound) > 0 else 2.0,
                'suggested_relax_lb': -max(1.0, abs(r.upper_bound)*2.0) if r.reversibility else r.lower_bound,
                'reason': 'essential_by_deletion_test'
            })
    sugg_df = pd.DataFrame(meta)
    sugg_df.to_csv(f"{OUTPUT_PREFIX}_relaxation_suggestions_top{TOP_N}.csv", index=False)
    print("Wrote relaxation suggestions:", f"{OUTPUT_PREFIX}_relaxation_suggestions_top{TOP_N}.csv")

    # 5) Optional: run single_gene_deletion to map gene-level essentials (if GPR present)
    try:
        print("Running single_gene_deletion (optional, may be slower)...")
        gene_del = single_gene_deletion(model, model.genes.list_attr('id'))
        gene_del = gene_del.reset_index().rename(columns={'index':'gene_id','growth':'growth_after_deletion'})
        gene_del['growth_fraction'] = gene_del['growth_after_deletion'] / wt_growth
        gene_del['is_essential'] = gene_del['growth_fraction'] <= GROWTH_DROP_THRESHOLD
        gene_del.to_csv(f"{OUTPUT_PREFIX}_single_gene_deletion.csv", index=False)
        print("Wrote gene deletion results.")
    except Exception as e:
        print("single_gene_deletion failed or not applicable:", e)

    # 6) گزارش نهایی خلاصه
    print("Done. Key outputs:")
    print(" -", f"{OUTPUT_PREFIX}_single_reaction_deletion.csv")
    print(" -", f"{OUTPUT_PREFIX}_essential_reactions.csv")
    print(" -", f"{OUTPUT_PREFIX}_relaxation_suggestions_top{TOP_N}.csv")
    print(" - (optional) {OUTPUT_PREFIX}_single_gene_deletion.csv")
