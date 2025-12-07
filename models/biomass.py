import cobra
from cobra.flux_analysis import find_blocked_reactions, flux_variability_analysis

MODEL_PATH = "models/final_constrained_rnaseq_thermo/iJN1463_Cit_ExprThermoConstrainedFile.xml"
BIOMASS_ID = "BIOMASS_KT2440_WT3"   # اگر id متفاوت است، بعدا لیست می‌کنیم
model = cobra.io.read_sbml_model(MODEL_PATH)

# 1) آیا واکنش بیومس وجود دارد و bounds آن چیست؟
print("BIOMASS present:", BIOMASS_ID in model.reactions)
if BIOMASS_ID in model.reactions:
    r = model.reactions.get_by_id(BIOMASS_ID)
    print("BIOMASS bounds:", r.lower_bound, r.upper_bound)

# 2) وضعیت exchangeهای کلیدی (برای محیط Glu مثال)
for rxn in ["EX_glc__D_e","EX_o2_e","EX_nh4_e","EX_pi_e","EX_so4_e","ATPM","ATPS4r"]:
    if rxn in model.reactions:
        r = model.reactions.get_by_id(rxn)
        print(r.id, "lb=", r.lower_bound, "ub=", r.upper_bound)
    else:
        print(rxn, "NOT FOUND")

# 3) یک optimize ساده با objective بیومس و چاپ status
model.objective = BIOMASS_ID
sol = model.optimize()
print("opt status:", sol.status, "biomass:", sol.objective_value)

# 4) واکنش‌های کاملاً بسته (blocked)
blocked = find_blocked_reactions(model)
print("num blocked reactions:", len(blocked))
print(blocked[:50])

# 5) FVA برای دیدن آیا بیومس می‌تواند >0 باشد در حالت relaxed (اگر feasible)
try:
    fva = flux_variability_analysis(model, fraction_of_optimum=0.9)
    print("FVA done; sample rows:")
    print(fva.loc[[BIOMASS_ID]] if BIOMASS_ID in fva.index else fva.head())
except Exception as e:
    print("FVA error:", e)
