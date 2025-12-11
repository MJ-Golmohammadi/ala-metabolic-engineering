from cobra import io
adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")
adjusted_model.objective = "BIOMASS_KT2440_WT3"
#print(adjusted_model.optimize().status, adjusted_model.optimize().objective_value)
# adjusted_model = مدل خروجی pipeline (همان متغیری که داری)
m = adjusted_model
eps = 1e-6

tiny = []
for r in m.reactions:
    lb = float(r.lower_bound)
    ub = float(r.upper_bound)
    if abs(ub - lb) <= 1e-6 or abs(ub) <= 1e-6 or abs(lb) <= 1e-6 and ub <= 1e-3:
        tiny.append((r.id, lb, ub))
print("tiny/equality reactions count:", len(tiny))
print("sample tiny reactions:", tiny[:80])

