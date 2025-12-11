from cobra import io
adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")
adjusted_model.objective = "BIOMASS_KT2440_WT3"
#print(adjusted_model.optimize().status, adjusted_model.optimize().objective_value)
# adjusted_model = مدل خروجی pipeline (همان متغیری که داری)
m = adjusted_model
eps = 1e-6

# لیست کامل tiny/equality
tiny = [(r.id, float(r.lower_bound), float(r.upper_bound)) 
        for r in m.reactions if abs(float(r.upper_bound)-float(r.lower_bound)) <= 1e-6]
print("tiny count:", len(tiny))
print(tiny[:100])

# باز کردن 20 واکنش اول tiny به بازه معقول
for rxn_id, lb, ub in tiny[:20]:
    r = m.reactions.get_by_id(rxn_id)
    r.lower_bound = -1000.0 if lb < 0 else 0.0
    r.upper_bound = 1000.0
m.objective = "BIOMASS_KT2440_WT3"
print("status after opening tiny batch:", m.optimize().status, m.optimize().objective_value)

