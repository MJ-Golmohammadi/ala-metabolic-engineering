from cobra import io
adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")
adjusted_model.objective = "BIOMASS_KT2440_WT3"
#print(adjusted_model.optimize().status, adjusted_model.optimize().objective_value)
# adjusted_model = مدل خروجی pipeline (همان متغیری که داری)
m = adjusted_model
eps = 1e-6

# باز کردن موقت با حدود معقول
for rxn_id, new_bounds in [('CUt2pp',(-10,1000)),('HVCD',(-1000,1000)),('VCACT',(-1000,1000)),
                           ('PIt7ipp',(0,1000)),('SK_pqqA_kt_c',(-10,1000)),('ATPM',(0,1000))]:
    if rxn_id in m.reactions:
        r = m.reactions.get_by_id(rxn_id)
        r.lower_bound = float(new_bounds[0])
        r.upper_bound = float(new_bounds[1])
        print("opened", rxn_id, "->", r.lower_bound, r.upper_bound)
m.objective = "BIOMASS_KT2440_WT3"
print("status after opening tiny:", m.optimize().status, m.optimize().objective_value)

