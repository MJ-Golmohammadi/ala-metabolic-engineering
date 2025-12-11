from cobra import io
adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")
adjusted_model.objective = "BIOMASS_KT2440_WT3"
#print(adjusted_model.optimize().status, adjusted_model.optimize().objective_value)
exs = [(r.id, float(r.lower_bound), float(r.upper_bound)) for r in adjusted_model.reactions if r.id.startswith("EX_")]
print("EX bounds:", exs)
for rname in ["ATPM", "ATPS4r", "ATPS", "BIOMASS_KT2440_WT3"]:
    if rname in adjusted_model.reactions:
        r = adjusted_model.reactions.get_by_id(rname)
        print(rname, float(r.lower_bound), float(r.upper_bound))


