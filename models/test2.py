from cobra import io
adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")
adjusted_model.objective = "BIOMASS_KT2440_WT3"
#print(adjusted_model.optimize().status, adjusted_model.optimize().objective_value)
closed = [r.id for r in adjusted_model.reactions if float(r.lower_bound)==0.0 and float(r.upper_bound)==0.0]
tiny = [r.id for r in adjusted_model.reactions if abs(float(r.upper_bound)) <= 1e-6 and abs(float(r.lower_bound)) <= 1e-6]
print("closed count:", len(closed))
print("closed sample:", closed[:50])
print("tiny capacity count:", len(tiny))
print("tiny sample:", tiny[:50])

