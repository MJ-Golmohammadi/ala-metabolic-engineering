from cobra import io
adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")
adjusted_model.objective = "BIOMASS_KT2440_WT3"
#print(adjusted_model.optimize().status, adjusted_model.optimize().objective_value)
sol = adjusted_model.optimize()
print("status:", sol.status)
print("objective_value:", sol.objective_value)
print("solver message:", sol.message if hasattr(sol, 'message') else None)
