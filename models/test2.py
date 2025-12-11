m = cobra.io.read_sbml_model("models/iJN1463.xml")
m.objective = "BIOMASS_KT2440_WT3"
sol = m.optimize()
print(sol.status, sol.objective_value)
