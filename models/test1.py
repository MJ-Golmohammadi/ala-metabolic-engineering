import cobra

model = cobra.io.read_sbml_model(
    "models/final_constrained_rnaseq_thermo/iJN1463_Glu_ExprThermoConstrainedFile.xml"
)

print("Objective before:", model.objective)

model.objective = "BIOMASS_KT2440_WT3"
solution = model.optimize()

print("Growth:", solution.objective_value)
