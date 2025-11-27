import cobra

model = cobra.io.read_sbml_model(
    "models/final_constrained_rnaseq_thermo/iJN1463_Glu_ExprThermoConstrainedFile.xml"
)

dead = []
for rxn in model.reactions:
    if rxn.lower_bound == 0 and rxn.upper_bound == 0:
        dead.append(rxn.id)

print("Number of dead reactions:", len(dead))
print(dead[:500])  
