from cobra import io
adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")
adjusted_model.objective = "BIOMASS_KT2440_WT3"
#print(adjusted_model.optimize().status, adjusted_model.optimize().objective_value)
from cobra.flux_analysis import flux_variability_analysis
# FVA برای بیومس و چند واکنش کلیدی ALA (مثال: list_of_ala_rxns)
targets = ["BIOMASS_KT2440_WT3"] + list_of_ala_rxns
fva = flux_variability_analysis(adjusted_model, reaction_list=targets, fraction_of_optimum=0.9)
print(fva)
