from cobra import io
adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")
adjusted_model.objective = "BIOMASS_KT2440_WT3"
#print(adjusted_model.optimize().status, adjusted_model.optimize().objective_value)
def incremental_restore_until_feasible(scaled_model, original_bounds, rxn_scores, biomass_rxn_id="BIOMASS_KT2440_WT3", batch=10):
    sorted_rxns = sorted(rxn_scores.items(), key=lambda x: x[1], reverse=True)
    restored = []
    # اگر از ابتدا feasible است، برگردان
    scaled_model.objective = biomass_rxn_id
    sol = scaled_model.optimize()
    if sol.status == 'optimal' and float(sol.fluxes.get(biomass_rxn_id,0.0)) > 1e-8:
        return scaled_model, restored
    for i in range(0, len(sorted_rxns), batch):
        for rxn_id, _ in sorted_rxns[i:i+batch]:
            if rxn_id in scaled_model.reactions and rxn_id in original_bounds:
                lb, ub = original_bounds[rxn_id]
                r = scaled_model.reactions.get_by_id(rxn_id)
                r.lower_bound = float(lb)
                r.upper_bound = float(ub)
                restored.append(rxn_id)
        scaled_model.objective = biomass_rxn_id
        sol = scaled_model.optimize()
        if sol.status == 'optimal' and float(sol.fluxes.get(biomass_rxn_id,0.0)) > 1e-8:
            print("Feasible after restoring", len(restored), "reactions")
            return scaled_model, restored
    print("Rollback finished but still infeasible")
    return scaled_model, restored

# فراخوانی (در صورتی که original_bounds و rxn_scores را داری)
scaled_model, restored = incremental_restore_until_feasible(adjusted_model, original_bounds, rxn_scores, "BIOMASS_KT2440_WT3", batch=10)
print("restored count:", len(restored), "examples:", restored[:30])



