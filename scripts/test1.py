from scripts/1_baseline_simulation.py import *
with model:
    model.objective = "BIOMASS_KT2440_WT3"
    sol = model.optimize()
    print(sol.status, sol.objective_value)
