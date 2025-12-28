#!/usr/bin/env python3
# scripts/2_metabolic_engineering.py
"""
Main engineering pipeline (final version).

- Reads expression CSV (expression_txt_files/expression_by_reaction.csv)
- Computes scaling factors per environment
- Applies wild-secondary baseline locks on base models
- Builds engineered strains using create_engineered_strain
- Simulates objectives with simulate_with_objective (returns sol, min_growth_abs)
- Computes yields with calculate_yield_metrics
- Saves consolidated CSV for all environments
"""

import os
import sys
import yaml
import cobra
import pandas as pd
from pathlib import Path
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_utils import (
    read_expression_scaling,
    apply_wild_secondary_fluxes,
    simulate_with_objective,
    calculate_yield_metrics,
    ensure_demand_for_metabolite,
    get_substrate_rxn_for_environment,
    create_engineered_strain
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_base_model_paths(models_dir: str = "models/final_constrained_rnaseq_thermo"):
    return {
        "Glu": os.path.join(models_dir, "iJN1463_Glu_preprocessed_with_DM.xml"),
        "Cit": os.path.join(models_dir, "iJN1463_Cit_preprocessed_with_DM.xml"),
        "Ser": os.path.join(models_dir, "iJN1463_Ser_preprocessed_with_DM.xml"),
        "Fer": os.path.join(models_dir, "iJN1463_Fer_preprocessed_with_DM.xml"),
    }


def main():
    cfg_path = "config/engineering_scenarios.yaml"
    expr_csv = "expression_txt_files/expression_by_reaction.csv"  # user-provided CSV
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not os.path.exists(expr_csv):
        raise FileNotFoundError(f"Expression CSV not found: {expr_csv}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    base_model_paths = build_base_model_paths()
    scenarios = cfg.get("engineering_scenarios", {})
    objectives = cfg.get("objectives", {})
    environments_cfg = cfg.get("environments", {})

    all_results = []

    # For each environment: compute scaling factors, load model, apply wild_secondary locks, run scenarios
    for env_name, model_path in base_model_paths.items():
        logger.info(f"Processing environment: {env_name}")

        # compute scaling factors from expression CSV
        scaling_df = read_expression_scaling(expr_csv, environment=env_name, expr_col_prefix="expression_")
        reaction_factors = dict(zip(scaling_df["reaction_id"], scaling_df["factor"]))

        # load base model
        if not os.path.exists(model_path):
            logger.error(f"Model file not found for {env_name}: {model_path}")
            continue
        base_model = cobra.io.read_sbml_model(model_path)

        # apply environment bounds to base model
        env_bounds = environments_cfg.get(env_name, {})
        for rxn_id, bounds in (env_bounds or {}).items():
            if rxn_id in base_model.reactions:
                try:
                    base_model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
                except Exception:
                    logger.debug(f"Could not set bound for {rxn_id} on base model")

        # ensure DM for G1SAT product metabolite exists once on base model
        try:
            # find product metabolite of G1SAT (prefer cytosolic)
            if "G1SAT" in base_model.reactions:
                g1 = base_model.reactions.get_by_id("G1SAT")
                product_met = None
                for m, coeff in g1.metabolites.items():
                    try:
                        sto = g1.get_coefficient(m)
                    except Exception:
                        sto = g1.metabolites.get(m, 0.0)
                    if sto > 0:
                        if hasattr(m, "compartment") and m.compartment == "c":
                            product_met = m
                            break
                        if product_met is None:
                            product_met = m
                if product_met is not None:
                    ensure_demand_for_metabolite(base_model, product_met.id, dm_prefix="DM")
        except Exception as e:
            logger.debug(f"Could not ensure DM for G1SAT on base model: {e}")

        # compute and apply wild_secondary locks on base_model
        wild_secondary_results = apply_wild_secondary_fluxes(base_model, reaction_factors, objective_rxn="BIOMASS_KT2440_WT3")
        # wild_secondary_results: reaction_id -> (wt_flux, wild_secondary)

        # build precursor_map from config.pathway_definitions if present
        precursor_map = {}
        if "pathway_definitions" in cfg:
            for pname, pdata in cfg["pathway_definitions"].items():
                rxns = pdata.get("reactions", [])
                if len(rxns) >= 2:
                    target = rxns[-1]
                    precursor_map[target] = rxns[:-1]

        # consumer_map default
        consumer_map = cfg.get("competitive_pathways", {})
        # if not provided, ensure G1SAT -> PPBNGS
        if not consumer_map:
            consumer_map = {"G1SAT": "PPBNGS"}

        # iterate scenarios
        for scenario_name, modifications in scenarios.items():
            logger.info(f"  Scenario: {scenario_name}")

            # create engineered strain using wild_secondary_map and precursor_map
            wild_map_signed = {k: v for k, (_, v) in wild_secondary_results.items()}
            engineered = create_engineered_strain(
                base_model,
                modifications,
                wild_secondary_map=wild_map_signed,
                precursor_map=precursor_map,
                consumer_map=consumer_map,
                lock_on_modify=True,
                cap_factor=10.0
            )

            # ensure DM exists in engineered model too
            try:
                if "G1SAT" in engineered.reactions:
                    g1 = engineered.reactions.get_by_id("G1SAT")
                    product_met = None
                    for m, coeff in g1.metabolites.items():
                        try:
                            sto = g1.get_coefficient(m)
                        except Exception:
                            sto = g1.metabolites.get(m, 0.0)
                        if sto > 0:
                            if hasattr(m, "compartment") and m.compartment == "c":
                                product_met = m
                                break
                            if product_met is None:
                                product_met = m
                    if product_met is not None:
                        ensure_demand_for_metabolite(engineered, product_met.id, dm_prefix="DM")
            except Exception:
                pass

            # simulate objectives
            for obj_name, obj_rxn in objectives.items():
                sol, min_growth_abs = simulate_with_objective(engineered, obj_rxn, env_bounds, min_growth_fraction=0.01)

                if sol is None or getattr(sol, "status", None) != "optimal":
                    status = getattr(sol, "status", "failed")
                    logger.warning(f"    {obj_name}: solution status = {status} (scenario={scenario_name}, env={env_name})")
                    all_results.append({
                        "environment": env_name,
                        "scenario": scenario_name,
                        "objective": obj_name,
                        "growth_rate": 0.0,
                        "ala_flux": 0.0,
                        "ala_flux_net": 0.0,
                        "ala_consumption": 0.0,
                        "substrate_uptake": 0.0,
                        "solution_status": status,
                        "modifications": str(modifications),
                        "modification_count": len(modifications),
                        "valid_production": False
                    })
                    continue

                substrate_rxn = get_substrate_rxn_for_environment(env_name)
                metrics = calculate_yield_metrics(sol, "G1SAT", substrate_rxn, environment=env_name, min_growth_abs=min_growth_abs)

                all_results.append({
                    "environment": env_name,
                    "scenario": scenario_name,
                    "objective": obj_name,
                    "growth_rate": metrics["biomass_flux"],
                    "ala_flux": float(sol.fluxes.get("G1SAT", 0.0)),
                    "ala_flux_net": metrics["net_ala_flux"],
                    "ala_consumption": metrics["ala_consumption_flux"],
                    "substrate_uptake": metrics["substrate_uptake"],
                    "solution_status": sol.status,
                    "modifications": str(modifications),
                    "modification_count": len(modifications),
                    "yield_mmol_g": metrics["yield_mmol_g"],
                    "yield_mmol_mmol": metrics["yield_mmol_mmol"],
                    "carbon_yield": metrics["carbon_yield"],
                    "valid_production": metrics["valid_production"]
                })

    # save results
    outdir = Path("results/tables")
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    out_csv = outdir / "enhanced_engineering_results_all_envs.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()
