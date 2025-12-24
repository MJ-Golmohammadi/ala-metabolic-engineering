#!/usr/bin/env python3
"""
Systematic Multi-Environment Metabolic Engineering Evaluation for 5-ALA Production

This script runs engineering scenarios across four environments (Glu, Cit, Ser, Fer).
Key features:
- Uses simulate_with_objective which returns (solution, min_growth_abs)
- Applies epsilon-constraint correctly for ALA objective
- Ensures DM for product exists once per base model (to allow export of excess product)
- Handles infeasible/failed optimizations by recording zeros and continuing
- Computes yield metrics via calculate_yield_metrics
- Produces a consolidated CSV and robustness summary
"""

import cobra
import pandas as pd
import numpy as np
import yaml
from typing import Dict
from pathlib import Path
import sys
import os
import logging

# ensure project root is on path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_utils import (
    simulate_with_objective,
    calculate_yield_metrics,
    ensure_demand_for_product,
    get_substrate_rxn_for_environment
)
from models.create_engineered_strain import create_engineered_strain

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Helper: base model path map (four environments)
# ---------------------------------------------------------
def build_base_model_paths(models_dir: str = "models/final_constrained_rnaseq_thermo") -> Dict[str, str]:
    """
    Return mapping environment -> SBML model path.
    Ensure filenames match your preprocessed models.
    """
    return {
        "Glu": os.path.join(models_dir, "iJN1463_Glu_preprocessed_with_DM.xml"),
        "Cit": os.path.join(models_dir, "iJN1463_Cit_preprocessed_with_DM.xml"),
        "Ser": os.path.join(models_dir, "iJN1463_Ser_preprocessed_with_DM.xml"),
        "Fer": os.path.join(models_dir, "iJN1463_Fer_preprocessed_with_DM.xml"),
    }


# ---------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------
def evaluate_multi_environment_engineering(
    base_model_paths: Dict[str, str],
    scenarios: Dict,
    environments: Dict,
    objectives: Dict
) -> pd.DataFrame:
    """
    For each environment and each engineering scenario:
      - load base model
      - ensure DM for G1SAT exists once
      - create engineered strain (create_engineered_strain handles modifications)
      - simulate objectives using simulate_with_objective (returns sol, min_growth_abs)
      - compute yield metrics using calculate_yield_metrics
      - handle infeasible solutions by recording zeros
    Returns a DataFrame with all results.
    """

    all_results = []

    for env_name, model_path in base_model_paths.items():
        logger.info(f"Evaluating environment: {env_name}")

        # load base model
        try:
            base_model = cobra.io.read_sbml_model(str(model_path))
        except Exception as e:
            logger.error(f"Failed to load model for {env_name}: {e}")
            continue

        # environment bounds from config (may be empty)
        env_config = environments.get(env_name, {})

        # Apply environment bounds to base model (so ensure_demand uses same context)
        with base_model:
            for rxn_id, bounds in (env_config or {}).items():
                if rxn_id in base_model.reactions:
                    try:
                        base_model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
                    except Exception:
                        logger.debug(f"Could not set bound for {rxn_id} in base model")

        # Ensure DM for G1SAT exists ONCE on the base model (so engineered copies inherit it)
        try:
            # ensure_demand_for_product expects a reaction id; it will create DM for product metabolite
            dm_id = ensure_demand_for_product(base_model, "G1SAT", dm_prefix="DM")
            logger.info(f"Ensured DM for G1SAT on base model: {dm_id}")
        except Exception as e:
            logger.warning(f"Could not ensure DM for G1SAT on base model: {e}")
            dm_id = None

        # iterate scenarios
        for scenario_name, modifications in scenarios.items():
            logger.info(f"  Scenario: {scenario_name}")

            try:
                # Create engineered strain using create_engineered_strain
                # create_engineered_strain should accept base_model and modifications and return a new model
                engineered = create_engineered_strain(
                    base_model,
                    modifications,
                    # pass environment and reference objectives if function supports them
                    # (create_engineered_strain implementation may ignore extra args)
                    lock_on_modify=True,
                    cap_factor=10.0
                )

                # Ensure DM exists in engineered model too (safe no-op if already present)
                try:
                    dm_id_eng = ensure_demand_for_product(engineered, "G1SAT", dm_prefix="DM")
                except Exception:
                    dm_id_eng = None

                # Evaluate all objectives for this engineered strain
                for obj_name, objective in objectives.items():

                    # simulate_with_objective returns (solution, min_growth_abs)
                    sol, min_growth_abs = simulate_with_objective(
                        engineered,
                        objective,
                        env_config,
                        min_growth_fraction=0.01
                    )

                    # If optimization failed or infeasible, record zeros and continue
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
                            "yield_mmol_g": 0.0,
                            "yield_mmol_mmol": 0.0,
                            "carbon_yield": 0.0,
                            "valid_production": False,
                            "dm_id": dm_id_eng,
                        })
                        continue

                    # compute substrate reaction id for this environment
                    substrate_rxn = get_substrate_rxn_for_environment(env_name)

                    # compute yield metrics using the SAME min_growth_abs used in optimization
                    yield_metrics = calculate_yield_metrics(
                        sol,
                        "G1SAT",
                        substrate_rxn,
                        environment=env_name,
                        min_growth_abs=min_growth_abs
                    )

                    biomass_flux = yield_metrics.get("biomass_flux", 0.0)
                    gross_ala = float(sol.fluxes.get("G1SAT", 0.0))
                    net_ala = yield_metrics.get("net_ala_flux", 0.0)
                    consumption = yield_metrics.get("ala_consumption_flux", 0.0)
                    substrate_uptake = yield_metrics.get("substrate_uptake", 0.0)

                    result = {
                        "environment": env_name,
                        "scenario": scenario_name,
                        "objective": obj_name,
                        "growth_rate": biomass_flux,
                        "ala_flux": gross_ala,
                        "ala_flux_net": net_ala,
                        "ala_consumption": consumption,
                        "substrate_uptake": substrate_uptake,
                        "solution_status": sol.status,
                        "modifications": str(modifications),
                        "modification_count": len(modifications),
                        "yield_mmol_g": yield_metrics.get("yield_mmol_g", 0.0),
                        "yield_mmol_mmol": yield_metrics.get("yield_mmol_mmol", 0.0),
                        "carbon_yield": yield_metrics.get("carbon_yield", 0.0),
                        "valid_production": yield_metrics.get("valid_production", False),
                        "dm_id": dm_id_eng,
                    }

                    all_results.append(result)

                    logger.info(
                        f"    {obj_name}: growth={biomass_flux:.5f}, "
                        f"ALA_net={net_ala:.5f}, valid={yield_metrics.get('valid_production', False)}"
                    )

            except Exception as e:
                logger.error(f"Scenario {scenario_name} failed in environment {env_name}: {e}")
                # record failure row
                all_results.append({
                    "environment": env_name,
                    "scenario": scenario_name,
                    "objective": "all",
                    "growth_rate": 0.0,
                    "ala_flux": 0.0,
                    "ala_flux_net": 0.0,
                    "ala_consumption": 0.0,
                    "substrate_uptake": 0.0,
                    "solution_status": "failed",
                    "modifications": str(modifications),
                    "modification_count": len(modifications),
                    "yield_mmol_g": 0.0,
                    "yield_mmol_mmol": 0.0,
                    "carbon_yield": 0.0,
                    "valid_production": False,
                    "dm_id": dm_id,
                })
                continue

    return pd.DataFrame(all_results)


# ---------------------------------------------------------
# Robustness metrics
# ---------------------------------------------------------
def calculate_robustness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple robustness metrics per scenario using max_biomass rows.
    Returns DataFrame with mean, std, CV and a robustness score.
    """
    if df.empty:
        raise ValueError("Engineering results dataframe is empty.")

    if "scenario" not in df.columns:
        raise ValueError("Missing 'scenario' column in engineering results.")

    robustness = []

    for scenario in df["scenario"].unique():
        sub = df[(df["scenario"] == scenario) & (df["objective"] == "max_biomass")]

        if sub.empty:
            continue

        ala_vals = sub["ala_flux_net"].values
        ala_mean = np.mean(ala_vals)
        ala_std = np.std(ala_vals)

        robustness.append({
            "scenario": scenario,
            "ala_mean": ala_mean,
            "ala_std": ala_std,
            "ala_cv": ala_std / ala_mean if ala_mean > 0 else np.nan,
            "robustness_score": 1 / (1 + (ala_std / ala_mean)) if ala_mean > 0 else 0,
            "environments_tested": len(sub)
        })

    return pd.DataFrame(robustness)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    # load config
    with open("config/engineering_scenarios.yaml", "r") as f:
        config = yaml.safe_load(f)

    # build base model paths (ensure filenames are correct)
    base_model_paths = build_base_model_paths(models_dir="models/final_constrained_rnaseq_thermo")

    logger.info("Starting multi-environment engineering evaluation...")

    # run evaluation
    df = evaluate_multi_environment_engineering(
        base_model_paths,
        config.get("engineering_scenarios", {}),
        config.get("environments", {}),
        config.get("objectives", {})
    )

    # save results
    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "enhanced_engineering_results.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved engineering results to {out_csv}")

    # compute robustness and save
    try:
        robustness_df = calculate_robustness_metrics(df)
        robustness_df.to_csv(results_dir / "engineering_robustness.csv", index=False)
        logger.info(f"Saved robustness metrics to {results_dir / 'engineering_robustness.csv'}")
    except Exception as e:
        logger.warning(f"Could not compute robustness metrics: {e}")

    logger.info("✅ Engineering evaluation completed successfully.")


if __name__ == "__main__":
    main()
