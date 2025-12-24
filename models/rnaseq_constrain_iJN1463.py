#!/usr/bin/env python3
"""
Minimal GEM preprocessing for 4 environments (Glu, Cit, Ser, Fer):

- No RNA‑seq constraints
- No thermodynamic (ΔG) constraints
- No modification of reaction bounds except EX_ reactions
- All EX_ reactions are closed by default: [-eps, +eps]
- Only the carbon-source EX_ reaction of each environment is opened
- For every reaction product, a DM_ reaction is automatically created
- Four separate output models are generated, one per environment
"""

import os
import cobra
import logging

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "models/iJN1463.xml"
OUTPUT_DIR = "models/final_constrained_rnaseq_thermo"
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger = logging.getLogger(__name__)

EPS = 1e-6          # Default closed bound for EX_ reactions
DM_UPPER = 1000.0   # Upper bound for DM_ reactions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Environment-specific EX_ uptake definitions
ENV_EXCHANGES = {
    "Glu": {"EX_glc__D_e": -1.0},
    "Cit": {"EX_cit_e": -1.0},
    "Ser": {"EX_ser__L_e": -2.0},
    "Fer": {"EX_fer_e": -0.6},
}


# -----------------------------
# Create DM reaction for a metabolite
# -----------------------------
def create_dm_for_metabolite(model, met):
    """
    Create a DM_ reaction for a metabolite if it does not already exist.
    DM reactions allow optional secretion of products.
    """
    dm_id = f"DM_{met.id}"
    if dm_id in model.reactions:
        return model.reactions.get_by_id(dm_id)

    dm = cobra.Reaction(dm_id)
    dm.name = f"Demand for {met.id}"
    dm.lower_bound = 0.0
    dm.upper_bound = DM_UPPER
    dm.add_metabolites({met: -1})
    model.add_reactions([dm])
    return dm


# -----------------------------
# Process model for one environment
# -----------------------------
def process_environment(env_name, env_settings):
    logging.info(f"Processing environment: {env_name}")

    # Load fresh model
    model = cobra.io.read_sbml_model(MODEL_PATH)

    # ---------------------------------------------------------
    # 1) Close all EX_ reactions by default
    # ---------------------------------------------------------
    for rxn in model.reactions:
        if rxn.id.startswith("EX_"):
            rxn.lower_bound = -EPS
            rxn.upper_bound = EPS

    # ---------------------------------------------------------
    # 2) Open only the environment-specific EX_ reactions
    # ---------------------------------------------------------
    for ex_id, uptake in env_settings.items():
        if ex_id in model.reactions:
            model.reactions.get_by_id(ex_id).lower_bound = uptake
            model.reactions.get_by_id(ex_id).upper_bound = 1000.0
        else:
            logging.warning(f"Exchange {ex_id} not found in model.")

    # ---------------------------------------------------------
    # 3) Create DM_ reactions for all products
    # ---------------------------------------------------------
    count_dm = 0
    for rxn in model.reactions:
        for met, coeff in rxn.metabolites.items():
            if coeff > 0:  # product metabolite
                create_dm_for_metabolite(model, met)
                count_dm += 1

    logging.info(f"DM reactions created: {count_dm}")

    # ---------------------------------------------------------
    # 4) Save environment-specific model
    # ---------------------------------------------------------
    out_path = os.path.join(
        OUTPUT_DIR,
        f"iJN1463_{env_name}_preprocessed_with_DM.xml"
    )
    cobra.io.write_sbml_model(model, out_path)
    logging.info(f"Saved model for {env_name}: {out_path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    logging.info("Starting preprocessing for all environments...")

    for env_name, env_settings in ENV_EXCHANGES.items():
        process_environment(env_name, env_settings)

    logging.info("All 4 environment-specific models generated successfully.")
