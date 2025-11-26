"""
Script for extracting gene-reaction associations from the genome-scale metabolic model (GEM) of *Pseudomonas putida* KT2440.
This mapping is essential for downstream analyses such as gene essentiality, pathway engineering, and transcriptomic integration.
"""

import cobra
import pandas as pd

# Load the genome-scale metabolic model (SBML format)
model = cobra.io.read_sbml_model("models/final_constrained_rnaseq_thermo/iJN1463_Cit_ExprThermoConstrainedFile.xml")

# Initialize list to store gene-reaction associations
mapping_data = []

# Iterate through all reactions in the model
for reaction in model.reactions:
    # For each reaction, retrieve associated genes
    for gene in reaction.genes:
        # Append mapping entry with reaction ID, gene ID, and GPR rule
        mapping_data.append({
            "reaction_id": reaction.id,
            "gene_id": gene.id,
            "gene_reaction_rule": reaction.gene_reaction_rule
        })

# Convert mapping list to a structured DataFrame
mapping_df = pd.DataFrame(mapping_data)

# Export the mapping to Excel for downstream use (e.g., annotation, visualization, transcriptomic overlay)
mapping_df.to_csv("results/tables/gene_reaction_mapping.csv", index=False)

print("✅ Gene-reaction mapping successfully exported to results/tables/gene_reaction_mapping.csv")
