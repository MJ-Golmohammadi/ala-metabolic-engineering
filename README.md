# 🧬 Comprehensive Computational Framework for 5-ALA Production in _Pseudomonas putida_ KT2440: A Multi-Omics Integrated Metabolic Engineering Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![COBRApy](https://img.shields.io/badge/COBRApy-0.26.3-green.svg)](https://opencobra.github.io/cobrapy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

 The pipeline integrates **flux balance analysis (FBA), RNA-seq expression constraints, thermodynamic modeling, flux variability analysis (FVA), machine learning, and multi-criteria decision analysis** to systematically identify optimal genetic engineering strategies across four distinct carbon environments (glucose, citrate, serine, and ferulate). The current implementation serves as a static constraint‑based framework for metabolic engineering target prioritization and hypothesis generation. Upon future development of a fully dynamic genome‑scale metabolic model, this pipeline can be readily extended to enable dynamic simulations and real‑time flux reallocation analyses.   “This project made extensive use of artificial intelligence (Microsoft Copilot and DeepSeek) for code generation, refactoring, documentation, etc. While the conceptual framework and biological interpretations remained human‑driven, AI significantly accelerated the implementation process and helped maintain high coding standards throughout the pipeline.”

 ## Project Status

The project is currently incomplete. While the computational framework has been partially implemented, the genome‑scale metabolic model (iJN1463) lacks the robustness needed for systematic engineering: many multi‑gene modification scenarios lead to infeasible solutions, preventing reliable optimization. Completion of this project awaits a more robust model that can tolerate multiple simultaneous genetic perturbations without loss of feasibility.


## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Repository Structure](#repository-structure)
- [Input Data](#input-data)
- [Output Results](#output-results)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This project presents a systematic computational pipeline for optimizing 5-ALA biosynthesis in *P. putida* through:

- **Multi-environment analysis** across four carbon sources (Glucose, Citrate, Serine, Ferulate)
- **15+ engineering scenarios** targeting C5 pathway enhancement
- **Integrated ML-FBA approach** for predictive metabolic engineering
- **Comprehensive bottleneck identification** using Flux Variability Analysis
- **Publication-ready visualizations** and statistical analysis

## ✨ Key Features

- **🔬 Multi-omics Integration**: Combines RNA-seq expression data with thermodynamic constraints
- **🧪 Multi-Environment Analysis**: Evaluates strain performance across diverse growth conditions
- **🤖 Machine Learning**: Random Forest + SHAP analysis for target prioritization
- **📊 Comprehensive Analytics**: FVA, Flux Coupling, Sensitivity, and Robustness analysis
- **🎨 Publication Quality**: Automated figure generation for scientific publications
- **⚡ Reproducible**: Complete workflow with configuration management

---

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Clone repository
git clone https://github.com/mohammadjg/ala-metabolic-engineering.git
cd ala-metabolic-engineering

# 2. Install requirements
pip install -r requirements.txt


jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 3. Run preprocessing file for models
python .\models\rnaseq_constrain_iJN1463.py

# 4. Run complete analysis
python run_analysis.py

