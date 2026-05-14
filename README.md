# 🧬 5-ALA Metabolic Engineering in *Pseudomonas putida* KT2440

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![COBRApy](https://img.shields.io/badge/COBRApy-0.26.3-green.svg)](https://opencobra.github.io/cobrapy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive computational framework for metabolic engineering of **5-Aminolevulinic Acid (5-ALA)** production in *Pseudomonas putida* KT2440. This repository integrates **Flux Balance Analysis (FBA)**, **RNA-seq constraints**, **thermodynamic modeling**, and **Machine Learning** to identify optimal genetic engineering strategies.

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


# 3. Run complete analysis
python run_analysis.py

