# Molecular Semiconductor Property Prediction Using AI

This repository contains the code, processed results, and analysis workflows produced for a Master's research project investigating machine learning approaches for predicting the optical properties of organic molecules to aid in semiconductor materials discovery. The project reproduces and extends previously published Deep4Chem/Chemprop methodology, comparing a graph neural network model with conventional descriptor-based tree ensemble algorithms. 

Particular attention is given to prediction of two experimentally relevant UV/Vis properties:

- **Absorption maximum, λmax (nm).**
- **Logarithm of the molar extinction coefficient, log10(ε).**

## Project Overview

The project consists of four principal components:

1. **Replication of a published Chemprop methodology:**
   - Reproduction of the original experimental environment using Chemprop v1.5.2.
   - Training and evaluation using the Deep4Chem dataset.
   - Performance comparison with that reported by the original authors.

2. **Descriptor-based machine learning:**
   - Calculation of RDKit molecular descriptors for chromophores and solvent inputs within Deep4Chem.
   - Removal of highly correlated molecular descriptors.
   - Training and optimisation of:
     - Random Forest (RF);
     - XGBoost (XGB);
     - LightGBM (LGBM).
   - Independent test-set evaluation using the error metrics RMSE and MAE.
   - Feature-importance and SHAP analyses to investigate model interpretability.

3. **Reaxys dataset investigation:**
   - Extraction and processing scripts for UV/Vis spectroscopy information obtained through licensed access to Reaxys.
   - Development of an amended extraction workflow to handle invalid or unsuitable spectral records.
   - Training and evaluation of Chemprop models on the large Reaxys-derived dataset.

4. **Modern Chemprop implementation:**
   - Migration of the workflow from Chemprop v1.5.2 to Chemprop v2.3.1.
   - Hyperparameter optimisation using the modern Chemprop `hpopt` workflow.
   - Retraining and evaluation on Deep4Chem and Reaxys datasets.
   - Comparison of Chemprop predictive performance between versions.

## Repository Structure

The repository is broadly organised as follows:

### `Data/`

Contains datasets that are permitted to be redistributed and intermediate data required by the modelling workflows. The "raw" Deep4Chem dataset, as provided by the original Chemprop methodology's authors, and training/test subsets alongside processed descriptor files are provided where appropriate.

Reaxys data are not redistributed in this repository. Reaxys is a proprietary database and the corresponding raw exports and derived row-level datasets must be obtained separately by users with appropriate access. The Reaxys registry numbers CSV file, again provided by the main methodology's authors, can be used directly to obtain raw exported Reaxys data prior to target property extraction into the final database.

### `Python_Scripts/`

Contains the Python scripts used throughout the project. Scripts are separated according to dataset and modelling approach, including:

- Deep4Chem Chemprop v1 and v2 workflows.
- Chemprop training, prediction, and evaluation.
- RDKit descriptor calculation and filtering for Deep4Chem.
- Hyperparameter optimisations.
- RF, XGB and LGBM modelling.
- Reaxys data processing and extraction.

### `Outputs/`

Contains non-proprietary outputs generated during model development and evaluation, including combinations of:

- RMSE and MAE results.
- Actual-versus-predicted plots.
- Evaluation metric summaries.
- Feature-importance summaries.
- SHAP summaries and visualisations.
- Hyperparameter optimisation results.
- Training configuration files and logs.
  
### `Python_Notebooks/`

Contains Jupyter and Google Colab notebooks used for model execution, result processing, visualisation, and analysis. These include workflows used for GPU-accelerated Chemprop training and prediction where appropriate.

Large trained Chemprop checkpoint files (`.pt` and `.ckpt`) are excluded from the public repository where appropriate.

## Machine Learning Models

### Chemprop

Chemprop uses directed message-passing and feed-forward neural networks to learn molecular representations directly from molecular graph structures represented by SMILES strings. Two major implementations were investigated:

- **Chemprop v1.5.2** - used to reproduce the software environment and methodology corresponding as closely as possible to the original authors' work.
- **Chemprop v2.3.1** - used to investigate performance attributed to the modern Chemprop architecture, training framework, and integrated hyperparameter optimisation workflow.

The official Chemprop project and documentation are available from the Chemprop repository.

### Descriptor-Based Models

Three tree-based ensemble algorithms were developed as alternative models:

- RF
- XGB
- LGBM

Unlike Chemprop, these models require pre-engineered molecular representations. RDKit descriptors were therefore calculated independently for the chromophore and solvent molecular inputs. Highly correlated descriptors were subsequently filtered to reduce feature redundancy. Model interpretation was performed using both independent feature-importance measures and SHAP values.

## Model Evaluation

Models were primarily evaluated using:

**Root Mean Squared Error (RMSE)** - RMSE emphasises large prediction errors and was used as the principal optimisation/evaluation metric.
**Mean Absolute Error (MAE)** - MAE provides the average absolute difference between experimental and predicted values, therefore giving a more directly interpretable measure of typical prediction error.

Both metrics were evaluated independently for absorption wavelength and log10(ε). Where ensembles or repeated models were used, mean performance and standard deviations were calculated across the individual models.

## Model Interpretability

Feature importance and SHAP analyses were performed for the RF, XGB and LGBM models. These analyses were used to identify molecular descriptors associated with predictions of the absorption wavelength and log(molar extinction coefficient) targets. Comparison across the three algorithms was used to distinguish consistently influential chemical and physical information from model-specific feature selection processes.

## Deep4Chem and Reaxys

The Deep4Chem dataset was used as the principal benchmark for reproducing and comparing machine learning methodologies. A substantially larger dataset was additionally reconstructed from exported Reaxys UV/Vis spectroscopy records using registry numbers associated with the original research methodology. The extraction workflow was modified during this project to improve handling of multiple spectra, invalid measurements, and numerical errors.

### Reaxys Data Availability

Reaxys is a proprietary Elsevier database. Consequently, raw Reaxys XML exports and the reconstructed Reaxys training/test datasets are not provided in this repository. The associated processing and modelling scripts are retained to document the methodology and allow authorised workflow reproduction using data obtained through their own licensed Reaxys access.

## Computing Environments

Experiments were performed across several computational environments, including:

- Jupyter Notebooks
- Google Colab
- University of Strathclyde HPC resources

Dedicated Conda environments were used to reproduce the Chemprop v1.5.2 software stack and to provide a separate modern Chemprop installation. The distinction between these environments was important as these Chemprop versions have substantially different dependencies, command-line interfaces, and training frameworks.

## Reproducibility

The repository is intended to provide the code and methodological information required to reproduce the modelling experiments where the underlying datasets are accessible.

Exact numerical reproduction may depend upon:

- Python version;
- Chemprop version;
- package versions;
- CPU/GPU hardware;
- PyTorch/CUDA version;
- dataset splitting;
- random seeds;
- availability and version of the underlying data.

The two Chemprop implementations should be treated as related but independent experimental environments due to their differing underlying frameworks.

## Software

Major packages used throughout the project include:

- Python
- Chemprop
- PyTorch
- RDKit
- Scikit-Learn
- XGBoost
- LightGBM
- SHAP
- Pandas
- NumPy
- SciPy
- Matplotlib

Package versions differ between the original and modern Chemprop environments and should be reproduced for exact methodological comparisons.

## Important Notes

- Proprietary Reaxys-derived datasets were intentionally excluded from this repository - access to the database must be obtained independently.
- Excessively large Chemprop model and checkpoints files are also excluded.
- File paths within many scripts may require modification for execution on different systems.

## Acknowledgements

Special thanks to my supervisor, Dr. Tahereh Nematiaram (aram-tahereh-git), for providing the opportunity to work on this project and for her continuous support.

This project builds upon the open-source Deep4Chem dataset and previously published Chemprop-based methodology for prediction of molecular optical properties. 

Chemprop is an open-source package for molecular property prediction using message passing neural networks. Reproducing or extending the Chemprop components of this work requires citation of the appropriate Chemprop publications.

Reaxys data used during this project were accessed through the University of Strathclyde institutional licensing.

Several results were obtained using the EPSRC funded ARCHIE-WeSt High Performance Computer (www.archie-west.ac.uk). EPSRC grant no. EP/K000586/1.

