# cacao-stomata-response

A Python package for trial-aware, control-anchored analysis of cacao stomatal pathogen-response morphometrics.

## Overview

This package analyzes the stimulus-response stomata dataset by comparing each infected condition against the matched control within the same genotype, light regime, hour, and trial. The pipeline is designed to keep the trial structure intact, quantify size and shape responses separately, and export all results to a single Excel workbook.

The developmental leaf dataset is optional. When provided, it is summarized descriptively as a separate developmental reference and is not merged into the stimulus-response analysis.

## Core analysis steps

1. Load and validate the stimulus-response workbook.
2. Standardize the shared morphometric features using control cells from the stimulus dataset.
3. Build matched control references within each genotype × light regime × hour × trial stratum.
4. Quantify feature-level response shifts for each infected condition.
5. Quantify multivariate divergence for size, shape, and combined feature sets.
6. Build within-dataset abiotic reference axes from control cells only.
7. Compare infection response vectors against light and time axes.
8. Summarize condition-level responses across trials.
9. Run leave-one-trial-out robustness checks.
10. Export all results to Excel.

## Feature modules

- Size: area, perimeter, length, width
- Shape: lwr, circularity, distance_is_cg

## Installation

### Conda
```bash
conda env create -f environment.yml
conda activate cacao-stomata-response
pip install -e .
```

### Pip
```bash
pip install -e .
```

## Command line usage

```bash
cacao-stomata-response \
  --stomata-file "Cacao stomata.xlsx" \
  --leaf-file "Cacao leaf.xlsx" \
  --output-file "cacao_stomata_response_results.xlsx"
```

The leaf file is optional:

```bash
cacao-stomata-response \
  --stomata-file "Cacao stomata.xlsx" \
  --output-file "cacao_stomata_response_results.xlsx"
```

## Output workbook

The workbook contains the following sheets:

- `readme`: run metadata and sheet descriptions
- `stimulus_counts`: raw condition counts in the stimulus dataset
- `reference_issues`: missing controls or missing axis references
- `scaling_reference`: control-based scaling parameters
- `control_reference`: matched control centroids by genotype, light regime, hour, and trial
- `matched_cell_data`: centered cell-level data in standardized feature space
- `trial_feature_effects`: feature-level control vs infected comparisons within each trial
- `trial_response_vectors`: response vectors by genotype, light regime, hour, trial, and strain
- `trial_module_divergence`: multivariate divergence summaries for size, shape, and all features
- `control_axes`: control-derived light and time axes
- `response_alignment`: response-vector alignment against available light and time axes
- `condition_feature_consensus`: feature-level aggregation across trials
- `condition_vector_consensus`: vector-level aggregation across trials
- `alignment_consensus`: alignment aggregation across trials
- `trial_robustness`: leave-one-trial-out vector stability
- `development_group_summary`: optional descriptive summary from the developmental leaf dataset
- `development_leaf_summary`: optional leaf-level summary from the developmental leaf dataset

## Notes

- The pipeline does not generate figures.
- All outputs are exported to Excel.
- The stimulus dataset is the only dataset used for response geometry, divergence, and axis alignment.
