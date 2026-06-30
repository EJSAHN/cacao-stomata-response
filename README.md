# cacao-stomata-response

Trial-aware, control-anchored analysis pipeline for cacao stomatal pathogen-response morphometrics.

## Overview

This package analyzes the stimulus-response stomata dataset by comparing each infected condition against the matched control within the same genotype, light regime, hour, and trial. The pipeline is designed to keep the trial structure intact, quantify size and shape responses separately, and export all results to a single Excel workbook.

The developmental leaf dataset is optional. When provided, it is summarized descriptively as a separate developmental reference and is not merged into the stimulus-response analysis.

## Terminology note

A few internal column and label names differ from the wording used in the associated manuscript. They refer to the same quantities:

- `strain` (code) corresponds to **isolate** in the manuscript (e.g., GH8, GH21, ZTH0145, R. solani), plus the control label.
- `light_dark` encodes the photoperiod regime as `L` (light) and `D` (dark).
- The control group is identified by an exact strain label, configurable with `--control-label` (default: `Control`).
- The light/time axes built in step 6 are **control-derived light/time reference axes**: they summarize baseline photoperiod- and time-related variation within control conditions and are not independently induced abiotic stress treatments.

## Core analysis steps

1. Load and validate the stimulus-response workbook.
2. Standardize the shared morphometric features using control cells from the stimulus dataset.
3. Build matched control references within each genotype x light regime x hour x trial stratum.
4. Quantify feature-level response shifts for each infected condition.
5. Quantify multivariate divergence for size, shape, and combined feature sets.
6. Build control-derived light/time reference axes from control cells only.
7. Compare infection response vectors against the light and time reference axes.
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

Optional arguments:

- `--control-label` sets the exact strain label used for control cells (default: `Control`).
- `--regularization` sets the covariance regularization used in the Gaussian 2-Wasserstein calculation (default: `1e-6`).

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
- `control_axes`: control-derived light/time reference axes
- `response_alignment`: response-vector alignment against available light and time reference axes
- `condition_feature_consensus`: feature-level aggregation across trials
- `condition_vector_consensus`: vector-level aggregation across trials
- `alignment_consensus`: alignment aggregation across trials
- `trial_robustness`: leave-one-trial-out vector stability
- `development_group_summary`: optional descriptive summary from the developmental leaf dataset
- `development_leaf_summary`: optional leaf-level summary from the developmental leaf dataset

## Notes

- The pipeline does not generate figures.
- All outputs are exported to Excel.
- The stimulus dataset is the only dataset used for response geometry, divergence, and reference-axis alignment.
