# Calgary Building Inspections Analysis

Analysis of **~105,000 Calgary building inspections** asking what actually drives inspection failure: geographic zone, or the type of inspection? Delivered as a **modular, production-style pipeline** with a figure-rich analysis notebook and report. Seed = 42.

## Key findings (reproduced by the pipeline)
- **Geographic zone barely matters.** The zone–outcome association is statistically significant but a negligible effect size: **Cramér's V = 0.015**.
- **Inspection type dominates.** In the Random Forest, zone features account for just **1.4%** of importance versus **98.6%** for inspection type and year.
- Random Forest reaches **ROC-AUC 0.720**; the strongest single feature is the Foundation inspection type.

The practical message: target inspection resources by inspection *type*, not by neighbourhood.

## Repository architecture
Decoupled, block-based; type hints and `logging` throughout.

```
.
├── data_loaders/        # Read the cleaned inspections CSV
├── transformers/        # clean, statistics (chi-square/Cramer's V), features, model_training
├── data_exporters/      # Write metrics, importances, stats summary
├── utils/               # Shared config + logging
├── pipelines/
│   └── calgary_inspections/   # run.py + metadata.yaml + requirements.txt
├── calgary_building_inspections.ipynb   # Full analysis (10 figures)
├── calgary_inspections.sql              # SQL extraction
├── Calgary_ERD.pdf                      # Entity-relationship diagram
├── Calgary_Inspections_Report.pdf       # Report
├── Calgary_Inspection_Clean.csv         # Cleaned dataset
└── outputs/                             # Generated metrics + results_summary.json
```

## Run it
```bash
pip install -r pipelines/calgary_inspections/requirements.txt
python -m pipelines.calgary_inspections.run
```

## Approach
- **Clean:** drop null outcomes/years, coerce dtypes and dates.
- **Association test:** chi-square test of independence between zone and failure, with Cramér's V effect size.
- **Features:** one-hot encode zone and inspection type, with inspection year.
- **Models:** Logistic Regression, Decision Tree, Random Forest; 5-fold CV ROC-AUC plus held-out test metrics; Random-Forest importance split between zone and type/year.

## Results (held-out test set)
| Model | CV ROC-AUC | ROC-AUC | PR-AUC | Precision | Recall |
|-------|:---------:|:------:|:-----:|:--------:|:-----:|
| Logistic Regression | 0.680 | 0.677 | 0.557 | 0.807 | 0.152 |
| Decision Tree | 0.669 | 0.670 | 0.635 | 0.853 | 0.116 |
| Random Forest | 0.720 | 0.720 | 0.605 | 0.706 | 0.220 |

## Tech
Python · pandas · scikit-learn · SciPy · SQL

*Project from the Master of Data Analytics program, University of Niagara Falls Canada.*
