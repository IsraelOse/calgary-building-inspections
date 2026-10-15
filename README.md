# Calgary Building Inspections Analysis

Analysis of **107,067 building-inspection records** across four geographic zones of Calgary, asking what actually drives inspection outcomes.

## Key findings
- **Geographic zone barely matters.** The zone effect is statistically significant but practically negligible (Cramér's V = **0.015**).
- **Inspection *type* is the dominant predictor** — a ~**73 percentage-point** spread in failure rate across inspection types.
- A Random Forest classifier reaches **ROC-AUC 0.720**, confirming inspection type and related operational features carry the signal, not location.

The practical takeaway: targeting resources by inspection type is far more effective than targeting by neighbourhood.

## Repo layout
| Path | What it is |
|------|-----------|
| `calgary_building_inspections.ipynb` | Main analysis (chapters 5–8) |
| `Calgary_Inspection_Clean.csv` | Cleaned dataset |
| `calgary_inspections.sql` | SQL used for extraction/shaping |
| `Calgary_ERD.pdf` | Entity-relationship diagram |
| `Calgary_Inspections_Report.pdf` | Final report |
| `Calgary_Inspections_Presentation.pptx` | Slide deck |
| `results_summary.json` | Machine-readable results |

## Tech
Python · pandas · scikit-learn · SQL · (reporting in Word/PowerPoint)

> **Academic group project (Group 1)**, DAMO Master of Data Analytics. Shared coursework; teammate contributions acknowledged. _Add teammate names here._
