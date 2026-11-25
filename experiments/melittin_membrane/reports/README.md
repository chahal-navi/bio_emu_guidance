# Reports

Numbered in the order they were produced. Each supersedes conclusions in the one before it,
so read them in order — several findings were reversed by later evidence.

| # | report | what it established |
|---|---|---|
| 01 | `01_objective_functions_report.html` | Screened 9 candidate objectives. Found the membrane **frame** was the dominant defect, not the objectives: searching for the placement let objectives improve their score by moving the membrane instead of fixing the structure. Part three adds the placement screen, which identified plain Ez on Cα as the best objective. |
| 02 | `02_opm_frame_sweep_report.html` | Transferring the frame from the OPM-oriented melittin structure repaired every objective on paper, then the guidance re-run still destroyed 74–98% of structures. Diagnosed two causes: the screen validated on finished structures rather than noisy intermediates, and the failure is position/orientation channel drift, not the objective. |
| 03 | `03_guidance_results_report.html` | Ez under the OPM frame improves **100% of structures** at every λ (paired p = 4×10⁻⁸ at λ=20, zero structural cost). Corrects every earlier significance test, which was unpaired on paired data. Maps the λ curve and finds a fixed effect-vs-damage exchange rate — a ceiling on position-only guidance. |

Regenerate any of them with the matching script in the parent directory
(`make_report.py`, `make_sweep_report.py`, `make_guidance_report.py`).
