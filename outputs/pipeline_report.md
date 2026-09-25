# Pipeline Report
## Input
- SMILES: CCO
- Dose (nM): 100.0
## IC50 Predictions
| Channel | pIC50 | IC50 (nM) |
|---|---:|---:|
| HERG | 4.167536905713713 | 67992.82632328934 |
| NAV | 4.6218658780395785 | 23885.488179291708 |
| CAV | 4.253881897931607 | 55733.72906633142 |

## Dose-response
![Dose-response](dose_response_curve.png)

## ORd Simulation
![ORd voltage](ord_simulation_voltage.png)

## Simulation Features
- **status**: complete
- **features**: {'RMP': -90.74628898038942, 'Peak': 32.398092195150284, 'APD50': 195.072404959668, 'APD90': 236.13747936891136, 'Triangulation': 41.06507440924335}
- **artifacts**: {'voltage_plot': 'C:\\Users\\HP\\Desktop\\capstone_project\\classification_backend\\inference\\results\\ord_voltage.png'}

## Classification
- Predicted class: **3** (High)
- Description: High blocking
- Probabilities: [0.07314412466656776, 0.4146362149607505, 0.48614955960528344, 0.026070100767398834]
