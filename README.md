# UAV-FL-PM2.5

Data and reproducible code for the paper:

**Communication-Efficient Transfer Federated Learning on a LoRaWAN-Connected Multi-UAV System for Field Monitoring of Industrial PM₂.₅**
Montaser N. A. Ramadan, Hasan Saygin.

A LoRaWAN-connected three-UAV system performs dense geo-tagged sensing and runs **transfer federated learning (T-FL)** for on-board PM₂.₅ prediction across three contrasting industrial zones in Istanbul (vehicle maintenance, metal galvanisation, recycling). This repository contains the quality-controlled field data, the factory pre-training data, and the scripts that produce every quantitative result and figure in the paper.

## Key results (reproduced by the code here)

| Model / setting | RMSE (µg m⁻³) | MAE | R² |
|---|---|---|---|
| Persistence baseline | 9.4 | 4.3 | 0.83 |
| 1D-CNN, FL | 14.3 | 9.0 | 0.60 |
| 1D-CNN, T-FL | 9.4 | 6.1 | 0.83 |
| TinyLSTM, FL | 8.6 | 4.9 | 0.85 |
| **TinyLSTM, T-FL** | **6.3** | **4.3** | **0.92** |

- Spatial clustering: Moran's *I* = 0.57 / 0.53 / 0.50 (each *p* = 0.001).
- Non-IID across zones: Kruskal–Wallis *H* = 265 (*p* < 10⁻⁵⁰); pairwise Jensen–Shannon divergence 0.18–0.51 bits.
- Update compression (8-bit quantisation + top-20 % sparsification): 5.96 kB → 0.89 kB (85 %); LoRaWAN airtime 8.7 s → 1.3 s.

## Repository structure

```
data/
  uav/                       # field flights (5 s logging): lat, lon, alt, speed, CO2, CH2O, TVOC, PM2.5, PM10, T, RH
    zone1_vehicle_maintenance.csv
    zone2_galvanization.csv
    zone3_recycling.csv
    test_flight.csv          # pre-mission test (excluded from analysis)
  pretraining/               # 3 factories, 5-min interval, ~5 months each (transfer-learning source)
    PCBA_5month.csv
    CNC_5month.csv
    plastic_5month.csv
code/
  spatial_analysis.py        # cleaning + Moran's I + Getis-Ord Gi* + IDW LOOCV + non-IID stats -> results_spatial.json
  fl_experiment.py           # pretrain TinyLSTM/1D-CNN, FedAvg fine-tune, FL vs T-FL, compression -> results_fl.json
  make_figures.py            # spatial / non-IID / IDW figures
  make_fl_figures.py         # convergence + FL-vs-T-FL figures
figures/                     # generated analysis figures
heatmaps/                    # rendered 3D PM2.5 heatmaps over satellite imagery (09:00 and 14:00 per zone)
```

## Reproduce

```bash
pip install -r requirements.txt
cd code
python spatial_analysis.py     # spatial + non-IID statistics
python fl_experiment.py        # transfer federated learning (CPU is fine)
python make_figures.py
python make_fl_figures.py
```

All scripts are seeded (`seed = 42`). `fl_experiment.py` runs on CPU in a few minutes.

## Data notes

- UAV nodes use low-cost sensors and were not co-located with a reference-grade monitor; absolute concentrations carry the usual low-cost-sensor uncertainty, while spatial contrasts and model comparisons (internal to the same sensors) are robust to it.
- Cleaning drops GNSS fixes outside an Istanbul bounding box, implausible altitudes (> 2000 m), and speed spikes (> 20 m s⁻¹): 682 raw → 678 usable zone samples.
- *Map lines delineate study areas and do not necessarily depict accepted national boundaries.*

## Citation

```
@article{ramadan_uav_fl_pm25,
  title   = {Communication-Efficient Transfer Federated Learning on a LoRaWAN-Connected Multi-UAV System for Field Monitoring of Industrial PM2.5},
  author  = {Ramadan, Montaser N. A. and Saygin, Hasan},
  journal = {ICT Express},
  note    = {Under review}
}
```

## License

Code: MIT (see `LICENSE`). Data and figures: CC BY 4.0.
