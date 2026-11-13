# UAV-FL-PM2.5

Data and reproducible code for the paper:

**Communication-Efficient Federated Learning for UAV-Sensed Industrial PM2.5: A Field-Data-Driven Feasibility Study**
Montaser N. A. Ramadan, Hasan Saygin (IEEE Internet of Things Journal, under review).

Three UAVs carrying a low-cost multi-gas node surveyed three contrasting industrial zones in Istanbul (vehicle maintenance, metal galvanisation, recycling), one flight per zone. The repository contains the quality-controlled flight logs, the stationary factory corpus used for pretraining, and the scripts that produce every number, table and figure in the paper and its supplement: the federated-learning experiments, the LoRaWAN airtime model and protocol simulator, and the on-device cost model.

What is measured and what is modelled: the flights, sensor readings and training are real; the federated deployment is emulated on a workstation; the LoRaWAN link is modelled analytically and simulated at protocol level (no radio was flown); on-device training cost is counted, not measured on a microcontroller.

## Key results (TinyLSTM, pooled test RMSE in µg m⁻³, five seeds unless stated)

| Setting | RMSE |
|---|---|
| Persistence (5 s ahead) | 9.48 |
| Local, random init | 10.60 ± 1.09 |
| Federated (FedAvg), random init | 9.34 ± 0.53 |
| Local, transfer init (T-Local) | 7.58 ± 0.83 |
| **Federated, transfer init (T-FL)** | **6.62 ± 0.46** |
| Centralised (pools raw data) | 5.62 ± 0.46 |

- Federation over T-Local, ten paired seeds: 0.86 (95 % CI 0.47 to 1.26).
- 8-bit updates against float32, ten paired seeds: 0.006 worse (95 % CI −0.003 to +0.015).
- Top-10 % sparsification with 8-bit values and error feedback: 6.97 ± 0.60 with 468 B (3 LoRaWAN frames) per round, against 5,956 B (28 frames) dense.
- Complete federated rounds per day, three nodes, 8-bit updates: modelled ceiling 159; simulated 131 (loss-free) and 93 (10 % packet error).

## Repository structure

```
data/
  uav/                          # flights, 5 s logging: lat, lon, alt, speed, CO2, CH2O, TVOC, PM2.5, PM10, T, RH
    zone1_vehicle_maintenance.csv
    zone2_galvanization.csv
    zone3_recycling.csv
    test_flight.csv             # pre-mission test, excluded from analysis
  pretraining/                  # three factories, 5-min interval, ~5 months each
    PCBA_5month.csv  CNC_5month.csv  plastic_5month.csv
code/
  spatial_analysis.py           # cleaning, Moran's I, Getis-Ord Gi*, IDW LOOCV, non-IID statistics
  fl_experiment_v3.py           # leakage-safe split, init x federation grid, compression factorial -> results_fl_v3.json
  fl_experiment_v4.py           # first revision: fragmentation, downlink bound, client drift, LOZO -> results_fl_v4.json
  fl_experiment_v5.py           # second revision: horizons, paired tests, pretraining ablation,
                                #   error-feedback / random-k / adaptive / sign compression -> results_fl_v5.json
  lorawan_sim.py                # event-level LoRaWAN protocol simulation of complete rounds
  table3_r2.py                  # Table 3: analytical ceilings and simulated rates
  supp_tables_r2.py             # supplementary Tables S3/S4
  ondevice_model.py             # on-device operation, memory, time and energy model
  extra_r2.py                   # centralised bound under UAV-domain scaling; gradient-boosted-tree baseline
  fl_experiment.py, fl_experiment_v2.py   # earlier versions, kept for provenance
heatmaps/
  zone1_single_flight.png  zone2_single_flight.png  zone3_single_flight.png
                                # sensor-reported PM2.5 interpolated from the single real flight per zone
figures/                        # analysis figures
```

## Reproduce

```bash
pip install -r requirements.txt
cd code
python spatial_analysis.py
python fl_experiment_v3.py      # main results (CPU, tens of minutes)
python fl_experiment_v4.py
python fl_experiment_v5.py      # sections F C P H; about an hour on CPU
python lorawan_sim.py           # optional argument: one payload configuration name
python table3_r2.py
python supp_tables_r2.py
python ondevice_model.py
python extra_r2.py
```

All learning experiments are seeded (seeds 42, 1, 2, 3, 4; ten-seed tests add 5 to 9). Note that `fl_experiment_v3.federated()` quantises updates to 8 bits by default; v5 always states the operator explicitly.

## Data notes

- Each zone was surveyed in a single late-morning or midday flight of 17 to 21 minutes (21–23 July 2025). There are no repeated or early-morning flights.
- The airborne sensors were not co-located with a reference-grade monitor, nor compared with one another; absolute concentrations are sensor-reported and uncalibrated.
- Cleaning drops GNSS fixes outside an Istanbul bounding box, implausible altitudes (> 2000 m) and speed spikes (> 20 m s⁻¹): 682 raw → 678 usable samples.
- *Map lines delineate study areas and do not necessarily depict accepted national boundaries.*

## Citation

```
@article{ramadan_uav_fl_pm25,
  title   = {Communication-Efficient Federated Learning for UAV-Sensed Industrial PM2.5: A Field-Data-Driven Feasibility Study},
  author  = {Ramadan, Montaser N. A. and Saygin, Hasan},
  journal = {IEEE Internet of Things Journal},
  note    = {Under review}
}
```

## License

Code: MIT (see `LICENSE`). Data and figures: CC BY 4.0.
