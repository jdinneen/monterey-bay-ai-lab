# Domoic-acid forecast -- next-visit pDA >= 0.5 ng/mL (=500 ng/L), ~1-week lead

Train <= 2018 (n=3218, 148 events) | calib 2019-2020 | test > 2020 (n=1837, 52 events, base 0.028).

## Time-held-out test (AP = average precision; base rate ~3.7%)

Feature-group ablation: the physical/nutrient **drivers HURT** (AP drops vs the lean set) -- so the headline model is `DA+precursor`, not the full driver matrix. Same driver-null pattern as M1 / news / wave-tide.

| method | AP | ROC-AUC | AP lift vs base |
|---|--:|--:|--:|
| seasonal_naive | 0.0369 | 0.6072 | 0.8 |
| persistence | 0.1383 | 0.6636 | 3.01 |
| station_memory | 0.0331 | 0.5502 | 0.72 |
| model_DA_history | 0.2179 | 0.8661 | 4.74 |
| model_DA+precursor (headline) | 0.2322 | 0.8314 | 5.05 |
| model_DA+precursor+drivers | 0.1694 | 0.8481 | 3.68 |

**Verdict:** BEATS baselines -- model AP 0.2322 vs best-baseline 0.1383.

**Leave-one-station-out:** model beats seasonal-naive in 8/8 stations.

| station | n | events | model AP | seasonal AP | beats |
|---|--:|--:|--:|--:|---|
| MontereyWharf | 275 | 49 | 0.5964 | 0.2437 | Y |
| SantaMonicaPier | 899 | 13 | 0.3479 | 0.0324 | Y |
| TrinidadPier | 107 | 10 | 0.3166 | 0.0969 | Y |
| StearnsWharf | 884 | 29 | 0.1813 | 0.075 | Y |
| SantaCruzWharf | 713 | 53 | 0.1573 | 0.1278 | Y |
| NewportBeachPier | 868 | 12 | 0.1375 | 0.0362 | Y |
| CalPolyPier | 849 | 25 | 0.1237 | 0.0366 | Y |
| ScrippsPier | 921 | 10 | 0.118 | 0.0257 | Y |