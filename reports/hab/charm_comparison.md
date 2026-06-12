# DA forecast vs NOAA C-HARM (operational benchmark)

C-HARM dataset: `wvcharmV3_0day (C-HARM v3.1 nowcast, P[particulate DA > 500 ng/L])`. Overlap window 2022-11-01..2026-06-08 (8 stations, 1132 common visits, 45 events).

C-HARM nowcast uses **same-day** satellite+ROMS; our model uses only **prior-visit** data (~1 week old). Both scored on identical rows. AP at a 4.0% event rate.

| model | pooled AP | pooled ROC-AUC |
|---|--:|--:|
| our forecast (prior-visit) | 0.271 | 0.8393 |
| C-HARM v3.1 nowcast (same-day) | 0.0265 | 0.2966 |

C-HARM mean predicted probability = 0.686 vs actual event rate 0.04 (calibration sanity).

**Per-station: we match/beat C-HARM AP in 8/8 stations.**

| station | n | events | our AP | C-HARM AP | our AUC | C-HARM AUC | we win |
|---|--:|--:|--:|--:|--:|--:|---|
| SantaCruzWharf | 171 | 9 | 0.2109 | 0.0478 | 0.7905 | 0.3841 | Y |
| ScrippsPier | 172 | 7 | 0.2778 | 0.0289 | 0.8654 | 0.2416 | Y |
| CalPolyPier | 174 | 6 | 0.1494 | 0.0334 | 0.8328 | 0.4236 | Y |
| TrinidadPier | 46 | 6 | 0.6263 | 0.0902 | 0.9458 | 0.1708 | Y |
| NewportBeachPier | 151 | 5 | 0.145 | 0.0536 | 0.7397 | 0.4164 | Y |
| SantaMonicaPier | 171 | 5 | 0.7143 | 0.0194 | 0.994 | 0.1349 | Y |
| StearnsWharf | 174 | 4 | 0.3129 | 0.0198 | 0.8434 | 0.2368 | Y |
| HumboldtSouthBay | 39 | 3 | 0.1346 | 0.0789 | 0.4398 | 0.3241 | Y |