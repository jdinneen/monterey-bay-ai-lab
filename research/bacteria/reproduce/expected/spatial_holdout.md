# Leave-one-county-out spatial generalization

- rainfall used: True | min test events/county: 150
- counties held out: 9 | median calibrated model AP: 0.5222
- model beats AB411 rule: 9/9 counties | beats station-memory: 9/9 | deploy-ready: 8/9

| held-out county | n | events | base | model AP | model AUC | AB411 AP | memory AP | beats AB411 | beats memory | deploy |
|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|:-:|
| San Diego | 26326 | 15918 | 0.6046 | 0.9492 | 0.9494 | 0.6022 | 0.9389 | Y | Y | N |
| Los Angeles | 22612 | 3588 | 0.1587 | 0.552 | 0.8444 | 0.2514 | 0.313 | Y | Y | Y |
| Orange | 29728 | 1709 | 0.0575 | 0.3508 | 0.823 | 0.1037 | 0.1943 | Y | Y | Y |
| San Mateo | 5308 | 983 | 0.1852 | 0.5222 | 0.8256 | 0.2412 | 0.4176 | Y | Y | Y |
| Long Beach City | 4014 | 704 | 0.1754 | 0.5851 | 0.7696 | 0.3691 | 0.1827 | Y | Y | Y |
| San Francisco | 4256 | 607 | 0.1426 | 0.55 | 0.8682 | 0.2732 | 0.2604 | Y | Y | Y |
| Santa Cruz | 3931 | 415 | 0.1056 | 0.4141 | 0.7841 | 0.207 | 0.1631 | Y | Y | Y |
| Ventura | 5543 | 261 | 0.0471 | 0.4504 | 0.8833 | 0.1485 | 0.097 | Y | Y | Y |
| Santa Barbara | 3130 | 175 | 0.0559 | 0.2812 | 0.8179 | 0.099 | 0.1106 | Y | Y | Y |