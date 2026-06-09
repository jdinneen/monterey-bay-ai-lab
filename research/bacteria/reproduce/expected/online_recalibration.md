# Era-local (prequential) recalibration vs static

- lag_days=2 | block=M

| stratum | n | events | static ECE | **online ECE** | memory ECE | static deploy | **online deploy** |
|---|--:|--:|--:|--:|--:|:-:|:-:|
| ALL | 118247 | 26659 | 0.0687 | 0.0138 | 0.0883 | Y | Y |
| EXCLUDE_SAN_DIEGO | 91825 | 10626 | 0.0088 | 0.0327 | 0.0082 | Y | Y |
| SAN_DIEGO | 26422 | 16033 | 0.2852 | 0.1735 | 0.3696 | Y | Y |
| SAN_FRANCISCO | 4256 | 656 | 0.0783 | 0.1454 | 0.0196 | N | N |
| MONTEREY | 5298 | 534 | 0.0235 | 0.0589 | 0.0118 | Y | Y |