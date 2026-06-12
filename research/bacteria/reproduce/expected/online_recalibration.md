# Era-local (prequential) recalibration vs static

- lag_days=2 | block=M

| stratum | n | events | static ECE | **online ECE** | memory ECE | static deploy | **online deploy** |
|---|--:|--:|--:|--:|--:|:-:|:-:|
| ALL | 118247 | 26659 | 0.0795 | 0.0168 | 0.0884 | Y | Y |
| EXCLUDE_SAN_DIEGO | 91825 | 10626 | 0.0117 | 0.0307 | 0.0082 | Y | Y |
| SAN_DIEGO | 26422 | 16033 | 0.3177 | 0.1815 | 0.3699 | Y | Y |
| SAN_FRANCISCO | 4256 | 656 | 0.0103 | 0.0652 | 0.0201 | Y | Y |
| MONTEREY | 5298 | 534 | 0.0183 | 0.0585 | 0.0118 | Y | Y |