# Environment

Captured by `src/env_info.py`. The untrained arm depends on the torch RNG,
which is version-dependent, so a different torch version can change every
untrained-arm number even with the same seed.

| component | version |
|---|---|
| python | `3.8.20` |
| platform | `macOS-26.5.2-arm64-arm-64bit` |
| machine | `arm64` |
| processor | `arm` |
| numpy | `1.23.5` |
| scipy | `1.9.1` |
| torch | `2.2.2` |
| torch_threads | `12` |

Regenerate with `python src/env_info.py`.
