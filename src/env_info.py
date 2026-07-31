"""Environment capture.

The untrained arm's weights come from `torch.manual_seed` + `TextEncoder` construction, so
they depend on the torch RNG, which is version-dependent. That version was never recorded
for the original runs. Every entry point prints this banner so future runs are pinned.
"""

import platform
import sys


def env_dict():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "(unknown)",
    }
    for mod in ("numpy", "scipy", "torch"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as exc:                      # noqa: BLE001
            info[mod] = f"(unavailable: {exc.__class__.__name__})"
    try:
        import torch
        info["torch_threads"] = str(torch.get_num_threads())
    except Exception:                                  # noqa: BLE001
        pass
    return info


def banner(tag=""):
    d = env_dict()
    return ("[env] " + (f"{tag} | " if tag else "")
            + " ".join(f"{k}={v}" for k, v in d.items()))


def print_banner(tag=""):
    print(banner(tag), flush=True)


def write_markdown(path="results/ENVIRONMENT.md"):
    d = env_dict()
    with open(path, "w") as f:
        f.write("# Environment\n\n")
        f.write("Captured by `src/env_info.py`. The untrained arm depends on the torch RNG,\n"
                "which is version-dependent, so a different torch version can change every\n"
                "untrained-arm number even with the same seed.\n\n")
        f.write("| component | version |\n|---|---|\n")
        for k, v in d.items():
            f.write(f"| {k} | `{v}` |\n")
        f.write("\nRegenerate with `python src/env_info.py`.\n")
    return path


if __name__ == "__main__":
    print_banner()
    print("wrote", write_markdown())
