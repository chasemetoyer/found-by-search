"""Record which kernel instruction sets PyTorch and its libraries actually use.

Run in the same environment as training (including any ISA pins). This makes a
non-identical determinism result interpretable: it separates "pins ignored"
from "pins insufficient". Each verbose probe runs in its own short subprocess so
verbose logging never reaches training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PIN_VARIABLES = ("ATEN_CPU_CAPABILITY", "MKL_CBWR", "ONEDNN_MAX_CPU_ISA",
                 "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# Shapes match the archived network: batched 2x2 convolutions (oneDNN path in
# torch 2.14.1 for batch > 1 and non-1x1 kernels) and linear layers (BLAS path).
CONV_CODE = ("import torch; torch.manual_seed(0); x = torch.randn(16, 3, 7, 7); "
             "c = torch.nn.Conv2d(3, 16, 2); y = c(x); y.sum().backward()")
GEMM_CODE = ("import torch; torch.manual_seed(0); a = torch.randn(512, 64); "
             "l = torch.nn.Linear(64, 64); y = l(a); y.sum().backward()")


def verbose_lines(code: str, variable: str, keep) -> dict:
    proc = subprocess.run([sys.executable, "-c", code], env={**os.environ, variable: "1"},
                          capture_output=True, text=True, timeout=300)
    lines = (proc.stdout + "\n" + proc.stderr).splitlines()
    return {"returncode": proc.returncode, "lines": [line for line in lines if keep(line)][:12]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    config = torch.__config__.show()
    answer = {
        "environment": {name: os.environ.get(name) for name in PIN_VARIABLES},
        "torch_version": torch.__version__,
        "aten_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkl_available": torch.backends.mkl.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "torch_config_flags": [line.strip() for line in config.splitlines()
                               if any(t in line for t in ("BLAS_INFO", "LAPACK_INFO", "USE_MKL", "USE_MKLDNN",
                                                          "Math Kernel", "oneDNN", "MKL-DNN", "CPU capability"))],
        "torch_num_threads": torch.get_num_threads(),
        "onednn_verbose": verbose_lines(CONV_CODE, "ONEDNN_VERBOSE",
                                        lambda l: l.startswith(("onednn_verbose", "dnnl_verbose"))
                                        and (",info," in l or ",exec," in l)),
        "mkl_verbose": verbose_lines(GEMM_CODE, "MKL_VERBOSE", lambda l: l.startswith("MKL_VERBOSE")),
    }
    args.out.write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n")
    print("ISA_PROBE " + json.dumps({k: answer[k] for k in ("environment", "aten_cpu_capability",
                                                          "mkl_available", "mkldnn_available")}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
