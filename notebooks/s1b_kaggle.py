# syndrome-stream S1b on Kaggle: sliding-window Relay-BP on IBM's [[144,12,12]] gross code.
# Notebook settings: Accelerator "GPU T4 x2", Internet on. Paste this file into one cell, or
# Save Version -> Save & Run All. The results print at the end and are saved as JSON.
import os
import subprocess
import sys

W = "/kaggle/working"
os.chdir(W)


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


sh("rm -rf syndrome syndrome-stream relay && "
   "git clone -q https://github.com/Shakhtar-Sankur/syndrome && "
   "git clone -q https://github.com/Shakhtar-Sankur/syndrome-stream && "
   "git clone -q --depth 1 --filter=blob:none https://github.com/trmue/relay")
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
sh(f"{sys.executable} -m pip install -q stim==1.16.0 scipy")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
cc = subprocess.run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", shell=True,
                    capture_output=True, text=True).stdout.split()[0].replace(".", "")
# syndrome's Relay-BP kernel: same decodings as IBM's reference (-fmad=false keeps IBM's rounding)
sh(f"cd syndrome/csrc && nvcc -O3 -fmad=false -std=c++17 -arch=sm_{cc} -Xcompiler -fPIC -shared "
   "-o libsyndrome_cuda.so relay_bp_cuda.cu")
for p in ("0.006",):
    sh(f"set -o pipefail; {sys.executable} syndrome-stream/bench/s1b_gross_windows.py --syndrome syndrome "
       f"--relay relay --p {p} --shots 2000 --windows whole,6:3,4:2,3:1 --backend cuda "
       f"--out {W}/s1b_gross_p{p}.json 2>&1 | tee {W}/s1b_gross_p{p}.log")
sh(f"cat {W}/s1b_gross_p*.json")
