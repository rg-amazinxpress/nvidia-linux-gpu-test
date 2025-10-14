#!/bin/bash
set -e

# --- Config ---
export RMA_ID="RMA_#"
export RESULTS="/mnt/rma_usb/results/RMA_${RMA_ID}"
mkdir -p "$RESULTS"

# --- System Info ---
uname -a | tee "$RESULTS/uname.log"
lsb_release -a 2>/dev/null | tee "$RESULTS/lsb_release.log"
lspci -vv | tee "$RESULTS/lspci_full.log"
lspci | grep -i nvidia | tee "$RESULTS/lspci_nvidia.log"
dmesg | grep -Ei "pci|pcie" | tee "$RESULTS/dmesg_pci.log"

# --- Quick Fail if No GPU ---
if ! lspci | grep -qi nvidia; then
  echo "FAIL: No NVIDIA device enumerated" | tee "$RESULTS/summary.txt"
  exit 2
fi

# --- Install Tools ---
sudo apt update
sudo apt install -y nvidia-cuda-toolkit git build-essential

# --- CUDA Functional Tests ---
/usr/lib/nvidia-cuda-toolkit/bin/deviceQuery | tee "$RESULTS/deviceQuery.log"
/usr/lib/nvidia-cuda-toolkit/bin/bandwidthTest | tee "$RESULTS/bandwidthTest.log"

# --- Build gpu-burn ---
git clone https://github.com/wilicc/gpu-burn.git "$RESULTS/gpu-burn-src"
cd "$RESULTS/gpu-burn-src"
make

# --- Telemetry Collector ---
nvidia-smi --query-gpu=index,name,serial,temperature.gpu,temperature.memory,power.draw,utilization.gpu,utilization.memory,ecc.errors.uncorrected --format=csv -l 1 | tee "$RESULTS/telemetry.csv" &
TELEMETRY_PID=$!

# --- Stress Test (Compute + VRAM) ---
./gpu_burn 300 | tee "$RESULTS/gpu-burn.log"

# --- Stop Telemetry ---
kill "$TELEMETRY_PID" || true

# --- VRAM Error Check ---
# Query ECC and memory error counters explicitly
nvidia-smi -q -d MEMORY,ECC,ERROR | tee "$RESULTS/nvidia-smi_vram.log"

# Run a CUDA malloc/free loop to exercise VRAM allocation
cat > "$RESULTS/vram_test.cu" <<'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    const size_t MB = 256;
    const int iters = 50;
    for (int i=0; i<iters; i++) {
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, MB*1024*1024);
        if (err != cudaSuccess) {
            printf("cudaMalloc failed at iter %d: %s\n", i, cudaGetErrorString(err));
            return 1;
        }
        cudaMemset(ptr, 0xA5, MB*1024*1024);
        cudaFree(ptr);
    }
    printf("VRAM allocation/free test completed successfully.\n");
    return 0;
}
EOF
nvcc "$RESULTS/vram_test.cu" -o "$RESULTS/vram_test"
"$RESULTS/vram_test" | tee "$RESULTS/vram_test.log"

# --- Final Diagnostics ---
nvidia-smi -q -d MEMORY,POWER,UTILIZATION,PCI,ERROR | tee "$RESULTS/nvidia-smi_detailed.log"
dmesg | grep -Ei "nvidia|drm|pcie|gpu|fault|error|panic|Xid" | tee "$RESULTS/dmesg_errors.log"

# --- Verdict ---
VERDICT="PASS"
grep -qi "uncorrect" "$RESULTS/nvidia-smi_vram.log" && VERDICT="FAIL_VRAM_ECC"
grep -qiE "cudaMalloc failed" "$RESULTS/vram_test.log" && VERDICT="FAIL_VRAM_ALLOC"
grep -qiE "GPU has fallen|Xid|panic" "$RESULTS/dmesg_errors.log" && VERDICT="FAIL_BUS_OR_KERNEL"
echo "VERDICT=${VERDICT}" | tee "$RESULTS/summary.txt"

# --- Bundle Evidence ---
tar -czf "$RESULTS/RMA_${RMA_ID}_evidence.tgz" -C "$RESULTS" .
echo "Evidence saved: $RESULTS/RMA_${RMA_ID}_evidence.tgz"