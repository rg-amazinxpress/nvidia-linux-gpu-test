# NVIDIA Linux GPU Test Suite

A comprehensive GPU testing framework designed for **RMA (Return Merchandise Authorization) testing** and GPU health validation in Ubuntu Linux environments. This tool creates a complete evidence package to prove GPU defects when returning hardware to manufacturers.

## Overview

This test suite performs comprehensive GPU diagnostics including:
- **System Information Collection**: Hardware enumeration, PCI device detection, and kernel logs
- **CUDA Functionality Tests**: Device query and bandwidth testing (requires proprietary NVIDIA drivers)
- **VRAM Stress Testing**: 256MB memory allocation/deallocation cycles with pattern testing (0xA5 pattern)
- **Thermal and Power Monitoring**: Real-time telemetry collection during 5-minute stress tests
- **Error Detection**: ECC error monitoring, kernel error analysis, and allocation failure detection
- **Automated Evidence Collection**: All test results bundled into compressed archive for RMA submission

## Important Notes

⚠️ **Driver Requirements**: This test requires **proprietary NVIDIA drivers** to function properly. The open-source Nouveau driver does not support CUDA functionality and will cause test failures.

⚠️ **RMA Focus**: This tool is specifically designed for hardware RMA testing - it creates comprehensive evidence packages to prove GPU defects.

## Prerequisites

- **Operating System**: Ubuntu Linux (stable distribution recommended)
- **Hardware**: NVIDIA GPU with CUDA support
- **Dependencies**: 
  - **Proprietary NVIDIA drivers** (NOT Nouveau open-source driver)
  - CUDA toolkit
  - Build tools (gcc, make)
  - Git
  - USB drive mounted at `/mnt/rma_usb/` (for evidence storage)

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/nvidia-linux-gpu-test.git
cd nvidia-linux-gpu-test
```

### 2. Make the Script Executable
```bash
chmod +x gpu-test.sh
```

### 3. Run the Test Suite
```bash
sudo ./gpu-test.sh
```

## What the Test Does

### System Information Collection
- Captures system information (`uname`, `lsb_release`)
- Enumerates PCI devices and identifies NVIDIA hardware
- Collects kernel logs related to PCI/GPU initialization
- **Output**: `uname.log`, `lsb_release.log`, `lspci_*.log`, `dmesg_pci.log`

### CUDA Functionality Tests
- **Device Query**: Verifies CUDA device detection and basic properties
- **Bandwidth Test**: Tests memory bandwidth between host and GPU
- **Note**: These tests will fail if using Nouveau driver instead of proprietary NVIDIA drivers
- **Output**: `deviceQuery.log`, `bandwidthTest.log`

### VRAM Stress Testing
- **Custom VRAM Test**: Allocates 256MB chunks in a loop (50 iterations)
- **Pattern Testing**: Writes specific patterns (0xA5) to verify memory integrity
- **Allocation/Deallocation**: Stress tests memory management
- **Output**: `vram_test.log`, `vram_test.cu` (source code)

### Thermal and Power Monitoring
- Real-time telemetry collection during 5-minute GPU stress test
- Monitors GPU temperature, power draw, utilization, and ECC error counters
- **Output**: `telemetry.csv` (CSV format with timestamps)

### Error Detection and Analysis
- **ECC Error Monitoring**: Detects uncorrected memory errors
- **Kernel Error Analysis**: Scans for GPU-related kernel errors (Xid, panic, etc.)
- **Allocation Failure Detection**: Identifies VRAM allocation issues
- **Output**: `nvidia-smi_vram.log`, `dmesg_errors.log`, `nvidia-smi_detailed.log`

### Evidence Collection
- **Automated Bundling**: Creates compressed archive with all test results
- **RMA Package**: Named `RMA_{ID}_evidence.tgz` for manufacturer submission
- **Complete Documentation**: All logs, telemetry, and diagnostic data included

## Test Results

The test generates comprehensive logs in the results directory:

- `summary.txt` - Final test verdict
- `deviceQuery.log` - CUDA device information
- `bandwidthTest.log` - Memory bandwidth test results
- `gpu-burn.log` - GPU stress test output
- `vram_test.log` - VRAM allocation test results
- `telemetry.csv` - Real-time monitoring data
- `nvidia-smi_*.log` - Detailed GPU status information
- `dmesg_*.log` - Kernel log analysis
- `RMA_*_evidence.tgz` - Complete evidence bundle

## Test Verdicts

The test can return one of the following verdicts:

- **PASS**: All tests completed successfully
- **FAIL_VRAM_ECC**: Uncorrected ECC errors detected
- **FAIL_VRAM_ALLOC**: VRAM allocation failures (often indicates driver issues)
- **FAIL_BUS_OR_KERNEL**: GPU bus errors or kernel panics

**Note**: The current test results show `FAIL_VRAM_ALLOC` because the system is using the Nouveau open-source driver instead of proprietary NVIDIA drivers, which prevents CUDA functionality.

## Configuration

You can modify the test parameters by editing `gpu-test.sh`:

- **RMA_ID**: Change the RMA identifier (default: "26169")
- **Results Directory**: Modify the output path (default: "/mnt/rma_usb/results/")
- **VRAM Test Parameters**: Adjust memory size and iteration count
- **Stress Test Duration**: Modify GPU burn test duration (default: 300 seconds)

## Example Usage

```bash
# Basic test run
sudo ./gpu-test.sh

# Check results
ls -la results/NVIDIA_RTX_A6000/

# View summary
cat results/NVIDIA_RTX_A6000/summary.txt

# Extract evidence bundle
tar -xzf results/NVIDIA_RTX_A6000/RMA_26169_evidence.tgz
```

## Troubleshooting

### No NVIDIA Device Detected
- Ensure NVIDIA drivers are properly installed
- Verify GPU is properly seated and powered
- Check PCI device enumeration

### CUDA Toolkit Issues
- Install NVIDIA CUDA toolkit: `sudo apt install nvidia-cuda-toolkit`
- Verify CUDA installation: `nvcc --version`

### Driver Issues (Most Common)
- **Problem**: Test fails with "no CUDA-capable device is detected"
- **Cause**: Using Nouveau open-source driver instead of proprietary NVIDIA drivers
- **Solution**: Install proprietary NVIDIA drivers:
  ```bash
  sudo apt update
  sudo apt install nvidia-driver-535  # or latest version
  sudo reboot
  ```

### Permission Issues
- Run with `sudo` for system-level access
- Ensure user has access to GPU devices

### USB Mount Issues
- Ensure USB drive is mounted at `/mnt/rma_usb/`
- Create directory if needed: `sudo mkdir -p /mnt/rma_usb/results`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Disclaimer

This tool is designed for hardware testing and validation purposes. Use at your own risk and ensure proper system backups before running intensive GPU tests.
