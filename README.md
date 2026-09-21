# PowGrid TRU GPU Miner (`powgrid-tru-gpu-miner`)

High-performance, multi-architecture NVIDIA CUDA GPU miner for the **TRU ($TRU)** blockchain network, engineered by **PoWGrid**.

---

## Highlights & Features

- **High-Performance TRUHash Engine**:
  - Native dual-stream asynchronous CUDA pipeline with zero GPU idle time between batches.
  - Optimized double-SHA256 compression kernel with unrolled message schedules, pre-calculated midstates, and round 60 early-abort threshold filtering.
- **Next-Gen Architecture Support**:
  - Full native compute capability targeting:
    - **Blackwell** (`sm_120`, `compute_120`) – RTX 5090, 5080, 5070, 5060 series.
    - **Ada Lovelace** (`sm_89`) – RTX 4090, 4080, 4070 series.
    - **Ampere** (`sm_80`, `sm_86`) – RTX 3090, 3080, 3070, A100 series.
    - **Turing** (`sm_75`) – RTX 2080, 2070, 2060, GTX 1660 series.
- **Stratum over WebSocket (WSS)**:
  - Low-latency real-time bidirectional Stratum mining protocol over encrypted WebSockets (`wss://`).
  - Instant job dispatch upon new block discovery with zero polling delay.
  - Automatic reconnection and heartbeat keep-alive mechanism.
- **Professional ANSI TUI Dashboard**:
  - Crisp 78-column real-time console user interface.
  - Live per-GPU hashrate, rolling average, share acceptance/rejection metrics, pool latency, temperature, fan speed, power usage, and electrical efficiency (MH/W).
- **Hardware Power Control**:
  - Built-in power capping (`--watt <WATTS>`) to protect thermal margins and maintain optimal efficiency without external tools.
- **HiveOS Integration**:
  - Native scripts and manifest for custom HiveOS package installation (`h-manifest.conf`, `h-config.sh`, `h-run.sh`, `h-stats.sh`).

---

## Requirements

- **GPU**: NVIDIA GPU with Compute Capability >= 7.5 (Turing or newer).
- **Driver**: NVIDIA Linux Driver >= 550.xx (or Windows display driver with CUDA 12/13 support).
- **CUDA**: CUDA Toolkit 12.x or 13.x installed (for building from source).
- **OS**: Linux (x86_64, Ubuntu 20.04+, Debian 11+, HiveOS) or Windows 10/11 (64-bit).

---

## Quick Start

### Basic CLI Usage

```bash
./powgrid-tru-gpu-miner --pool wss://tru.powgrid.xyz/stratum --wallet <YOUR_TRU_WALLET_ADDRESS> --worker rig1
```

### With Power Limit & Tuning Preset

```bash
# Run with extreme tuning preset and 280W power cap
./powgrid-tru-gpu-miner \
  --pool wss://tru.powgrid.xyz/stratum \
  --wallet <YOUR_TRU_WALLET_ADDRESS> \
  --worker rig1 \
  --preset extreme \
  --watt 280
```

---

## CLI Options Reference

| Flag | Argument | Description | Default |
|---|---|---|---|
| `--pool`, `-o` | `<URL>` | Mining pool Stratum URL (WSS or HTTP) | `wss://tru.powgrid.xyz/stratum` |
| `--wallet`, `-u` | `<ADDR>` | TRU payout wallet address (**required**) | None |
| `--worker`, `-w` | `<NAME>` | Worker / rig identifier | `rig-1` |
| `--device` | `<ID>` | CUDA device index | `0` |
| `--preset` | `<NAME>` | Performance profile: `auto`, `extreme`, `max`, `balanced` | `auto` |
| `--batch-size` | `<NUM>` | Custom nonce batch size (e.g. `33554432`) | Dynamic per GPU arch |
| `--watt`, `--power-limit` | `<NUM>` | GPU power consumption limit in Watts | None (hardware default) |
| `--http` | None | Force HTTP REST long-polling instead of WSS | Disabled |
| `--hiveos` | None | Headless logging mode tailored for HiveOS agent | Disabled |
| `--help`, `-h` | None | Show usage instructions and exit | |

---

## Tuning Presets

- **`auto`**: Dynamically queries GPU compute capability, SM count, and VRAM bandwidth to assign the optimal batch size and thread grid.
- **`extreme`**: Maximizes occupancy (33.5M+ nonces per batch) for flagship desktop GPUs (RTX 5090, 4090) with high memory throughput.
- **`max`**: High batch size tuned for performance rigs with good cooling.
- **`balanced`**: Moderate batch size for laptops or rigs operating under thermal or acoustic constraints.

---

## Building from Source

### Prerequisites

1. Install Rust and Cargo (version 1.75+):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. Install NVIDIA CUDA Toolkit (12.x or 13.x) with `nvcc` on your `PATH`:
   ```bash
   nvcc --version
   ```

### Compile Release Binary

```bash
git clone https://github.com/PoWGrid/powgrid-tru-gpu-miner.git
cd powgrid-tru-gpu-miner
cargo build --release
```

The compiled standalone executable will be located at:
```bash
./target/release/powgrid-tru-gpu-miner
```

---

## HiveOS Custom Miner Integration

To package for HiveOS:
```bash
chmod +x package.sh
./package.sh
```

In HiveOS Flight Sheet:
1. Select Coin: **TRU**
2. Select Wallet: Your TRU Wallet
3. Pool: **Configure in miner**
4. Miner: **Custom**
5. Setup Miner Config:
   - **Miner name**: `powgrid-tru-miner`
   - **Installation URL**: URL hosting `powgrid-tru-gpu-miner-hiveos.tar.gz`
   - **Hash algorithm**: `truhash`
   - **Pool URL**: `wss://tru.powgrid.xyz/stratum`
   - **Pass**: `x`

---

## Security & Transparency

- **100% Native Code**: Written in pure Rust and CUDA C++ with zero external telemetry, hidden fees, or developer fees.
- **Ephemeral Credentials**: The miner requires your own TRU wallet address passed via command-line arguments and never stores or transmits credentials.

---

## License

This project is licensed under the [MIT License](LICENSE).
