#!/usr/bin/env bash
set -e

PKG_DIR="/tmp/powgrid-tru-pkg"
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/linux-x64/powgrid-tru-gpu-miner"
mkdir -p "$PKG_DIR/hiveos/powgrid-tru-miner"

BIN="/home/user/git_test/projects/tru/repos/pool/powgrid-tru-gpu-miner/target/release/powgrid-tru-gpu-miner"
cp "$BIN" "$PKG_DIR/linux-x64/powgrid-tru-gpu-miner/"
cp "$BIN" "$PKG_DIR/hiveos/powgrid-tru-miner/"

# Linux start script
cat << 'STARTEOF' > "$PKG_DIR/linux-x64/powgrid-tru-gpu-miner/start_mining.sh"
#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

WALLET="YOUR_TRU_WALLET_ADDRESS"
POOL="wss://tru.powgrid.xyz/stratum"
WORKER="rig-linux1"

echo "===================================================="
echo "  ⚡ POWGRID TRU (\$TRU) HIGH-PERFORMANCE GPU MINER"
echo "  Algorithm: TRUHash (CUDA Dual-Stream)"
echo "===================================================="
echo ""
echo "Starting miner..."
echo "Pool:   $POOL"
echo "Wallet: $WALLET"
echo "Worker: $WORKER"
echo ""

chmod +x ./powgrid-tru-gpu-miner 2>/dev/null || true
./powgrid-tru-gpu-miner --pool "$POOL" --wallet "$WALLET" --worker "$WORKER" "$@"
STARTEOF
chmod +x "$PKG_DIR/linux-x64/powgrid-tru-gpu-miner/start_mining.sh"
cp -f "$PKG_DIR/linux-x64/powgrid-tru-gpu-miner/start_mining.sh" "$PKG_DIR/linux-x64/powgrid-tru-gpu-miner/start-pool.sh"

cat << 'READEEOF' > "$PKG_DIR/linux-x64/powgrid-tru-gpu-miner/README.md"
# PowGrid TRU GPU Miner (TRUHash)
Optimized Blackwell / Ada / Ampere / Turing CUDA Miner with 78-col ANSI TUI & WSS Stratum.
Algorithm: TRUHash

### Quick Start:
```bash
./start_mining.sh
```
Or run directly:
```bash
./powgrid-tru-gpu-miner --pool wss://tru.powgrid.xyz/stratum --wallet YOUR_TRU_ADDRESS --worker rig1
```
READEEOF

# HiveOS files
cat << 'MANIFEOF' > "$PKG_DIR/hiveos/powgrid-tru-miner/h-manifest.conf"
CUSTOM_NAME="powgrid-tru-miner"
CUSTOM_VERSION="1.1"
CUSTOM_BUILD="1"
CUSTOM_LOG_BASENAME="/var/log/miner/$CUSTOM_NAME/$CUSTOM_NAME"
CUSTOM_CONFIG_FILENAME="/hive/miners/custom/$CUSTOM_NAME/powgrid.conf"
CUSTOM_ALGO="truhash"
MANIFEOF

cat << 'CONFEOF' > "$PKG_DIR/hiveos/powgrid-tru-miner/h-config.sh"
#!/usr/bin/env bash
. `dirname $0`/h-manifest.conf

[[ -z $CUSTOM_TEMPLATE ]] && echo "No wallet address specified" && exit 1

POOL_URL="${CUSTOM_URL:-wss://tru.powgrid.xyz/stratum}"

WALLET="$CUSTOM_TEMPLATE"
WORKER="${CUSTOM_WORKER:-%WORKER_NAME%}"

mkdir -p $(dirname "$CUSTOM_CONFIG_FILENAME")

cat <<EOF > "$CUSTOM_CONFIG_FILENAME"
POOL_URL="${POOL_URL}"
WALLET="${WALLET}"
WORKER="${WORKER}"
USER_CONFIG="${CUSTOM_USER_CONFIG}"
EOF

echo "PowGrid TRU HiveOS config generated: $CUSTOM_CONFIG_FILENAME"
CONFEOF
chmod +x "$PKG_DIR/hiveos/powgrid-tru-miner/h-config.sh"

cat << 'RUNEOF' > "$PKG_DIR/hiveos/powgrid-tru-miner/h-run.sh"
#!/usr/bin/env bash
cd `dirname $0`
. h-manifest.conf

[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo "No config file specified in manifest" && exit 1
[[ ! -f $CUSTOM_CONFIG_FILENAME ]] && echo "Config file $CUSTOM_CONFIG_FILENAME not found" && exit 1

. $CUSTOM_CONFIG_FILENAME

mkdir -p /var/log/miner/$CUSTOM_NAME

ARGS="--pool ${POOL_URL} --wallet ${WALLET} --worker ${WORKER} --hiveos"

if [[ ! -z "$USER_CONFIG" ]]; then
    ARGS="$ARGS $USER_CONFIG"
fi

echo "=========================================================="
echo " Starting PowGrid TRU GPU Miner (TRUHash) for HiveOS v${CUSTOM_VERSION}"
echo " Args: $ARGS"
echo "=========================================================="

./powgrid-tru-gpu-miner $ARGS 2>&1 | tee ${CUSTOM_LOG_BASENAME}.log
RUNEOF
chmod +x "$PKG_DIR/hiveos/powgrid-tru-miner/h-run.sh"

cat << 'STATSEOF' > "$PKG_DIR/hiveos/powgrid-tru-miner/h-stats.sh"
#!/usr/bin/env bash
cd `dirname $0`
. h-manifest.conf

STATS_FILE="/tmp/powgrid_tru_miner_stats.json"

if [[ ! -f $STATS_FILE ]]; then
    stats="null"
    khs=0
    return 0 2>/dev/null || exit 0
fi

eval $(python3 -c "
import json
try:
    with open('$STATS_FILE') as f:
        d = json.load(f)
    uptime = d.get('uptime', 0)
    hs = d.get('hashrate_avg', 0)
    acc = d.get('accepted', 0)
    rej = d.get('rejected', 0)
    print(f'UPTIME={uptime}; HS={hs}; ACC={acc}; REJ={rej};')
except:
    print('UPTIME=0; HS=0; ACC=0; REJ=0;')
")

khs=$(python3 -c "print(round(float('$HS') / 1000.0, 2))" 2>/dev/null || echo 0)

gpu_count=$(gpu-detect NVIDIA AMD 2>/dev/null | wc -l)
[[ -z $gpu_count || $gpu_count -eq 0 ]] && gpu_count=1

stats=$(python3 -c "
import json
hs_val = float('$HS')
gpu_cnt = int('$gpu_count')
per_gpu = round(hs_val / max(1, gpu_cnt), 2)
hs_arr = [per_gpu] * gpu_cnt
out = {
    'hs': hs_arr,
    'hs_units': 'hs',
    'uptime': int('$UPTIME'),
    'ar': [int('$ACC'), int('$REJ')],
    'algo': 'truhash'
}
print(json.dumps(out))
")
STATSEOF
chmod +x "$PKG_DIR/hiveos/powgrid-tru-miner/h-stats.sh"

# Create archives
DIST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dist"
mkdir -p "$DIST_DIR"

cd "$PKG_DIR/linux-x64"
tar -czvf "$DIST_DIR/powgrid-tru-gpu-miner-linux-x64.tar.gz" powgrid-tru-gpu-miner
cp -f "$DIST_DIR/powgrid-tru-gpu-miner-linux-x64.tar.gz" /tmp/

cd "$PKG_DIR/hiveos"
tar -czvf "$DIST_DIR/powgrid-tru-gpu-miner-hiveos.tar.gz" powgrid-tru-miner
cp -f "$DIST_DIR/powgrid-tru-gpu-miner-hiveos.tar.gz" /tmp/

echo ""
echo "=== Packaging Complete ==="
ls -lh "$DIST_DIR"/powgrid-tru-gpu-miner-*.tar.gz

