#!/usr/bin/env bash
set -euo pipefail

fail() { echo "Birdly install failed: $*" >&2; exit 1; }
[[ $EUID -eq 0 ]] || fail "run sudo ./install.sh"
[[ $(uname -s) == Linux ]] || fail "Linux is required"
[[ $(uname -m) == aarch64 ]] || fail "a 64-bit ARM operating system is required"
[[ -r /etc/os-release ]] || fail "cannot identify the operating system"
. /etc/os-release
[[ ${ID:-} == raspbian || ${ID:-} == debian ]] || fail "Raspberry Pi OS 64-bit or Debian ARM64 is required"
command -v python3 >/dev/null || fail "python3 is required"
command -v systemctl >/dev/null || fail "systemd is required"
command -v sha256sum >/dev/null || fail "sha256sum is required"
python3 - <<'PY' || fail "Python 3.11 or 3.12 is required"
import sys
assert sys.version_info[:2] in ((3, 11), (3, 12))
PY
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$source_dir"
sha256sum -c SHA256SUMS || fail "package model or label checksum mismatch"
available_kib=$(df -Pk /opt | awk 'NR==2 {print $4}')
[[ $available_kib -ge 800000 ]] || fail "at least 800 MB of free disk space is required under /opt"
echo "Available disk: $((available_kib / 1024)) MiB"
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
if ! id birdly >/dev/null 2>&1; then
    useradd --system --user-group --home-dir /opt/birdly-pi --shell /usr/sbin/nologin birdly
fi
install -d -m 0755 -o birdly -g birdly /opt/birdly-pi
for directory in app model scripts systemd; do
    cp -a "$directory" /opt/birdly-pi/
done
cp -a requirements.txt SHA256SUMS README.md /opt/birdly-pi/
cd /opt/birdly-pi
sha256sum -c SHA256SUMS || fail "installed model or label checksum mismatch"
chown -R birdly:birdly /opt/birdly-pi
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv || fail "python3-venv is required"
fi
.venv/bin/python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt || fail "an ARM64 dependency wheel could not be installed"
.venv/bin/python - <<'PY' || fail "ONNX Runtime CPU is unavailable"
import onnxruntime as ort
assert 'CPUExecutionProvider' in ort.get_available_providers()
PY
if [[ ! -e /etc/birdly-pi.env ]]; then
    printf '%s\n' 'BIRDLY_BIND_ADDRESS=0.0.0.0' 'BIRDLY_PORT=8080' 'BIRDLY_CPU_THREADS=2' 'BIRDLY_MAX_INFERENCE_CONCURRENCY=1' > /etc/birdly-pi.env
    chmod 0644 /etc/birdly-pi.env
fi
install -m 0644 systemd/birdly-pi.service /etc/systemd/system/birdly-pi.service
systemctl daemon-reload
systemctl enable birdly-pi
systemctl restart birdly-pi
set -a
. /etc/birdly-pi.env
set +a
ready=0
for attempt in $(seq 1 60); do
    if systemctl is-active --quiet birdly-pi && .venv/bin/python - "$BIRDLY_PORT" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/ready', timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then ready=1; break; fi
    sleep 1
done
[[ $ready -eq 1 ]] || fail "service is not ready; run journalctl -u birdly-pi -n 100"
echo "Birdly ready at http://${BIRDLY_BIND_ADDRESS}:${BIRDLY_PORT}"
systemctl --no-pager status birdly-pi | head -n 12
echo "Diagnostics: sudo journalctl -u birdly-pi -n 100"
