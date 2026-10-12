# Birdly Raspberry Pi 5 package

This package runs the existing ConvNeXt-Tiny ONNX FP32 model on the CPU. It uses one Uvicorn worker. The response contains `taxon_id`, `common_name`, `scientific_name`, and `score`. Confidence filtering can return zero to five predictions.

## Build and transfer

From the repository root, use Git Bash or another Bash installation:

```bash
bash scripts/build_pi_package.sh
```

The build uses `deploy/cloud-run/build/model` by default. Use `--model-source PATH` to select another complete production ONNX bundle. The output is `dist/birdly-pi5-arm64.tar.gz`.

From Windows PowerShell:

```powershell
scp -i "$env:USERPROFILE\.ssh\ovh-elixir" dist\birdly-pi5-arm64.tar.gz <pi-user>@<pi-ip>:~/
```

## Install

On the Pi:

```bash
tar -xzf birdly-pi5-arm64.tar.gz
cd birdly-pi5
sudo ./install.sh
```

The installer requires Python 3.11, 3.12, or 3.13. It checks the model files, installs under `/opt/birdly-pi`, creates `/etc/birdly-pi.env` only if absent, and starts `birdly-pi.service`. Reinstall with a new archive and the same commands. The installer preserves the environment file.

## Check and test

```bash
sudo systemctl status birdly-pi
sudo journalctl -u birdly-pi -f
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/ready
curl -F 'image=@/path/to/bird.jpg;type=image/jpeg' http://127.0.0.1:8080/predict
/opt/birdly-pi/.venv/bin/python /opt/birdly-pi/scripts/smoke_test.py /path/to/bird.jpg
/opt/birdly-pi/.venv/bin/python /opt/birdly-pi/scripts/benchmark.py /path/to/bird.jpg --requests 20 --warmup 2 --concurrency 1
```

The benchmark measures HTTP requests. It does not report model-only latency. It reports process peak memory, device memory, CPU use, temperature, and throttling when the Pi provides those values. The normal rate limit is 15 requests per minute. Set `BIRDLY_RATE_LIMIT_PER_MINUTE=0` in `/etc/birdly-pi.env` and restart the service before a longer benchmark. Restore the limit after the test.

## Configuration

Edit `/etc/birdly-pi.env`, then run `sudo systemctl restart birdly-pi`. Defaults are `BIRDLY_BIND_ADDRESS=0.0.0.0`, `BIRDLY_PORT=8080`, `BIRDLY_CPU_THREADS=2`, and `BIRDLY_MAX_INFERENCE_CONCURRENCY=1`. The API also accepts `BIRDLY_RATE_LIMIT_PER_MINUTE`, `BIRDLY_MAX_UPLOAD_BYTES`, `BIRDLY_MAX_IMAGE_WIDTH`, `BIRDLY_MAX_IMAGE_HEIGHT`, and `BIRDLY_REQUEST_TIMEOUT_SECONDS`. Use `BIRDLY_CORS_ORIGINS` for explicit browser origins.

Use `sudo systemctl restart birdly-pi` or `sudo systemctl stop birdly-pi` to control the service. For uninstall, stop and disable the service, remove `/etc/systemd/system/birdly-pi.service` and `/opt/birdly-pi`, then run `sudo systemctl daemon-reload`. Remove `/etc/birdly-pi.env` only if you no longer need its settings.

The default bind address allows other devices on the trusted local network to connect to port 8080. The installer does not configure a firewall or Internet access. Restrict network access at your router or firewall. If memory is low, keep concurrency at 1, close other processes, and inspect `journalctl -u birdly-pi` and `/proc/meminfo`. Do not add swap without checking storage wear and system requirements.
