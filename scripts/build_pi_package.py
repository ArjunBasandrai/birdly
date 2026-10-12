from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from prepare_cloud_run import _validate_model_bundle, _validate_taxon_names

ROOT = Path(__file__).resolve().parents[1]
PI = ROOT / "deploy" / "pi5"
DEFAULT_MODEL = ROOT / "deploy" / "cloud-run" / "build" / "model"
ARCHIVE = ROOT / "dist" / "birdly-pi5-arm64.tar.gz"
APP_FILES = (
    "src/birdly/__init__.py",
    "src/birdly/api/__init__.py",
    "src/birdly/api/app.py",
    "src/birdly/api/config.py",
    "src/birdly/inference/__init__.py",
    "src/birdly/inference/onnx_backend.py",
    "src/birdly/inference/predict.py",
    "src/birdly/inference/taxon_names.json",
)
MODEL_FILES = (
    "class_mapping.json",
    "preprocessing.json",
    "onnx/birdly-fp32.onnx",
    "onnx/metadata.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(model_source: Path, archive: Path = ARCHIVE) -> Path:
    model_source = model_source.resolve()
    _validate_model_bundle(model_source)
    mapping = json.loads(
        (model_source / "class_mapping.json").read_text(encoding="utf-8")
    )
    _validate_taxon_names(mapping, ROOT / "src/birdly/inference/taxon_names.json")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="birdly-pi5-") as temporary:
        package = Path(temporary) / "birdly-pi5"
        package.mkdir()
        for relative in APP_FILES:
            destination = package / "app" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        for relative in MODEL_FILES:
            destination = package / "model" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_source / relative, destination)
        package_files = {
            "install.sh": PI / "install.sh",
            "requirements.txt": PI / "requirements.txt",
            "app/main.py": PI / "main.py",
            "app/api.yaml": PI / "api.yaml",
            "systemd/birdly-pi.service": PI / "birdly-pi.service",
            "scripts/smoke_test.py": PI / "smoke_test.py",
            "scripts/benchmark.py": PI / "benchmark.py",
            "README.md": PI / "README.md",
        }
        for relative, source in package_files.items():
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative == "install.sh":
                destination.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
            else:
                shutil.copy2(source, destination)
        checksummed = [
            *(package / "model").rglob("*"),
            package / "app/src/birdly/inference/taxon_names.json",
        ]
        lines = [
            f"{sha256(path)}  {path.relative_to(package).as_posix()}"
            for path in checksummed
            if path.is_file()
        ]
        (package / "SHA256SUMS").write_text(
            "\n".join(sorted(lines)) + "\n", encoding="utf-8"
        )
        with tarfile.open(archive, "w:gz") as output:
            for path in [package, *package.rglob("*")]:
                relative = path.relative_to(package)
                info = output.gettarinfo(
                    str(path), arcname=str(Path("birdly-pi5") / relative)
                )
                if relative.as_posix() == "install.sh":
                    info.mode = 0o755
                if path.is_file():
                    with path.open("rb") as stream:
                        output.addfile(info, stream)
                else:
                    output.addfile(info)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL)
    arguments = parser.parse_args()
    result = build(arguments.model_source)
    print(f"{result} ({result.stat().st_size} bytes)")
