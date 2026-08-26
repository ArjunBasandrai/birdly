from pathlib import Path
import yaml

def init_birdly(data_root: str) -> None:
    birdly_dir = Path(".birdly")
    birdly_dir.mkdir(parents=True, exist_ok=True)

    config_file = birdly_dir / "local.yaml"
    config = {
        "raw_data_root": str(data_root)
    }
    with open(config_file, "w") as f:
        yaml.dump(config, f)
