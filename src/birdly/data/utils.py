import yaml


def load_config():
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(".birdly/local.yaml", "r") as f:
        local_config = yaml.safe_load(f)
    
    with open('params.yaml', "r") as file:
        params = yaml.safe_load(file)


    return config, local_config, params
