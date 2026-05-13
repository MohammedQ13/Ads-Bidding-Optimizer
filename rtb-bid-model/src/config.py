import yaml
import os


def load_config(path=None):
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, 'config.yaml')
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
