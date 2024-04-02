from typing import Optional

import gymnasium as gym

from sample_factory.utils.utils import is_module_available
from examples.watermaze2d.watermaze2d_env_shaping import Watermaze2dShapingWrapper


def ratlapwater_available():
    return is_module_available("Watermaze2d")


class Watermaze2dSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


Watermaze2d_ENVS = [
    Watermaze2dSpec("Watermaze2d", "watermaze2d"),
]


def watermaze2d_env_by_name(name):
    for cfg in Watermaze2d_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Watermaze2d env")


def make_watermaze2d_env(env_name, _cfg, _env_config, render_mode: Optional[str] = None, **kwargs):
    watermaze2d_spec = watermaze2d_env_by_name(env_name)
    env = Watermaze2dShapingWrapper(watermaze2d_spec.env_id, _cfg=_cfg)
    return env
