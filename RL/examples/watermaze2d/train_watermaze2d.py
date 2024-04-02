import sys
import os

sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))))

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from examples.watermaze2d.watermaze2d_params import add_watermaze2d_env_args, watermaze2d_override_defaults
from examples.watermaze2d.watermaze2d_utils import Watermaze2d_ENVS, make_watermaze2d_env
from examples.model import make_core

def register_watermaze2d_components():
    for env in Watermaze2d_ENVS:
        register_env(env.name, make_watermaze2d_env)
    global_model_factory().register_model_core_factory(make_core)

def parse_watermaze2d_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_watermaze2d_env_args(partial_cfg.env, parser)
    watermaze2d_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_watermaze2d_components()
    cfg = parse_watermaze2d_cfg()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
