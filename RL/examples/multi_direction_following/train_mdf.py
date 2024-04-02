import sys
import os

sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))))

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from examples.multi_direction_following.mdf_params import add_mdf_env_args, mm_override_defaults
from examples.multi_direction_following.mdf_utils import MM_ENVS, make_mdf_env
from examples.model import make_core

def register_mdf_components():
    for env in MM_ENVS:
        register_env(env.name, make_mdf_env)
    global_model_factory().register_model_core_factory(make_core)

def parse_mdf_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_mdf_env_args(partial_cfg.env, parser)
    mm_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    register_mdf_components()
    cfg = parse_mdf_cfg()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    status = run_rl(cfg)
    return status

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
