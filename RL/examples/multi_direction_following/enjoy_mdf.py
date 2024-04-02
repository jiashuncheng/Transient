import sys
import os
sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))))

from sample_factory.enjoy import enjoy
from examples.multi_direction_following.train_mdf import parse_mdf_cfg, register_mdf_components


def main():
    """Script entry point."""
    register_mdf_components()
    cfg = parse_mdf_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
