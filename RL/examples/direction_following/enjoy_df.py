import sys
import os
sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))))

from sample_factory.enjoy import enjoy
from examples.direction_following.train_df import parse_df_cfg, register_df_components


def main():
    """Script entry point."""
    register_df_components()
    cfg = parse_df_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
