import sys
import os
sys.path.insert(0,'{}'.format(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))))
from sample_factory.enjoy import enjoy
from examples.watermaze2d.train_watermaze2d import parse_watermaze2d_cfg, register_watermaze2d_components


def main():
    """Script entry point."""
    register_watermaze2d_components()
    cfg = parse_watermaze2d_cfg(evaluation=True)
    status = enjoy(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())
