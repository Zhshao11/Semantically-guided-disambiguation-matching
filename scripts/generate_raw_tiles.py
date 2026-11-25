"""Generate a small frozen set only after lineage and base audit are present."""

import argparse
import json
from pathlib import Path

from sdgm.raw_tiles import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/raw_rebuild.json")
    parser.add_argument(
        "--destination",
        default="/mnt/sda/zhushihao/data/vis_sar/WHU_rebuilt_pilot_20261125",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    audit = json.loads((output / "base_audit/admission.json").read_text())
    print(
        json.dumps(generate_dataset(config, output, args.destination, audit), indent=2)
    )


if __name__ == "__main__":
    main()
