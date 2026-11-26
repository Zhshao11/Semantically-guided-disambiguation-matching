import argparse
import json
from pathlib import Path
from sdgm.os_family import freeze_families


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_family_baseline.json")
    args = parser.parse_args()
    print(
        json.dumps(freeze_families(json.loads(Path(args.config).read_text())), indent=2)
    )


if __name__ == "__main__":
    main()
