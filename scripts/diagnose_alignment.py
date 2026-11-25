import argparse
import json
from pathlib import Path

from sdgm.preview import other_previews, whu_diagnostics


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zzvr.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    whu_diagnostics(
        records(output / "whu" / "manifest.jsonl"),
        output,
        config.get("whu_diagnostic_count", 12),
        config["seed"],
    )
    other_previews(
        records(output / "srif" / "manifest.jsonl"),
        records(output / "spacenet9" / "manifest.jsonl"),
        output,
        config["srif"],
    )
    print(f"Previews and WHU diagnostics written to {output}")


if __name__ == "__main__":
    main()
