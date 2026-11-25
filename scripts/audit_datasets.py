"""Run all three audits using explicit paths in a JSON configuration."""

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from sdgm.audit import audit_spacenet, audit_srif, audit_whu
from sdgm.io import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zzvr.json")
    parser.add_argument(
        "--dataset", choices=["all", "whu", "srif", "spacenet9"], default="all"
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    packages = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "rasterio", "pyproj", "Pillow"]
    }
    save_json(
        output / "environment.json",
        dict(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            hostname=platform.node(),
            python=sys.version,
            executable=sys.executable,
            packages=packages,
            config=config,
        ),
    )
    summaries = {}
    if args.dataset in ["all", "srif"]:
        summaries["SRIF"] = audit_srif(config["srif"], output)
    if args.dataset in ["all", "spacenet9"]:
        summaries["SpaceNet9"] = audit_spacenet(config["spacenet9"], output)
    if args.dataset in ["all", "whu"]:
        summaries["WHU"] = audit_whu(config["whu"], output, config.get("workers", 4))
    save_json(output / f"audit_{args.dataset}.json", summaries)
    current_summaries = {}
    for name in ["whu", "srif", "spacenet9"]:
        summary_path = output / name / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            current_summaries[summary["dataset"]] = summary
    save_json(output / "audit_all.json", current_summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    return int(
        any(summary["issue_severity"].get("error", 0) for summary in summaries.values())
    )


if __name__ == "__main__":
    sys.exit(main())
