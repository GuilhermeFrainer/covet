# -*- coding: utf-8 -*-
"""Script to generate stemmed standard experiment YAML configurations."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

DATASETS = ["anes", "fed", "gadarian", "trump", "trump_s25000", "yelp"]


def main():
    """Generates stemmed experiment config files for all active datasets."""
    count = 0
    for dataset in DATASETS:
        src_dir = EXPERIMENTS_DIR / dataset
        dest_dir = EXPERIMENTS_DIR / f"{dataset}_stemmed"
        dest_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"Warning: Source directory {src_dir} does not exist.")
            continue

        for yaml_file in src_dir.glob(f"{dataset}_standard_*.yaml"):
            with open(yaml_file, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)

            # Update extends field
            content["extends"] = f"datasets/{dataset}_stemmed.yaml"

            # Update experiment metadata
            if "experiment" in content:
                old_name = content["experiment"].get("name", "")
                if old_name:
                    content["experiment"]["name"] = old_name.replace(
                        f"{dataset}_standard_", f"{dataset}_stemmed_standard_"
                    )
                old_desc = content["experiment"].get("description", "")
                if old_desc:
                    content["experiment"]["description"] = old_desc.replace(
                        f"on {dataset}", f"on {dataset}_stemmed"
                    )

            dest_file = dest_dir / yaml_file.name
            with open(dest_file, "w", encoding="utf-8") as f:
                yaml.dump(content, f, sort_keys=False)
            count += 1

    print(f"Successfully generated {count} stemmed experiment configuration files.")


if __name__ == "__main__":
    main()
