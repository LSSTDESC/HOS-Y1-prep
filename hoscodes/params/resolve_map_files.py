#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


VALID_BCM_MODELS = {
    "dmb",
    "dmb_Mc2e14",
    "dmb_Mc5e13",
    "dmb_theta_ej_3",
    "dmb_theta_ej_6",
    "dmb_Mc2.5e13",
    "dmb_Mc4e14",
    "dmb_theta_ej_2",
    "dmb_theta_ej_5",
    "dmo",
}

VALID_IA_MODELS = {
    "noIA",
    "noAI",
    "NLA",
    "TT",
    "TATT",
    "deltaNLA",
    "deltaTT",
}

VALID_IA_AMPLITUDES = {
    "AIAp1",
    "AIA0",
    "AIAm1",
    "noAI_bta1",
    "AIAp1_bta1",
    "AIAp1_bta2",
    "AIAp1_C2p1_bta1",
    "C2p1",
    "C2m1",
}

VALID_MAP_TYPES = {
    "GalCat",
    "shear",
    "kappa",
    "kappa-KS",
}

VALID_NOISE_OPTIONS = {
    "noisefree",
    "noisy",
}

VALID_NUMBER_DENSITIES = {
    "0.5",
    "0.6",
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration must contain a top-level dictionary."
        )

    return config


def require_key(
    mapping: dict[str, Any],
    key: str,
    section: str,
) -> Any:
    """Return a required configuration value."""

    if key not in mapping:
        raise KeyError(
            f"Required key '{key}' is missing from section '{section}'."
        )

    return mapping[key]


def validate_config(config: dict[str, Any]) -> None:
    """Validate the main model, input, and output-directory selections."""

    dataset = require_key(config, "dataset", "root")
    model = require_key(config, "model", "root")
    input_config = require_key(config, "input", "root")
    map_generation = require_key(config, "map_generation", "root")

    require_key(dataset, "parent_dir", "dataset")
    require_key(dataset, "subdir", "dataset")

    tomographic_bins = require_key(
        dataset,
        "tomographic_bins",
        "dataset",
    )

    if not isinstance(tomographic_bins, list) or not tomographic_bins:
        raise ValueError(
            "'dataset.tomographic_bins' must be a non-empty list."
        )

    if not all(isinstance(tomo, int) for tomo in tomographic_bins):
        raise TypeError(
            "All values in 'dataset.tomographic_bins' must be integers."
        )

    if len(set(tomographic_bins)) != len(tomographic_bins):
        raise ValueError(
            "'dataset.tomographic_bins' contains duplicate values."
        )

    bcm_model = require_key(model, "bcm_model", "model")
    ia_model = require_key(model, "ia_model", "model")
    number_density = str(
        require_key(model, "number_density", "model")
    )
    noise = require_key(model, "noise", "model")

    map_type = require_key(input_config, "map_type", "input")
    ia_filename_tag = require_key(
        input_config,
        "ia_filename_tag",
        "input",
    )

    require_key(map_generation, "nside_out", "map_generation")
    require_key(map_generation, "output_root", "map_generation")

    if bcm_model not in VALID_BCM_MODELS:
        raise ValueError(
            f"Unknown BCM model '{bcm_model}'. "
            f"Valid values are: {sorted(VALID_BCM_MODELS)}"
        )

    if ia_model not in VALID_IA_MODELS:
        raise ValueError(
            f"Unknown IA model '{ia_model}'. "
            f"Valid values are: {sorted(VALID_IA_MODELS)}"
        )

    if number_density not in VALID_NUMBER_DENSITIES:
        raise ValueError(
            f"Unknown number density '{number_density}'. "
            f"Valid values are: {sorted(VALID_NUMBER_DENSITIES)}"
        )

    if noise not in VALID_NOISE_OPTIONS:
        raise ValueError(
            f"Unknown noise option '{noise}'. "
            f"Valid values are: {sorted(VALID_NOISE_OPTIONS)}"
        )

    if map_type not in VALID_MAP_TYPES:
        raise ValueError(
            f"Unknown map type '{map_type}'. "
            f"Valid values are: {sorted(VALID_MAP_TYPES)}"
        )

    if not isinstance(ia_filename_tag, str) or not ia_filename_tag:
        raise ValueError(
            "'input.ia_filename_tag' must be a non-empty string."
        )

    patterns = require_key(input_config, "patterns", "input")

    if not isinstance(patterns, dict):
        raise TypeError("'input.patterns' must be a dictionary.")

    if map_type not in patterns:
        raise KeyError(
            f"No search patterns are defined for map type '{map_type}'."
        )

    if not isinstance(patterns[map_type], list) or not patterns[map_type]:
        raise ValueError(
            f"The search-pattern list for '{map_type}' is empty."
        )

    ia_amplitude = model.get("ia_amplitude")

    if (
        ia_amplitude is not None
        and ia_amplitude not in VALID_IA_AMPLITUDES
    ):
        raise ValueError(
            f"Unknown IA amplitude '{ia_amplitude}'. "
            f"Valid values are: {sorted(VALID_IA_AMPLITUDES)}"
        )


def get_input_base_directory(config: dict[str, Any]) -> Path:
    """
    Construct the input directory for the selected BCM model.

    The resulting path follows:

        parent_dir / subdir / bcm_model
    """

    dataset = config["dataset"]
    bcm_model = config["model"]["bcm_model"]

    return (
        Path(dataset["parent_dir"]).expanduser()
        / dataset["subdir"]
        / bcm_model
    )


def build_format_values(
    config: dict[str, Any],
    tomo: int,
) -> dict[str, Any]:
    """Build values available to the input filename glob patterns."""

    model = config["model"]
    input_config = config["input"]
    map_generation = config["map_generation"]

    return {
        "tomo": tomo,
        "bcm_model": model["bcm_model"],
        "ia_enabled": model.get("ia_enabled", True),
        "ia_model": model["ia_model"],
        "ia_amplitude": model.get("ia_amplitude") or "none",
        "ia_filename_tag": input_config["ia_filename_tag"],
        "number_density": str(model["number_density"]),
        "noise": model["noise"],
        "map_type": input_config["map_type"],
        "nside": map_generation["nside_out"],
    }


def find_file_for_tomographic_bin(
    base_directory: Path,
    patterns: list[str],
    format_values: dict[str, Any],
) -> Path:
    """
    Find exactly one input file for a tomographic bin.

    Search patterns are tried in order. The first pattern producing exactly
    one match is used. An error is raised if no file or multiple files match.
    """

    searched_patterns: list[str] = []

    for pattern_template in patterns:
        pattern = pattern_template.format(**format_values)
        searched_patterns.append(str(base_directory / pattern))

        matches = sorted(
            path.resolve()
            for path in base_directory.glob(pattern)
            if path.is_file()
        )

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            formatted_matches = "\n".join(
                f"  - {path}" for path in matches
            )

            raise RuntimeError(
                "Multiple files matched the same tomographic bin.\n"
                f"Tomographic bin: {format_values['tomo']}\n"
                f"Pattern: {base_directory / pattern}\n"
                f"Matching files:\n{formatted_matches}\n\n"
                "Make the corresponding input glob pattern more specific."
            )

    formatted_searches = "\n".join(
        f"  - {pattern}" for pattern in searched_patterns
    )

    raise FileNotFoundError(
        "No input file was found.\n"
        f"Tomographic bin: {format_values['tomo']}\n"
        f"Patterns searched:\n{formatted_searches}"
    )


def resolve_input_files(config: dict[str, Any]) -> list[Path]:
    """Resolve one input file for each configured tomographic bin."""

    validate_config(config)

    base_directory = get_input_base_directory(config)
    map_type = config["input"]["map_type"]
    patterns = config["input"]["patterns"][map_type]
    tomographic_bins = config["dataset"]["tomographic_bins"]

    if not base_directory.is_dir():
        raise NotADirectoryError(
            f"Input base directory does not exist: {base_directory}"
        )

    input_files: list[Path] = []

    for tomo in tomographic_bins:
        format_values = build_format_values(config, tomo)

        input_file = find_file_for_tomographic_bin(
            base_directory=base_directory,
            patterns=patterns,
            format_values=format_values,
        )

        input_files.append(input_file)

    return input_files


def build_output_directory(config: dict[str, Any]) -> Path:
    """
    Construct the directory where processing codes may save their products.

    The output hierarchy follows:

        output_root / dataset.subdir / bcm_model

    This resolver does not determine output filenames or output formats.
    Individual processing codes are responsible for deciding which files
    they create inside this directory.
    """

    dataset = config["dataset"]
    model = config["model"]
    map_generation = config["map_generation"]

    output_directory = (
        Path(map_generation["output_root"]).expanduser()
        / dataset["subdir"]
        / model["bcm_model"]
    )

    if map_generation.get("create_output_directory", True):
        output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory.resolve()


def get_postprocessing_metadata(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return normalized postprocessing metadata."""

    postprocessing = config.get("postprocessing", {})
    mask = postprocessing.get("mask", {})
    smoothing = postprocessing.get("smoothing", {})

    mask_path = mask.get("path")

    if mask_path is not None:
        mask_path = str(Path(mask_path).expanduser())

    return {
        "enabled": postprocessing.get("enabled", False),
        "mask": {
            "enabled": mask.get("enabled", False),
            "path": mask_path,
            "name": mask.get("name"),
        },
        "smoothing": {
            "enabled": smoothing.get("enabled", False),
            "name": smoothing.get("name", "no-smoothing"),
            "scale_arcmin": smoothing.get("scale_arcmin", 0.0),
        },
    }


def build_manifest(
    config: dict[str, Any],
    input_files: list[Path],
    output_directory: Path,
) -> dict[str, Any]:
    """
    Build a JSON-serializable pipeline manifest.

    The manifest records the resolved input files and the directory available
    to downstream processing codes. It does not prescribe output filenames.
    """

    tomographic_bins = config["dataset"]["tomographic_bins"]

    resolved_inputs = [
        {
            "tomographic_bin": tomo,
            "input_file": str(input_file),
        }
        for tomo, input_file in zip(
            tomographic_bins,
            input_files,
            strict=True,
        )
    ]

    return {
        "dataset": {
            "name": config["dataset"].get("name"),
            "parent_dir": config["dataset"]["parent_dir"],
            "subdir": config["dataset"]["subdir"],
            "tomographic_bins": tomographic_bins,
        },
        "model": config["model"],
        "input": {
            "map_type": config["input"]["map_type"],
            "ia_filename_tag": config["input"]["ia_filename_tag"],
            "base_directory": str(get_input_base_directory(config)),
            "files": [str(path) for path in input_files],
            "resolved_tomographic_files": resolved_inputs,
        },
        "map_generation": {
            "nside_out": config["map_generation"]["nside_out"],
            "output_root": config["map_generation"]["output_root"],
            "output_directory": str(output_directory),
        },
        "postprocessing": get_postprocessing_metadata(config),
    }


def write_manifest(
    manifest: dict[str, Any],
    manifest_path: str | Path,
) -> Path:
    """Write the resolved pipeline manifest to JSON."""

    output_path = Path(manifest_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    return output_path.resolve()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Resolve weak-lensing catalogue or map files from a YAML "
            "configuration."
        )
    )

    parser.add_argument(
        "config",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Optional path where the resolved pipeline manifest will be "
            "written as JSON."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Resolve input files and the downstream output directory."""

    args = parse_arguments()

    config = load_config(args.config)
    input_files = resolve_input_files(config)
    output_directory = build_output_directory(config)

    manifest = build_manifest(
        config=config,
        input_files=input_files,
        output_directory=output_directory,
    )

    print("Resolved input files:")

    for entry in manifest["input"]["resolved_tomographic_files"]:
        print(
            f"  Tomographic bin {entry['tomographic_bin']}: "
            f"{entry['input_file']}"
        )

    print(f"\nOutput directory:\n  {output_directory}")

    print("\nPython input-file list:")
    print([str(path) for path in input_files])

    if args.manifest is not None:
        manifest_path = write_manifest(
            manifest=manifest,
            manifest_path=args.manifest,
        )

        print(f"\nManifest written to:\n  {manifest_path}")


if __name__ == "__main__":
    main()