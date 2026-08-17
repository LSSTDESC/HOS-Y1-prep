#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np

from hoscodes.map_utils import smoothing
from hoscodes.params.resolve_map_files import (
    build_output_directory,
    load_config,
    resolve_input_files,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create boundary-corrected, Gaussian-smoothed convergence maps "
            "from the inputs selected by a YAML configuration."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the convergence-map YAML configuration file.",
    )
    parser.add_argument(
        "--bcm-model",
        type=str,
        default=None,
        help="Override model.bcm_model from the YAML configuration.",
    )
    return parser.parse_args()


def get_smoothing_scales(config: dict[str, Any]) -> list[float]:
    """Read and validate the requested Gaussian FWHM values."""
    postprocessing = config.get("postprocessing", {})
    smoothing_config = postprocessing.get("smoothing", {})

    if not postprocessing.get("enabled", False):
        raise ValueError("postprocessing.enabled must be true for smoothing.")
    if not smoothing_config.get("enabled", False):
        raise ValueError(
            "postprocessing.smoothing.enabled must be true for smoothing."
        )

    scales = smoothing_config.get("scales_arcmin")
    if scales is None:
        # Retain compatibility with configurations containing one scale.
        scales = [smoothing_config.get("scale_arcmin")]

    if not isinstance(scales, list) or not scales or scales == [None]:
        raise ValueError(
            "postprocessing.smoothing.scales_arcmin must be a non-empty list."
        )

    parsed_scales = [float(scale) for scale in scales]
    if not all(np.isfinite(scale) and scale > 0 for scale in parsed_scales):
        raise ValueError("Every smoothing scale must be positive and finite.")

    return parsed_scales


def load_footprint(mask_path: str | Path) -> np.ndarray:
    """Load a full-sky binary HEALPix footprint."""
    footprint = np.asarray(np.load(Path(mask_path).expanduser())).squeeze()

    if footprint.ndim != 1:
        raise ValueError("The configured mask must be a one-dimensional map.")

    # This also rejects a list of selected pixel indices: smoothing needs a
    # full-sky footprint with exactly 12 * NSIDE**2 elements.
    hp.get_nside(footprint)
    return footprint > 0


def regrid_map_and_footprint(
    raw_map: np.ndarray,
    footprint: np.ndarray,
    nside_out: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Put a convergence map and its footprint on the configured NSIDE."""
    raw_map = np.asarray(raw_map, dtype=np.float64).squeeze()
    if raw_map.ndim != 1:
        raise ValueError("Each convergence input must be a 1D HEALPix map.")

    input_nside = hp.get_nside(raw_map)
    mask_nside = hp.get_nside(footprint)

    if mask_nside != input_nside:
        footprint = hp.ud_grade(
            footprint.astype(np.float64),
            nside_out=input_nside,
            order_in="RING",
            order_out="RING",
        ) > 0.0

    valid = footprint & np.isfinite(raw_map) & (raw_map > -1.0e29)

    if input_nside == nside_out:
        prepared_map = np.zeros_like(raw_map, dtype=np.float64)
        prepared_map[valid] = raw_map[valid]
        return prepared_map, valid

    # Regrid signal and coverage separately so sentinel values outside the
    # footprint cannot leak into the lower-resolution convergence map.
    weights = valid.astype(np.float64)
    weighted_map = np.zeros_like(raw_map, dtype=np.float64)
    weighted_map[valid] = raw_map[valid]

    regridded_signal = hp.ud_grade(
        weighted_map,
        nside_out=nside_out,
        order_in="RING",
        order_out="RING",
    )
    regridded_weights = hp.ud_grade(
        weights,
        nside_out=nside_out,
        order_in="RING",
        order_out="RING",
    )

    regridded_map = np.zeros_like(regridded_signal)
    regridded_footprint = regridded_weights > 0.0
    np.divide(
        regridded_signal,
        regridded_weights,
        out=regridded_map,
        where=regridded_footprint,
    )
    return regridded_map, regridded_footprint


def format_scale(scale_arcmin: float) -> str:
    """Return a filename-safe representation of an angular scale."""
    return f"{scale_arcmin:g}".replace(".", "p")


def main() -> None:
    args = parse_arguments()
    config_path = args.config.resolve()
    config = load_config(config_path)

    if args.bcm_model is not None:
        config["model"]["bcm_model"] = args.bcm_model

    input_files = resolve_input_files(config)
    tomographic_bins = config["dataset"]["tomographic_bins"]
    nside_out = int(config["map_generation"]["nside_out"])
    scales_arcmin = get_smoothing_scales(config)

    mask_config = config.get("postprocessing", {}).get("mask", {})
    if not mask_config.get("enabled", False) or not mask_config.get("path"):
        raise ValueError(
            "An enabled postprocessing.mask.path is required for "
            "boundary-corrected smoothing."
        )

    footprint = load_footprint(mask_config["path"])

    smoothing_config = config["postprocessing"]["smoothing"]
    output_subdir = smoothing_config.get("output_subdir", "smoothed_maps")
    output_directory = build_output_directory(config) / output_subdir
    output_directory.mkdir(parents=True, exist_ok=True)

    convergence_multiplier = float(
        config.get("map_generation", {}).get("convergence_multiplier", 1.0)
    )

    print(f"Output directory: {output_directory}")

    for tomo, input_file in zip(
        tomographic_bins,
        input_files,
        strict=True,
    ):
        print(f"\nProcessing tomographic bin {tomo}: {input_file}")
        raw_map = np.load(input_file)
        prepared_map, prepared_footprint = regrid_map_and_footprint(
            raw_map=raw_map,
            footprint=footprint,
            nside_out=nside_out,
        )
        # Apply the sign convention only after invalid sentinel pixels have
        # been removed; otherwise a large negative sentinel could become a
        # large positive value and be mistaken for valid convergence.
        prepared_map *= convergence_multiplier

        for scale_arcmin in scales_arcmin:
            smoothed_map = smoothing(
                raw_map=prepared_map,
                footprint_mask=prepared_footprint,
                scale_length=scale_arcmin,
            )

            output_name = (
                f"{input_file.stem}_smoothed_"
                f"{format_scale(scale_arcmin)}arcmin.npy"
            )
            output_path = output_directory / output_name
            np.save(
                output_path,
                np.asarray(smoothed_map.filled(hp.UNSEEN), dtype=np.float32),
            )
            print(f"  {scale_arcmin:g} arcmin -> {output_path}")

        del raw_map
        del prepared_map
        del prepared_footprint


if __name__ == "__main__":
    main()
