#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import healpy as hp
import numpy as np

from hoscodes.params.resolve_map_files import (
    build_output_directory,
    get_postprocessing_metadata,
    load_config,
    resolve_input_files,
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert tomographic galaxy catalogues into shear maps "
            "stored in one HDF5 file."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--bcm-model",
        type=str,
        default=None,
        help=(
            "Override model.bcm_model from the YAML configuration, "
            "for example: dmo or dmb_Mc2e14."
        ),
    )

    return parser.parse_args()


def radec_to_pixel(
    ra: np.ndarray,
    dec: np.ndarray,
    nside: int,
) -> np.ndarray:
    """
    Convert right ascension and declination to HEALPix pixel indices.

    Parameters
    ----------
    ra
        Right ascension in degrees.
    dec
        Declination in degrees.
    nside
        HEALPix NSIDE.

    Returns
    -------
    numpy.ndarray
        HEALPix pixel indices in RING ordering.
    """

    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)

    return hp.ang2pix(
        nside,
        theta,
        phi,
        nest=False,
    )


def create_healpy_map(
    ra: np.ndarray,
    dec: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    weights: np.ndarray,
    nside: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create weighted shear maps for one tomographic bin.

    The returned footprint is a bin-dependent occupancy map. It is one
    wherever the accumulated catalogue weight is nonzero and zero elsewhere.
    It is not the external survey mask from the configuration file.
    """

    npix = hp.nside2npix(nside)

    # Accumulate in float64 for numerical stability.
    g1_map = np.zeros(npix, dtype=np.float64)
    g2_map = np.zeros(npix, dtype=np.float64)
    weight_map = np.zeros(npix, dtype=np.float64)

    pixels = radec_to_pixel(
        ra=ra,
        dec=dec,
        nside=nside,
    )

    unique_pixels, inverse_indices = np.unique(
        pixels,
        return_inverse=True,
    )

    weight_map[unique_pixels] = np.bincount(
        inverse_indices,
        weights=weights,
        minlength=len(unique_pixels),
    )

    g1_map[unique_pixels] = np.bincount(
        inverse_indices,
        weights=g1 * weights,
        minlength=len(unique_pixels),
    )

    g2_map[unique_pixels] = np.bincount(
        inverse_indices,
        weights=g2 * weights,
        minlength=len(unique_pixels),
    )

    occupied = weight_map != 0.0

    g1_map[occupied] /= weight_map[occupied]
    g2_map[occupied] /= weight_map[occupied]

    footprint_map = occupied.astype(np.uint8)

    return g1_map, g2_map, weight_map, footprint_map


def read_catalogue(
    catalogue_file: str | Path,
    ia_enabled: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Read one catalogue and return coordinates, effective shear, and weights.

    Expected columns when IA is enabled
    -----------------------------------
    ra, dec, z, g1, g2, w, e1, e2, e1_noise, e2_noise, e1_IA, e2_IA

    Expected columns when IA is disabled
    ------------------------------------
    ra, dec, z, g1, g2, w
    """

    catalogue_file = Path(catalogue_file)

    catalogue = np.loadtxt(
        catalogue_file,
        skiprows=1,
    )

    if catalogue.ndim != 2:
        raise ValueError(
            f"Catalogue must contain a two-dimensional table: "
            f"{catalogue_file}"
        )

    if ia_enabled:
        if catalogue.shape[1] < 12:
            raise ValueError(
                "IA-enabled catalogue must contain at least 12 columns, "
                f"but {catalogue_file} contains {catalogue.shape[1]}."
            )

        (
            ra,
            dec,
            _redshift,
            g1,
            g2,
            weights,
            _e1,
            _e2,
            _e1_noise,
            _e2_noise,
            e1_ia,
            e2_ia,
        ) = catalogue[:, :12].T

        g1_effective = g1 + e1_ia
        g2_effective = g2 + e2_ia

    else:
        if catalogue.shape[1] < 6:
            raise ValueError(
                "Catalogue without IA must contain at least 6 columns, "
                f"but {catalogue_file} contains {catalogue.shape[1]}."
            )

        (
            ra,
            dec,
            _redshift,
            g1_effective,
            g2_effective,
            weights,
        ) = catalogue[:, :6].T

    return (
        ra,
        dec,
        g1_effective,
        g2_effective,
        weights,
    )


def build_hdf5_filename(config: dict[str, Any]) -> str:
    """
    Build the HDF5 filename for this catalogue-to-shear-map product.

    Output naming is intentionally handled by this postprocessing code,
    rather than by resolve_map_files.py.
    """

    model = config["model"]

    filename_parts = [
        "shear_maps",
        str(model["bcm_model"]),
        str(model["ia_model"]),
        str(model.get("ia_amplitude") or "no-amplitude"),
        str(model["noise"]),
        f"ng{model['number_density']}",
        f"nside{config['map_generation']['nside_out']}",
    ]

    return "_".join(filename_parts) + ".hdf5"


def write_string_attribute(
    hdf5_object: h5py.Group | h5py.File,
    name: str,
    value: Any,
) -> None:
    """Write a value safely as an HDF5 attribute."""

    if value is None:
        hdf5_object.attrs[name] = ""
    elif isinstance(value, (dict, list, tuple)):
        hdf5_object.attrs[name] = json.dumps(value)
    else:
        hdf5_object.attrs[name] = value


def write_root_metadata(
    hdf5_file: h5py.File,
    config: dict[str, Any],
    config_path: Path,
    postprocessing: dict[str, Any],
) -> None:
    """Write dataset, model, HEALPix, and mask metadata."""

    dataset_config = config["dataset"]
    model_config = config["model"]
    input_config = config["input"]
    map_config = config["map_generation"]

    hdf5_file.attrs["format_version"] = "1.0"
    hdf5_file.attrs["product_type"] = "tomographic_shear_maps"
    hdf5_file.attrs["creation_time_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    hdf5_file.attrs["config_file"] = str(config_path.resolve())

    write_string_attribute(
        hdf5_file,
        "dataset_name",
        dataset_config.get("name"),
    )
    write_string_attribute(
        hdf5_file,
        "dataset_subdir",
        dataset_config.get("subdir"),
    )
    write_string_attribute(
        hdf5_file,
        "tomographic_bins",
        dataset_config["tomographic_bins"],
    )

    write_string_attribute(
        hdf5_file,
        "input_map_type",
        input_config["map_type"],
    )
    write_string_attribute(
        hdf5_file,
        "input_ia_filename_tag",
        input_config["ia_filename_tag"],
    )

    write_string_attribute(
        hdf5_file,
        "bcm_model",
        model_config["bcm_model"],
    )
    write_string_attribute(
        hdf5_file,
        "ia_enabled",
        bool(model_config.get("ia_enabled", False)),
    )
    write_string_attribute(
        hdf5_file,
        "ia_model",
        model_config["ia_model"],
    )
    write_string_attribute(
        hdf5_file,
        "ia_amplitude",
        model_config.get("ia_amplitude"),
    )
    write_string_attribute(
        hdf5_file,
        "number_density",
        str(model_config["number_density"]),
    )
    write_string_attribute(
        hdf5_file,
        "noise",
        model_config["noise"],
    )

    hdf5_file.attrs["nside"] = int(map_config["nside_out"])
    hdf5_file.attrs["npix"] = hp.nside2npix(
        int(map_config["nside_out"])
    )
    hdf5_file.attrs["healpix_ordering"] = "RING"
    hdf5_file.attrs["coordinate_system"] = "equatorial"
    hdf5_file.attrs["ra_unit"] = "degree"
    hdf5_file.attrs["dec_unit"] = "degree"

    mask_config = postprocessing.get("mask", {})

    hdf5_file.attrs["external_mask_enabled"] = bool(
        mask_config.get("enabled", False)
    )
    write_string_attribute(
        hdf5_file,
        "external_mask_name",
        mask_config.get("name"),
    )
    write_string_attribute(
        hdf5_file,
        "external_mask_path",
        mask_config.get("path"),
    )

    smoothing_config = postprocessing.get("smoothing", {})

    hdf5_file.attrs["smoothing_enabled"] = bool(
        smoothing_config.get("enabled", False)
    )
    write_string_attribute(
        hdf5_file,
        "smoothing_name",
        smoothing_config.get("name"),
    )

    hdf5_file.attrs["smoothing_scale_arcmin"] = float(
        smoothing_config.get("scale_arcmin", 0.0)
    )

    hdf5_file.attrs["footprint_definition"] = (
        "Per-bin occupancy map: 1 where accumulated catalogue weight "
        "is nonzero, otherwise 0. This is distinct from the external mask."
    )


def write_tomographic_bin(
    hdf5_file: h5py.File,
    tomo: int,
    catalogue_file: Path,
    g1_map: np.ndarray,
    g2_map: np.ndarray,
    weight_map: np.ndarray,
    footprint_map: np.ndarray,
) -> None:
    """Write all products from one tomographic bin into an HDF5 group."""

    group_name = f"tomo_{tomo}"

    if group_name in hdf5_file:
        del hdf5_file[group_name]

    group = hdf5_file.create_group(group_name)

    group.attrs["tomographic_bin"] = int(tomo)
    group.attrs["input_catalogue"] = str(catalogue_file.resolve())
    group.attrs["number_of_occupied_pixels"] = int(
        np.count_nonzero(footprint_map)
    )
    group.attrs["sum_of_weights"] = float(np.sum(weight_map))

    dataset_options = {
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
        "chunks": True,
    }

    g1_dataset = group.create_dataset(
        "g1",
        data=g1_map.astype(np.float32, copy=False),
        **dataset_options,
    )
    g1_dataset.attrs["description"] = (
        "Weighted mean first shear component per HEALPix pixel."
    )
    g1_dataset.attrs["dtype_on_disk"] = "float32"

    g2_dataset = group.create_dataset(
        "g2",
        data=g2_map.astype(np.float32, copy=False),
        **dataset_options,
    )
    g2_dataset.attrs["description"] = (
        "Weighted mean second shear component per HEALPix pixel."
    )
    g2_dataset.attrs["dtype_on_disk"] = "float32"

    weight_dataset = group.create_dataset(
        "weight",
        data=weight_map.astype(np.float32, copy=False),
        **dataset_options,
    )
    weight_dataset.attrs["description"] = (
        "Sum of catalogue weights per HEALPix pixel."
    )
    weight_dataset.attrs["dtype_on_disk"] = "float32"

    footprint_dataset = group.create_dataset(
        "footprint",
        data=footprint_map,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=True,
    )
    footprint_dataset.attrs["description"] = (
        "Per-bin binary occupancy map; 1 where weight is nonzero."
    )
    footprint_dataset.attrs["dtype_on_disk"] = "uint8"


def main() -> None:
    """Convert all configured tomographic catalogues into one HDF5 file."""

    args = parse_arguments()
    
    config_path = Path(
        "/global/homes/j/jatorres/HOS-Y1-prep/"
        "hoscodes/params/shear_catalogue.yaml"
    )

    config_path = args.config.resolve()
    config = load_config(config_path)

    if args.bcm_model is not None:
        config["model"]["bcm_model"] = args.bcm_model

    input_files = resolve_input_files(config)
    output_directory = build_output_directory(config)
    postprocessing = get_postprocessing_metadata(config)

    nside_out = int(config["map_generation"]["nside_out"])
    tomographic_bins = config["dataset"]["tomographic_bins"]
    ia_enabled = bool(config["model"].get("ia_enabled", False))

    hdf5_filename = build_hdf5_filename(config)
    hdf5_path = output_directory / hdf5_filename

    print(f"Output HDF5 file: {hdf5_path}")

    mask_config = postprocessing["mask"]

    if mask_config["enabled"]:
        print(
            "Configured external mask stored as metadata: "
            f"{mask_config['path']}"
        )

    # Opening with mode="w" replaces an existing file with the same name.
    # Use mode="x" instead if accidental overwrite must raise an error.
    with h5py.File(hdf5_path, mode="w") as hdf5_file:
        write_root_metadata(
            hdf5_file=hdf5_file,
            config=config,
            config_path=config_path,
            postprocessing=postprocessing,
        )

        for tomo, catalogue_file in zip(
            tomographic_bins,
            input_files,
            strict=True,
        ):
            catalogue_file = Path(catalogue_file)

            print(f"\nProcessing tomographic bin {tomo}")
            print(f"Input catalogue: {catalogue_file}")

            (
                ra,
                dec,
                g1_effective,
                g2_effective,
                weights,
            ) = read_catalogue(
                catalogue_file=catalogue_file,
                ia_enabled=ia_enabled,
            )

            (
                g1_map,
                g2_map,
                weight_map,
                footprint_map,
            ) = create_healpy_map(
                ra=ra,
                dec=dec,
                g1=g1_effective,
                g2=g2_effective,
                weights=weights,
                nside=nside_out,
            )

            write_tomographic_bin(
                hdf5_file=hdf5_file,
                tomo=tomo,
                catalogue_file=catalogue_file,
                g1_map=g1_map,
                g2_map=g2_map,
                weight_map=weight_map,
                footprint_map=footprint_map,
            )

            # Release the large dense maps before processing the next bin.
            del g1_map
            del g2_map
            del weight_map
            del footprint_map

            hdf5_file.flush()

    print(f"\nFinished writing:\n{hdf5_path}")


if __name__ == "__main__":
    main()