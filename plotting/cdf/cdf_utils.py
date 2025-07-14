"""Helper functions for loading in, and managing CDf files."""
# TODO: If this repo moves to depending on imap_processing, remove this file and move to the functions there.

from __future__ import annotations

import logging
import os
from pathlib import Path

import imap_data_access
import numpy as np
import xarray as xr
from cdflib.xarray import cdf_to_xarray
from cdflib.xarray.cdf_to_xarray import ISTP_TO_XARRAY_ATTRS
from imap_data_access import ScienceFilePath

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for time conversion
TTJ2000_EPOCH = np.datetime64("2000-01-01T11:58:55.816", "ns")


def load_cdf(
    file_path: Path | str, remove_xarray_attrs: bool = True, **kwargs: dict
) -> xr.Dataset:
    """
    Load a CDF file into an xarray.Dataset.

    This function uses the `cdf_to_xarray` utility to parse a CDF file
    and convert it into an `xarray.Dataset`. Optionally, metadata attributes
    automatically added by `cdf_to_xarray` can be removed.

    Parameters
    ----------
    file_path : Path or str
        The path to the CDF file. Accepts a `pathlib.Path` or string.
    remove_xarray_attrs : bool, default=True
        If True, remove xarray attributes injected by `cdf_to_xarray`.
    **kwargs : dict, optional
        Additional keyword arguments passed to `cdf_to_xarray`.

    Returns
    -------
    xarray.Dataset
        Parsed dataset representing the contents of the CDF file.
    """
    if isinstance(file_path, imap_data_access.ImapFilePath):
        file_path = file_path.construct_path()

    dataset = cdf_to_xarray(file_path, kwargs)

    # cdf_to_xarray converts single-value attributes to lists
    # convert these back to single values where applicable
    for attribute in dataset.attrs:
        value = dataset.attrs[attribute]
        if isinstance(value, list) and len(value) == 1:
            dataset.attrs[attribute] = value[0]

    # Remove attributes specific to xarray plotting from vars and coords
    # TODO: This can be removed if/when feature is added to cdf_to_xarray to
    #      make adding these attributes optional
    if remove_xarray_attrs:
        for key in dataset.variables.keys():
            for xarray_key in ISTP_TO_XARRAY_ATTRS.values():
                dataset[key].attrs.pop(xarray_key, None)

    return dataset


def dataset_into_xarray(file_name: str) -> xr.Dataset | None:
    """
    Use IMAP file name/directory structure to load a CDF file as a xr.Dataset.

    Parameters
    ----------
    file_name : str
        Desired file to generate quicklook from.

    Returns
    -------
    None
        This function returns nothing.

    Notes
    -----
    This function only requires the file name, and not the whole path in load_cdf.
    """
    file_name_dict = ScienceFilePath.extract_filename_components(file_name)
    # print(file_name_dict)
    year = file_name_dict["start_date"][:4]
    month = file_name_dict["start_date"][4:6]

    # Define file_path
    # TODO: FIX THIS TO MAKE DYNAMIC
    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        file_name_dict["mission"],
        file_name_dict["instrument"],
        file_name_dict["data_level"],
        year,
        month,
    )
    full_path = os.path.join(file_path, file_name)
    # print('Full Path: ' + full_path)

    # Check if file exists
    if not os.path.exists(full_path):
        logger.warning("File does not exist: %s", full_path)
        data_set_return = None
    else:
        # Create xr.Dataset for plotting purposes
        data_set = load_cdf(full_path)
        data_set_return = data_set

    return data_set_return
