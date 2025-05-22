"""Class for abstracting and organizing quicklook plots."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from cdflib.xarray import cdf_to_xarray, xarray_to_cdf

import imap_data_access

from pathlib import Path

import xarray as xr

from cdflib.xarray.cdf_to_xarray import ISTP_TO_XARRAY_ATTRS

from typing import Union

import numpy as np

from datetime import datetime, timedelta

from dataclasses import dataclass, field

from abc import ABC, abstractmethod

TTJ2000_EPOCH = np.datetime64("2000-01-01T11:58:55.816", "ns")


def load_cdf(
    file_path: Union[Path, str], remove_xarray_attrs: bool = True, **kwargs: dict
) -> xr.Dataset:
    """
    Load the contents of a CDF file into an ``xarray`` dataset.

    Parameters
    ----------
    file_path : Path or ImapFilePath or str
        The path to the CDF file or ImapFilePath object.
    remove_xarray_attrs : bool
        Whether to remove the xarray attributes that get injected by the
        cdf_to_xarray function from the output xarray.Dataset. Default is True.
    **kwargs : dict, optional
        Keyword arguments for ``cdf_to_xarray``.

    Returns
    -------
    dataset : xarray.Dataset
        The ``xarray`` dataset for the CDF file.
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


def dataset_into_xarray(file_name: str) -> Union[xr.Dataset, bool]:
    """
    Uses IMAP file name/directory structure to load in CDF file as an xr.Dataset.

    Parameters
    ----------
    file_name : str
        Desired file to generate quicklook from.

    Returns
    -------
    None
        This function returns nothing.
    """

    # Pulls needed info from IMAP file name structure for file_path
    mission, instrument, level, descriptor, year_month, version_no = file_name.split("_")
    year = year_month[:4]
    month = year_month[4:6]

    # Define file_path
    file_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "imap",
        instrument,
        level,
        year,
        month,
    )
    full_path = os.path.join(file_path, file_name)
    # print('Full Path: ' + full_path)

    # Check if file exists
    if not os.path.exists(full_path):
        print("File does not exist")
        return False
    else:
        # Create xr.Dataset for plotting purposes
        data_set = load_cdf(full_path)
        return data_set


def convert_j200_to_utc(time_array: np.array) -> np.array:
    times = TTJ2000_EPOCH + time_array.astype("timedelta64[ns]")
    return times


# TODO: Check the way this is handled.
def generate_instrument_quicklook(filename: str):
    mission, instrument, level, descriptor, year_month, version_no = filename.split("_")
    for cls in (MagQuicklookGenerator, IdexQuicklookGenerator):
        try:
            return cls(filename)
        except(
            QuicklookGenerator.QuicklookGeneratorError,
        ):
            continue
        raise ValueError(
            f"Invalid input for {filename}. It does not match any file formats."
        )


@dataclass
class QuicklookGenerator(ABC):
    """General Quicklooks class.

    Attributes
    ----------
    TODO: Add Info
    """

    # Plot Info
    data_set: xr.Dataset = None
    x_variable: list[str] = None
    x_data: list[np.array] = None
    y_variable: list[str] = None
    y_data: list[np.array] = None

    # Making plots look pretty
    title: str = None
    x_axis_label: str = None
    y_axis_label: str = None
    same_axes: bool = None

    def __init__(self, file_name):
        """Initialize using a xr.Dataset.

        Parameters
        ----------
        file_name: str
            Xarray dataset holding CDF file info.
        """
        # Plot Info
        data_set: xr.Dataset = None
        self.data_set = dataset_into_xarray(file_name)

        class QuicklookGeneratorError(Exception):
            """Indicate that the QuicklookInput is invalid."""
            pass

    @abstractmethod
    def two_dimensional_plot(self):
        raise NotImplementedError


class MagQuicklookGenerator(QuicklookGenerator):
    """Mag subclass for mag quicklook plots."""

    def two_dimensional_plot(self, variable: str):
        if variable == "mag sensor co-ord":
            self.vector_comp_plot()
        elif variable == "rtn":
            self.rtn_comp_plot()
        elif variable == "gsm":
            self.gsm_comp_plot()

    def vector_comp_plot(self):
        num_lines = 3
        x_values = self.data_set['epoch'].values
        y_data = self.data_set['vectors']

        x_values_dt = convert_j200_to_utc(x_values)

        fig, axes = plt.subplots(
            nrows=num_lines, ncols=1, figsize=(10, 3 * num_lines), sharex=True
        )

        x_comp = y_data.isel({'direction': 0})
        axes[0].plot(x_values_dt, x_comp)
        axes[0].set_ylabel(f"Vector {0}\n (x component)" )

        y_comp = y_data.isel({'direction': 1})
        axes[1].plot(x_values_dt, y_comp)
        axes[1].set_ylabel(f"Vector {1}\n (y component)")

        z_comp = y_data.isel({'direction': 2})
        axes[2].plot(x_values_dt, z_comp)
        axes[2].set_ylabel(f"Vector {2}\n (x component)")

        axes[-1].set_xlabel("Time (ns)")
        fig.suptitle("XYZ Component Vectors (Magnetometer)")
        plt.tight_layout()
        plt.show()

    def rtn_comp_plot(self):
        raise NotImplementedError

    def gse_comp_plot(self):
        raise NotImplementedError


class IdexQuicklookGenerator(QuicklookGenerator):
    """Idex subclass for Idex quicklook plots."""

    def two_dimensional_plot(self):
        self.idex_quicklook()

    def idex_quicklook(self, time_index: int = 0):

        num_plots = 6
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=False, constrained_layout=True)

        # TOF_Low
        axes[0].plot(self.data_set["time_high_sample_rate"].isel(epoch=time_index).data,
                     self.data_set["TOF_Low"].isel(epoch=time_index).data)
        axes[0].axvline(x=0, color="red", linestyle='--')
        axes[0].set_ylabel("TOF_Low")
        # TOF_Mid
        axes[1].plot(self.data_set["time_high_sample_rate"].isel(epoch=time_index).data,
                     self.data_set["TOF_Mid"].isel(epoch=time_index).data)
        axes[1].axvline(x=0, color="red", linestyle='--')
        axes[1].set_ylabel("TOF_Mid")
        # TOF_High
        axes[2].plot(self.data_set["time_high_sample_rate"].isel(epoch=time_index).data,
                     self.data_set["TOF_High"].isel(epoch=time_index).data)
        axes[2].axvline(x=0, color="red", linestyle='--')
        axes[2].set_ylabel("TOF_High")
        # Ion_grid
        axes[3].plot(self.data_set["time_low_sample_rate"].isel(epoch=time_index).data,
                     self.data_set["Ion_Grid"].isel(epoch=time_index).data)
        axes[3].axvline(x=0, color="red")
        axes[3].set_ylabel("Ion Grid")
        # Target Low
        axes[4].plot(self.data_set["time_low_sample_rate"].isel(epoch=time_index).data,
                     self.data_set["Target_Low"].isel(epoch=time_index).data)
        axes[4].axvline(x=0, color="red", linestyle='--')
        axes[4].set_ylabel("Target Low")
        # Target_high
        axes[5].plot(self.data_set["time_low_sample_rate"].isel(epoch=time_index).data,
                     self.data_set["Target_High"].isel(epoch=time_index).data)
        axes[5].axvline(x=0, color="red", linestyle='--')
        axes[5].set_ylabel("Target High")

        axes[-1].set_xlabel("Time (μs)")
        fig.suptitle("IDEX L1A Dust Impact Waveforms")
        plt.show()


class UltraQuicklookGenerator(QuicklookGenerator):
    """Ultra subclass for Idex quicklook plots."""

    def two_dimensional_plot(self):
        raise NotImplementedError

