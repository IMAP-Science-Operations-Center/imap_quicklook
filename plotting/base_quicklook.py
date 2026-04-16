"""Base class and shared utilities for quicklook plot generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import xarray as xr
from imap_data_access import ScienceFilePath

from plotting.cdf.cdf_utils import dataset_into_xarray

# Global variable for time conversion
TTJ2000_EPOCH = np.datetime64("2000-01-01T11:58:55.816", "ns")


def convert_j2000_to_utc(time_array: np.ndarray) -> np.ndarray:
    """
    Convert time from j2000 to utc.

    Parameters
    ----------
    time_array : np.ndarray
        Desired array to convert from j200 to UTC.

    Returns
    -------
    np.ndarray
        The newly converted UTC time array.
    """
    times = TTJ2000_EPOCH + time_array.astype("timedelta64[ns]")
    return times


@dataclass
class QuicklookGenerator(ABC):
    """
    General Quicklooks class.

    Parameters
    ----------
    file_name : str
        Xarray dataset holding CDF file info.

    Attributes
    ----------
    data_set : xr.Dataset, optional
        The xarray dataset for plotting.
    instrument : str, optional
        The instrument name derived from the file.
    x_variable : list of str, optional
        List of x-axis variable names.
    x_data : list of np.array, optional
        Data for x-axis.
    y_variable : list of str, optional
        List of y-axis variable names.
    y_data : list of np.array, optional
        Data for y-axis.
    title : str, optional
        Title of the plot.
    x_axis_label : str, optional
        Label for the x-axis.
    y_axis_label : str, optional
        Label for the y-axis.
    same_axes : bool, optional
        Whether all subplots share the same axes.
    """

    # Plot Info
    data_set: xr.Dataset | None = None
    instrument: str | None = None
    # TODO: The following 4 attributes are never used.
    x_variable: list[str] | None = None
    x_data: list[np.ndarray] | None = None
    y_variable: list[str] | None = None
    y_data: list[np.ndarray] | None = None

    # Making plots look pretty
    title: str | None = None
    x_axis_label: str | None = None
    y_axis_label: str | None = None
    same_axes: bool | None = None

    def __init__(self, file_name: str) -> None:
        # Plot Info
        self.data_set = dataset_into_xarray(file_name)
        if self.data_set is None:
            raise FileNotFoundError(f"Could not find file: {file_name}")

        file_name_dict = ScienceFilePath.extract_filename_components(file_name)
        self.instrument = file_name_dict["instrument"]

    class QuicklookGeneratorError(Exception):
        """Indicate that the QuicklookInput is invalid."""

        pass

    @abstractmethod
    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        raise NotImplementedError
