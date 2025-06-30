"""Class for abstracting and organizing quicklook plots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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


# TODO: Check the way this is handled.
def generate_instrument_quicklook(filename: str) -> QuicklookGenerator:
    """
    Determine which abstract class to use for a given file.

    Parameters
    ----------
    filename : str
        Desired file to generate quicklook from.

    Returns
    -------
    QuicklookGenerator
        Proper abstract class for file.
    """
    mission, instrument, level, descriptor, year_month, version_no = filename.split("_")
    generator_map = {"mag": MagQuicklookGenerator, "idex": IdexQuicklookGenerator}
    cls = generator_map.get(instrument)
    if cls is not None:
        return cls(filename)

    raise ValueError(
        f"Invalid input for {filename}. It does not match any file formats."
    )


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
        mission, self.instrument, level, descriptor, year_month, version_no = (
            file_name.split("_")
        )

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


class MagQuicklookGenerator(QuicklookGenerator):
    """Mag subclass for mag quicklook plots."""

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        if variable == "mag sensor co-ord":
            self.vector_comp_plot()
        elif variable == "rtn":
            self.rtn_comp_plot()
        elif variable == "gsm":
            self.gse_comp_plot()

    def vector_comp_plot(self) -> None:
        """Create xyz component quicklook for mag instrument."""
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        num_lines = 3
        x_values = self.data_set["epoch"].values
        y_data = self.data_set["vectors"]

        x_values_dt = convert_j2000_to_utc(x_values)

        fig, axes = plt.subplots(
            nrows=num_lines, ncols=1, figsize=(10, 3 * num_lines), sharex=True
        )

        x_comp = y_data.isel({"direction": 0})
        axes[0].plot(x_values_dt, x_comp)
        axes[0].set_ylabel(f"Vector {0}\n (x component)")

        y_comp = y_data.isel({"direction": 1})
        axes[1].plot(x_values_dt, y_comp)
        axes[1].set_ylabel(f"Vector {1}\n (y component)")

        z_comp = y_data.isel({"direction": 2})
        axes[2].plot(x_values_dt, z_comp)
        axes[2].set_ylabel(f"Vector {2}\n (z component)")

        axes[-1].set_xlabel("Time (ns)")
        fig.suptitle("XYZ Component Vectors (Magnetometer)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def rtn_comp_plot(self) -> None:
        """Create rtn component quicklook for mag instrument."""
        raise NotImplementedError

    def gse_comp_plot(self) -> None:
        """
        Create xyz component quicklook for mag instrument.

        Returns
        -------
        None
            This function returns nothing.
        """
        raise NotImplementedError


class IdexQuicklookGenerator(QuicklookGenerator):
    """Idex subclass for Idex quicklook plots."""

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        self.idex_quicklook()

    def idex_quicklook(self, time_index: int = 0) -> None:
        """
        Determine which abstract class to use for a given file.

        Parameters
        ----------
        time_index : int
            Desired time index to generate quicklook from.
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        num_plots = 6
        fig, axes = plt.subplots(
            num_plots,
            1,
            figsize=(10, 3 * num_plots),
            sharex=False,
            constrained_layout=True,
        )

        # TOF_Low
        axes[0].plot(
            self.data_set["time_high_sample_rate"].isel(epoch=time_index).data,
            self.data_set["TOF_Low"].isel(epoch=time_index).data,
        )
        axes[0].axvline(x=0, color="red", linestyle="--")
        axes[0].set_ylabel("TOF_Low")
        # TOF_Mid
        axes[1].plot(
            self.data_set["time_high_sample_rate"].isel(epoch=time_index).data,
            self.data_set["TOF_Mid"].isel(epoch=time_index).data,
        )
        axes[1].axvline(x=0, color="red", linestyle="--")
        axes[1].set_ylabel("TOF_Mid")
        # TOF_High
        axes[2].plot(
            self.data_set["time_high_sample_rate"].isel(epoch=time_index).data,
            self.data_set["TOF_High"].isel(epoch=time_index).data,
        )
        axes[2].axvline(x=0, color="red", linestyle="--")
        axes[2].set_ylabel("TOF_High")
        # Ion_grid
        axes[3].plot(
            self.data_set["time_low_sample_rate"].isel(epoch=time_index).data,
            self.data_set["Ion_Grid"].isel(epoch=time_index).data,
        )
        axes[3].axvline(x=0, color="red")
        axes[3].set_ylabel("Ion Grid")
        # Target Low
        axes[4].plot(
            self.data_set["time_low_sample_rate"].isel(epoch=time_index).data,
            self.data_set["Target_Low"].isel(epoch=time_index).data,
        )
        axes[4].axvline(x=0, color="red", linestyle="--")
        axes[4].set_ylabel("Target Low")
        # Target_high
        axes[5].plot(
            self.data_set["time_low_sample_rate"].isel(epoch=time_index).data,
            self.data_set["Target_High"].isel(epoch=time_index).data,
        )
        axes[5].axvline(x=0, color="red", linestyle="--")
        axes[5].set_ylabel("Target High")

        axes[-1].set_xlabel("Time (μs)")
        fig.suptitle("IDEX L1A Dust Impact Waveforms")
        plt.show()


class UltraQuicklookGenerator(QuicklookGenerator):
    """Ultra subclass for Idex quicklook plots."""

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        raise NotImplementedError
