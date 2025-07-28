"""Class for abstracting and organizing quicklook plots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
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


class MagQuicklookGenerator(QuicklookGenerator):
    """MAG subclass for MAG quicklook plots."""

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        match variable:
            case "mag sensor co-ord":
                self.vector_component_plot()
            case "rtn":
                self.rtn_comp_plot()
            case "gse":
                self.gse_comp_plot()

    def vector_component_plot(self) -> None:
        """Create xyz component quicklook for mag instrument."""
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        num_lines = 3
        epoch = self.data_set["epoch"].values
        vector_data = self.data_set["vectors"]

        epoch_dt = convert_j2000_to_utc(epoch)

        fig, axes = plt.subplots(
            nrows=num_lines, ncols=1, figsize=(10, 3 * num_lines), sharex=True
        )

        x_comp = vector_data.isel({"direction": 0})
        axes[0].plot(epoch_dt, x_comp)
        axes[0].set_ylabel(f"Vector {0}\n (x component)")

        y_comp = vector_data.isel({"direction": 1})
        axes[1].plot(epoch_dt, y_comp)
        axes[1].set_ylabel(f"Vector {1}\n (y component)")

        z_comp = vector_data.isel({"direction": 2})
        axes[2].plot(epoch_dt, z_comp)
        axes[2].set_ylabel(f"Vector {2}\n (z component)")

        axes[-1].set_xlabel("Time (ns)")
        fig.suptitle("XYZ Component Vectors -- Magnetometer (nT)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def rtn_comp_plot(self) -> None:
        """Create rtn component quicklook for mag instrument."""
        raise NotImplementedError

    def gse_comp_plot(self) -> None:
        """Create gse component quicklook for mag instrument."""
        raise NotImplementedError


class IdexQuicklookGenerator(QuicklookGenerator):
    """IDEX subclass for IDEX quicklook plots."""

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
        Generate event based IDEX quicklook plot.

        Parameters
        ----------
        time_index : int
            Desired time index to generate quicklook from.
        """
        if self.data_set is None:
            raise RuntimeError("No data_set loaded.")

        num_plots = 6
        fig, axes = plt.subplots(
            num_plots,
            1,
            figsize=(10, 3 * num_plots),
            sharex=False,
            constrained_layout=True,
        )

        # TOF_Low
        y_data_tof_low = self.data_set["TOF_Low"].isel(epoch=time_index).data
        axes[0].plot(
            self.data_set["time_high_sample_rate"].isel(epoch=time_index).data,
            y_data_tof_low,
        )
        axes[0].axvline(x=0, color="red", linestyle="-")
        axes[0].set_ylabel("TOF_Low")
        text = (
            f"min = {y_data_tof_low.min():.3f}\n"
            f"max = {y_data_tof_low.max():.3f}\n"
            f"mean = {y_data_tof_low.mean():.3f}\n"
            f"std = {y_data_tof_low.std():.3f}"
        )
        axes[0].text(
            1.02,
            1,
            text,
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
        )
        # TOF_Mid
        y_data_tof_mid = self.data_set["TOF_Mid"].isel(epoch=time_index).data
        axes[1].plot(
            self.data_set["time_high_sample_rate"].isel(epoch=time_index).data,
            y_data_tof_mid,
        )
        axes[1].axvline(x=0, color="red", linestyle="-")
        axes[1].set_ylabel("TOF_Mid")
        text = (
            f"min = {y_data_tof_mid.min():.3f}\n"
            f"max = {y_data_tof_mid.max():.3f}\n"
            f"mean = {y_data_tof_mid.mean():.3f}\n"
            f"std = {y_data_tof_mid.std():.3f}"
        )
        axes[1].text(
            1.02,
            1,
            text,
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
        )
        # TOF_High
        y_data_tof_high = self.data_set["TOF_High"].isel(epoch=time_index).data
        axes[2].plot(
            self.data_set["time_high_sample_rate"].isel(epoch=time_index).data,
            y_data_tof_high,
        )
        axes[2].axvline(x=0, color="red", linestyle="-")
        axes[2].set_ylabel("TOF_High")
        text = (
            f"min = {y_data_tof_high.min():.3f}\n"
            f"max = {y_data_tof_high.max():.3f}\n"
            f"mean = {y_data_tof_high.mean():.3f}\n"
            f"std = {y_data_tof_high.std():.3f}"
        )
        axes[2].text(
            1.02,
            1,
            text,
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
        )
        # Ion_grid
        y_data_ion_grid = self.data_set["Ion_Grid"].isel(epoch=time_index).data
        axes[3].plot(
            self.data_set["time_low_sample_rate"].isel(epoch=time_index).data,
            self.data_set["Ion_Grid"].isel(epoch=time_index).data,
        )
        axes[3].axvline(x=0, color="red")
        axes[3].set_ylabel("Ion Grid")
        text = (
            f"min = {y_data_ion_grid.min():.3f}\n"
            f"max = {y_data_ion_grid.max():.3f}\n"
            f"mean = {y_data_ion_grid.mean():.3f}\n"
            f"std = {y_data_ion_grid.std():.3f}"
        )
        axes[3].text(
            1.02,
            1,
            text,
            transform=axes[3].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
        )
        # Target Low
        y_data_target_low = self.data_set["Target_Low"].isel(epoch=time_index).data
        axes[4].plot(
            self.data_set["time_low_sample_rate"].isel(epoch=time_index).data,
            y_data_target_low,
        )
        axes[4].axvline(x=0, color="red", linestyle="-")
        axes[4].set_ylabel("Target Low")
        text = (
            f"min = {y_data_target_low.min():.3f}\n"
            f"max = {y_data_target_low.max():.3f}\n"
            f"mean = {y_data_target_low.mean():.3f}\n"
            f"std = {y_data_target_low.std():.3f}"
        )
        axes[4].text(
            1.02,
            1,
            text,
            transform=axes[4].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
        )
        # Target_high
        y_data_target_high = self.data_set["Target_High"].isel(epoch=time_index).data
        axes[5].plot(
            self.data_set["time_low_sample_rate"].isel(epoch=time_index).data,
            y_data_target_high,
        )
        axes[5].axvline(x=0, color="red", linestyle="-")
        axes[5].set_ylabel("Target High")
        text = (
            f"min = {y_data_target_high.min():.3f}\n"
            f"max = {y_data_target_high.max():.3f}\n"
            f"mean = {y_data_target_high.mean():.3f}\n"
            f"std = {y_data_target_high.std():.3f}"
        )
        axes[5].text(
            1.02,
            1,
            text,
            transform=axes[5].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
        )

        axes[-1].set_xlabel("Time (μs)")
        fig.suptitle("IDEX L1A Dust Impact Waveforms")
        plt.show()


class UltraQuicklookGenerator(QuicklookGenerator):
    """ULTRA subclass for ULTRA quicklook plots."""

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        match variable:
            case "ultra status":
                self.ultra_status_plot()
            case "ultra hv":
                self.ultra_hv_plot()
            case "ultra general rates 1":
                self.ultra_general_rates_1_plot()
            case "ultra general rates 2":
                self.ultra_general_rates_2_plot()

    def ultra_status_plot(self) -> None:
        """Generate Ultra status plot."""
        raise NotImplementedError

    def ultra_hv_plot(self) -> None:
        """Generate Ultra hv plot."""
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        raise NotImplementedError

    def ultra_general_rates_1_plot(self) -> None:
        """Generate Ultra general rates 1 plot."""
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        raise NotImplementedError

    def ultra_general_rates_2_plot(self) -> None:
        """Generate Ultra general rates 2 plot."""
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        raise NotImplementedError


class SwapiQuicklookGenerator(QuicklookGenerator):
    """SWAPI subclass for SWAPI quicklook plots."""

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        match variable:
            case "count rates":
                self.swapi_count_rates()
            case "absolute detection efficiency":
                self.swapi_absolute_detection_efficiency()
            case "count line":
                self.swapi_count_line()

    def swapi_count_rates(self) -> None:
        """Generate SWAPI count rates plot."""
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        # Time data
        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)

        # Energy values from strings to numbers
        energy_labels = self.data_set["energy_label"].values.astype(float)

        # Get count rate data
        swp_rates = self.data_set["swp_pcem_rate"].values
        pui_rates = self.data_set["swp_scem_rate"].values

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

        # Plot SW
        pcm1 = axes[0].pcolormesh(
            epoch_dt, energy_labels, swp_rates.T, shading="auto", cmap="viridis"
        )
        axes[0].set_title("SWP PCEM Count Rate")
        axes[0].set_ylabel("Energy per charge (eV/q)")
        fig.colorbar(pcm1, ax=axes[0], label="Counts")

        # Plot PUI
        pcm2 = axes[1].pcolormesh(
            epoch_dt, energy_labels, pui_rates.T, shading="auto", cmap="viridis"
        )
        axes[1].set_title("SWP SCEM Count Rate")
        axes[1].set_xlabel("Time (UTC)")
        axes[1].set_ylabel("Energy per charge (eV/q)")
        fig.colorbar(pcm2, ax=axes[1], label="Counts")

        plt.tight_layout()
        plt.show()

    def swapi_absolute_detection_efficiency(self) -> None:
        """
        Graph SWAPI absolute detection efficiency.

        Notes:
        --------
            Calculate using coincidence^2 / (primary*secondary).
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        # Time data
        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)

        # Get count rate data
        pcem_rate = self.data_set["swp_pcem_rate"]
        pcem_rate_sum = pcem_rate.sum(dim="energy")

        scem_rate = self.data_set["swp_scem_rate"]
        scem_rate_sum = scem_rate.sum(dim="energy")

        coin_rate = self.data_set["swp_coin_rate"]
        coin_rate_sum = coin_rate.sum(dim="energy")

        denom = pcem_rate_sum * scem_rate_sum
        detection_efficiency = (coin_rate_sum**2) / (denom)
        detection_efficiency = detection_efficiency.where(denom != 0)

        plt.plot(epoch_dt, detection_efficiency)
        plt.title("SWAPI Absolute Detection Efficiency")
        plt.xlabel("Time (UTC)")
        plt.ylabel("Coincidence² / (Primary × Secondary)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def swapi_count_line(self) -> None:
        """Generate SWAPI count rates line plot."""

        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        # Time data
        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)

        # Get count data
        # TODO: Ensure energy=0 is correct
        pcem_counts = self.data_set["swp_pcem_rate"]
        pcem_counts_single = pcem_counts.isel(energy=0)
        # sw_total_counts = pcem_counts.sum(dim="energy")
        scem_counts = self.data_set["swp_scem_rate"]
        scem_counts_single = scem_counts.isel(energy=0)
        # pui_total_counts = scem_counts.sum(dim="energy")
        coin_counts = self.data_set["swp_coin_rate"]
        coin_counts_single = coin_counts.isel(energy=0)

        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)

        # Plot SW
        axes[0].plot(epoch_dt, pcem_counts_single, color="blue")
        axes[0].set_title("SWAPI PCEM Total Counts")
        axes[0].set_ylabel("Counts")

        # Plot PUI
        axes[1].plot(epoch_dt, scem_counts_single, color="red")
        axes[1].set_title("PUI SCEM Total Count Rate")
        axes[1].set_xlabel("Time (UTC)")
        axes[1].set_ylabel("Counts")

        # Plot Coin
        axes[2].plot(epoch_dt, coin_counts_single, color="green")
        axes[2].set_title("Coin Count Rate")
        axes[2].set_xlabel("Time (UTC)")
        axes[2].set_ylabel("Counts")

        # Improve layout
        plt.tight_layout()
        plt.show()


class QuicklookGeneratorType(Enum):
    """Map instrument to correct dataclass."""

    MAG = MagQuicklookGenerator
    IDEX = IdexQuicklookGenerator
    ULTRA = UltraQuicklookGenerator
    SWAPI = SwapiQuicklookGenerator


def get_instrument_quicklook(filename: str) -> QuicklookGenerator:
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
    file_name_dict = ScienceFilePath.extract_filename_components(filename)
    gen_cls = QuicklookGeneratorType[file_name_dict["instrument"].upper()]
    return gen_cls.value(filename)
