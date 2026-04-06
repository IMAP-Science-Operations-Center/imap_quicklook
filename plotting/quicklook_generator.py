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
        """Create xyz component quicklook for mag instrument in sensor coordinates."""
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        epoch_dt = convert_j2000_to_utc(self.data_set["epoch"].values)
        vector_data = self.data_set["vectors"]

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=True)

        for i, (ax, label) in enumerate(zip(axes, ["X", "Y", "Z"])):
            ax.plot(epoch_dt, vector_data.isel(direction=i))
            ax.set_ylabel(f"B_{label} [nT]")

        axes[-1].set_xlabel("Time (UTC)")
        fig.suptitle("MAG Sensor Coordinates (X, Y, Z) [nT]")
        plt.tight_layout()
        plt.show()

    def rtn_comp_plot(self) -> None:
        """Create rtn component quicklook for mag instrument."""
        raise NotImplementedError

    def gse_comp_plot(self) -> None:
        """Create GSE component + magnitude quicklook for mag instrument."""
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        epoch_dt = convert_j2000_to_utc(self.data_set["epoch"].values)
        b_gse = self.data_set["b_gse"]
        magnitude = self.data_set["magnitude"].values

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)

        for i, (ax, label) in enumerate(zip(axes[:3], ["X", "Y", "Z"])):
            ax.plot(epoch_dt, b_gse.isel(direction=i))
            ax.set_ylabel(f"B_{label} [nT]")

        axes[3].plot(epoch_dt, magnitude)
        axes[3].set_ylabel("|B| [nT]")
        axes[3].set_xlabel("Time (UTC)")

        fig.suptitle("MAG GSE Coordinates (X, Y, Z) + Magnitude [nT]")
        plt.tight_layout()
        plt.show()


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
            case "raw image events":
                self.raw_image_events()

    def raw_image_events(self) -> None:
        """Generate Ultra status plot."""
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
                self.swapi_counts()
            case "1d energy distribution":
                self.swapi_1d_energy_distribution()

    def swapi_count_rates(self) -> None:
        """Generate SWAPI plot of SW and PUI count rates per charge over time."""
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        # Time data
        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)

        # Energy values from strings to numbers
        energy_labels = self.data_set["esa_step_label"].values.astype(float)

        # Get count rate data
        # pcem and scem can be used to calculate solar wind or pick up ion rates respectively
        pcem_rate = self.data_set["swp_pcem_rate"].values  # Primary
        scem_rates = self.data_set["swp_scem_rate"].values  # Secondary
        coin_rates = self.data_set["swp_coin_rate"].values  # Coincidence

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)

        # Plot primary
        primary = axes[0].pcolormesh(
            epoch_dt, energy_labels, pcem_rate.T, shading="auto", cmap="viridis"
        )
        axes[0].set_title("SWAPI Primary Count Rate (PCEM)")
        axes[0].set_ylabel("Energy/charge [eV/q]")
        fig.colorbar(primary, ax=axes[0], label="Count Rate [Hz]")

        # Plot secondary
        secondary = axes[1].pcolormesh(
            epoch_dt, energy_labels, scem_rates.T, shading="auto", cmap="viridis"
        )
        axes[1].set_title("SWAPI Secondary Count Rate (SCEM)")
        axes[1].set_ylabel("Energy/charge [eV/q]")
        fig.colorbar(secondary, ax=axes[1], label="Count Rate [Hz]")

        # Plot coincidence
        coincidence = axes[2].pcolormesh(
            epoch_dt, energy_labels, coin_rates.T, shading="auto", cmap="viridis"
        )
        axes[2].set_title("SWAPI Coincidence Count Rate (COIN)")
        axes[2].set_xlabel("Time (UTC)")
        axes[2].set_ylabel("Energy/charge [eV/q]")
        fig.colorbar(coincidence, ax=axes[2], label="Count Rate [Hz]")

        plt.tight_layout()
        plt.show()

    def swapi_absolute_detection_efficiency(self) -> None:
        """
        Graph SWAPI absolute detection efficiency.

        Notes
        -----
        Calculate using coincidence^2 / (primary*secondary).
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        # Time data
        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)

        # Get count rate data
        # .sum(dim="esa_step") gives a total count for all energy steps at a particular time
        pcem_rate = self.data_set["swp_pcem_rate"]
        pcem_rate_sum = pcem_rate.sum(dim="esa_step")

        scem_rate = self.data_set["swp_scem_rate"]
        scem_rate_sum = scem_rate.sum(dim="esa_step")

        coin_rate = self.data_set["swp_coin_rate"]
        coin_rate_sum = coin_rate.sum(dim="esa_step")

        denom = pcem_rate_sum * scem_rate_sum
        detection_efficiency = (coin_rate_sum**2) / (denom)
        detection_efficiency = detection_efficiency.where(denom != 0)

        plt.plot(epoch_dt, detection_efficiency)
        plt.title("SWAPI Absolute Detection Efficiency")
        plt.xlabel("Time (UTC)")
        plt.ylabel("Coincidence² / (Primary × Secondary)")
        plt.tight_layout()
        plt.show()

    def swapi_1d_energy_distribution(self) -> None:
        """
        Generate SWAPI 1D energy distribution plot.

        Averages the coincidence count rate into 10-minute bins (starting from
        the first timestamp) and plots mean vs ESA energy for each interval.
        Only coarse ESA steps (indices 0–62) are included. Error bars show the
        standard deviation of samples within each bin.
        """
        import pandas as pd

        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        COARSE_STEPS = slice(0, 63)

        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)  # np.datetime64 array

        # Coarse-step COIN rate: shape (n_epoch, 63), cast to float to avoid
        # integer overflow when computing standard deviation
        coin_rate = self.data_set["swp_coin_rate"].isel(esa_step=COARSE_STEPS)
        coin_values = coin_rate.values.astype(float)

        # ESA energy axis from the label variable (first 63 entries)
        esa_energy = self.data_set["esa_step_label"].values[:63].astype(float)

        # Build 10-minute bins anchored to the first sample
        epoch_pd = pd.DatetimeIndex(epoch_dt.astype("datetime64[ns]"))
        t0 = epoch_pd[0]
        bin_edges = pd.date_range(
            start=t0,
            end=epoch_pd[-1] + pd.Timedelta("10min"),
            freq="10min",
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        for i in range(len(bin_edges) - 1):
            mask = (epoch_pd >= bin_edges[i]) & (epoch_pd < bin_edges[i + 1])
            if not mask.any():
                continue
            bin_data = coin_values[mask]  # shape: (n_in_bin, 63)
            mean_rate = bin_data.mean(axis=0)
            std_rate = bin_data.std(axis=0)
            label = bin_edges[i].strftime("%H:%M")
            ax.errorbar(
                esa_energy,
                mean_rate,
                yerr=std_rate,
                fmt="k.",
                capsize=3,
                label=label,
            )

        ax.set_xlabel("ESA Energy (eV/q)")
        ax.set_ylabel("COIN Rate (counts/s)")
        ax.set_title("SWAPI 1D Energy Distribution — 10-min averages (coarse steps)")
        ax.legend(
            title="Interval start (UTC)",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            borderaxespad=0,
        )
        plt.tight_layout()
        plt.show()

    def swapi_counts(self) -> None:
        """Generate SWAPI count rates line plot."""

        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        # Time data
        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)

        # Get count data — sum over all ESA steps for full-sweep totals
        pcem_counts = self.data_set["swp_pcem_counts"].sum(dim="esa_step")
        scem_counts = self.data_set["swp_scem_counts"].sum(dim="esa_step")
        coin_counts = self.data_set["swp_coin_counts"].sum(dim="esa_step")

        # sharey=False: each detector has a different count range
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=False)

        axes[0].plot(epoch_dt, pcem_counts, color="blue")
        axes[0].set_title("SWAPI PCEM Total Counts (full sweep)")
        axes[0].set_ylabel("Counts")

        axes[1].plot(epoch_dt, scem_counts, color="red")
        axes[1].set_title("SWAPI SCEM Total Counts (full sweep)")
        axes[1].set_ylabel("Counts")

        axes[2].plot(epoch_dt, coin_counts, color="green")
        axes[2].set_title("SWAPI COIN Total Counts (full sweep)")
        axes[2].set_xlabel("Time (UTC)")
        axes[2].set_ylabel("Counts")

        # Improve layout
        plt.tight_layout()
        plt.show()


class HiQuicklookGenerator(QuicklookGenerator):
    """Hi subclass for MAG quicklook plots."""

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        match variable:
            case "HI histogram":
                self.hi_histogram()
            case "DE Hisogram":
                self.de_histogram()
            case "DE TOF Plots":
                self.de_tof_plot()

    def hi_histogram(self) -> None:
        """TODO."""
        raise NotImplementedError

    def de_histogram(self) -> None:
        """TODO."""
        raise NotImplementedError

    def de_tof_plot(self) -> None:
        """TODO."""
        raise NotImplementedError


class LoQuicklookGenerator(QuicklookGenerator):
    """Hi subclass for MAG quicklook plots."""

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        match variable:
            case "star sensor":
                self.star_sensor()
            case "histogram":
                self.histogram()
            case "DE histogram":
                self.de_histogram()
            case "DE tof":
                self.de_tof()

    def star_sensor(self) -> None:
        """TODO."""
        raise NotImplementedError

    def histogram(self) -> None:
        """TODO."""
        raise NotImplementedError

    def de_histogram(self) -> None:
        """TODO."""
        raise NotImplementedError

    def de_tof(self) -> None:
        """TODO."""
        raise NotImplementedError


class GlowsQuicklookGenerator(QuicklookGenerator):
    """Hi subclass for MAG quicklook plots."""

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        match variable:
            case "general quicklook":
                self.general_quicklook()
            case "ancillary data":
                self.ancillary_data()

    def general_quicklook(self) -> None:
        """TODO."""
        raise NotImplementedError

    def ancillary_data(self) -> None:
        """TODO."""
        raise NotImplementedError


class HitQuicklookGenerator(QuicklookGenerator):
    """Hi subclass for MAG quicklook plots."""

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        match variable:
            case "hit ion flux":
                self.hit_ion_flux()
            case "electron count rate":
                self.electron_count_rate()

    def hit_ion_flux(self) -> None:
        """TODO."""
        raise NotImplementedError

    def electron_count_rate(self) -> None:
        """TODO."""
        raise NotImplementedError


class QuicklookGeneratorType(Enum):
    """Map instrument to correct dataclass."""

    MAG = MagQuicklookGenerator
    IDEX = IdexQuicklookGenerator
    ULTRA = UltraQuicklookGenerator
    SWAPI = SwapiQuicklookGenerator
    HI = HiQuicklookGenerator
    LO = LoQuicklookGenerator
    GLOWS = GlowsQuicklookGenerator
    HIT = HitQuicklookGenerator


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
