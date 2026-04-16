"""Class for abstracting and organizing quicklook plots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
            case "priority 1 events":
                self.priority_event_spectrogram(priority=1)
            case "priority 2 events":
                self.priority_event_spectrogram(priority=2)
            case "tof spin spectrogram":
                self.tof_spin_spectrogram()
            case "tof spectrum":
                self.tof_spectrum()

    def raw_image_events(self) -> None:
        """
        Generate the ULTRA Raw Image Events quicklook (3 panels).

        Requires ``self.data_set`` to be the L1B DE dataset and
        ``self.data_set_aux`` to be the L1A AUX dataset.

        Panel 1 — Corrected TOF spectrogram (time × cTOF, counts colorbar)
        Panel 2 — Energy spectrogram (time × energy, counts colorbar)
        Panel 3 — Deflection voltage state from L1A AUX (left / right plates)
        """
        if self.data_set is None:
            raise ValueError("Must load a DE dataset.")

        de = self.data_set
        aux = getattr(self, "data_set_aux", None)

        fill = -1e31

        # --- L1B DE: corrected TOF and energy --------------------------------
        epoch_de = convert_j2000_to_utc(de["epoch"].values)

        tof_raw = de["tof_corrected"].values.astype(float)  # nanosecond / 10
        tof_ns = tof_raw / 10.0  # convert to ns
        energy = de["energy"].values.astype(float)  # keV

        # Mask fill and non-physical values
        valid_mask = (tof_raw > fill * 0.99) & (tof_ns > 0)
        energy_mask = (energy > fill * 0.99) & (energy > 0)

        t_de_ns = epoch_de.astype("datetime64[ns]").astype(np.int64)

        # Build 300-second time bins spanning the full DE epoch range
        bin_width_ns = int(300e9)  # 300 s in nanoseconds
        t_start = t_de_ns.min()
        t_end = t_de_ns.max() + bin_width_ns
        time_edges_ns = np.arange(t_start, t_end, bin_width_ns)
        time_edges = time_edges_ns.astype("datetime64[ns]")
        # Log-spaced TOF bins: 0.5 → 500 ns (70 bins/decade as per spec)
        tof_edges = np.logspace(np.log10(0.5), np.log10(500), 71)

        # Log-spaced energy bins: 0.5 → 5000 keV
        en_edges = np.logspace(np.log10(0.5), np.log10(5000), 71)

        # Bin DE events into 2D histograms using np.histogramdd
        t_vals = t_de_ns[valid_mask]
        tof_vals = tof_ns[valid_mask]
        tof_hist, _, _ = np.histogram2d(
            t_vals,
            tof_vals,
            bins=[time_edges_ns, tof_edges],
        )  # shape (n_tbins, n_tof)

        t_vals_e = t_de_ns[energy_mask]
        en_vals = energy[energy_mask]
        en_hist, _, _ = np.histogram2d(
            t_vals_e,
            en_vals,
            bins=[time_edges_ns, en_edges],
        )  # shape (n_tbins, n_en)

        # Mask zero bins for log colormap
        tof_hist = tof_hist.astype(float)
        en_hist = en_hist.astype(float)
        tof_hist[tof_hist == 0] = np.nan
        en_hist[en_hist == 0] = np.nan

        # Rainbow colormap: black for NaN (zero-count bins)
        cmap = plt.get_cmap("rainbow").copy()
        cmap.set_bad("black")
        cmap.set_under("black")

        # --- L1A AUX: deflection voltage state --------------------------------
        if aux is not None:
            epoch_aux = convert_j2000_to_utc(aux["epoch"].values)
            left_chrg = aux["leftdeflectioncharge"].values.astype(float)
            right_chrg = aux["rightdeflectioncharge"].values.astype(float)

        # --- Figure layout ----------------------------------------------------
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 10),
            sharex=False,
            constrained_layout=False,
        )
        fig.patch.set_facecolor("black")
        fig.subplots_adjust(hspace=0.35, right=0.84, top=0.92, bottom=0.08)

        sensor = "45"
        date_str = str(epoch_de[0])[:10]

        def _add_spectrogram(
            ax: plt.Axes,
            time_edges: np.ndarray,
            y_edges: np.ndarray,
            hist: np.ndarray,
            ylabel: str,
            title: str,
        ) -> None:
            """
            Plot a log-scale 2D histogram as a spectrogram.

            Parameters
            ----------
            ax : plt.Axes
                Axes to draw on.
            time_edges : np.ndarray
                Bin edges along the time axis.
            y_edges : np.ndarray
                Bin edges along the y axis (TOF or energy).
            hist : np.ndarray
                2D count histogram of shape (n_time, n_y).
            ylabel : str
                Label for the y axis.
            title : str
                Text annotation placed inside the panel.
            """
            valid = hist[np.isfinite(hist)]
            vmin = float(valid.min()) if len(valid) else 1.0
            vmax = float(valid.max()) if len(valid) else 1e4
            if vmin >= vmax:
                vmin, vmax = 1.0, 1e4

            ax.set_facecolor("black")
            im = ax.pcolormesh(
                time_edges,
                y_edges,
                hist.T,
                norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=cmap,
                shading="flat",
            )
            ax.set_yscale("log")
            ax.set_ylabel(ylabel, color="white", fontsize=9)
            ax.tick_params(colors="white", which="both")
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.text(
                0.99,
                0.97,
                title,
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

            cbar_ax = fig.add_axes(
                [
                    0.86,
                    ax.get_position().y0 + 0.005,
                    0.015,
                    ax.get_position().height - 0.01,
                ]
            )
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Counts", color="white", fontsize=8)
            cbar.ax.tick_params(colors="white", labelsize=7)

        # Panel 1: cTOF spectrogram
        _add_spectrogram(
            axes[0],
            time_edges,
            tof_edges,
            tof_hist,
            ylabel="cTOF [ns]",
            title=f"U{sensor} Corrected TOF",
        )

        # Panel 2: Energy spectrogram
        _add_spectrogram(
            axes[1],
            time_edges,
            en_edges,
            en_hist,
            ylabel="Energy [keV]",
            title=f"U{sensor} Energy",
        )

        # Panel 3: Deflection voltage state (L1A AUX)
        axes[2].set_facecolor("black")
        for spine in axes[2].spines.values():
            spine.set_color("white")
        axes[2].tick_params(colors="white")

        if aux is not None:
            axes[2].step(
                epoch_aux,
                left_chrg,
                where="post",
                color="steelblue",
                linewidth=1.2,
                label="Left plate",
            )
            axes[2].step(
                epoch_aux,
                right_chrg,
                where="post",
                color="tomato",
                linewidth=1.2,
                label="Right plate",
                linestyle="--",
            )
            axes[2].set_ylim(-0.1, 1.4)
            axes[2].set_yticks([0, 1])
            axes[2].set_yticklabels(["Off", "On"], color="white")
            axes[2].legend(
                fontsize=8,
                loc="upper right",
                facecolor="black",
                edgecolor="white",
                labelcolor="white",
            )
        else:
            axes[2].text(
                0.5,
                0.5,
                "L1A AUX data not loaded",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
                color="gray",
                fontsize=11,
            )

        axes[2].set_ylabel("Deflection\nVoltage", color="white", fontsize=9)

        # Shared x-axis formatting: apply UTC labels to all three panels
        for ax in axes:
            ax.set_xlim(time_edges[0], time_edges[-1])
            ax.tick_params(axis="x", colors="white", labelsize=8)
        axes[-1].set_xlabel("UTC Time", color="white", fontsize=10)
        axes[0].tick_params(axis="x", labelbottom=False)
        axes[1].tick_params(axis="x", labelbottom=False)

        fig.suptitle(
            f"ULTRA U{sensor} Raw Image Events — {date_str}",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        plt.show()

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_rainbow_cmap() -> mcolors.Colormap:
        """
        Return a rainbow colormap with black for zero/masked bins.

        Returns
        -------
        mcolors.Colormap
            Rainbow colormap with bad and under values set to black.
        """
        cmap = plt.get_cmap("rainbow").copy()
        cmap.set_bad("black")
        cmap.set_under("black")
        return cmap

    @staticmethod
    def _spectrogram_panel(
        fig: plt.Figure,
        ax: plt.Axes,
        time_edges: np.ndarray,
        y_edges: np.ndarray,
        hist: np.ndarray,
        ylabel: str,
        label: str,
        cbar_left: float,
        log_y: bool = True,
    ) -> None:
        """
        Render a single log-colorscale 2-D spectrogram panel.

        Parameters
        ----------
        fig : plt.Figure
            Parent figure (used to place the colorbar axes).
        ax : plt.Axes
            Axes to draw the spectrogram on.
        time_edges : np.ndarray
            Bin edges along the time axis.
        y_edges : np.ndarray
            Bin edges along the y axis (TOF or energy).
        hist : np.ndarray
            2D count histogram of shape (n_time, n_y).
        ylabel : str
            Label for the y axis.
        label : str
            Annotation text placed inside the upper-right corner.
        cbar_left : float
            Left position of the colorbar axes in figure coordinates.
        log_y : bool
            Whether to apply a log scale to the y axis.
        """
        cmap = UltraQuicklookGenerator._make_rainbow_cmap()
        valid = hist[np.isfinite(hist)]
        vmin = float(valid.min()) if len(valid) else 1.0
        vmax = float(valid.max()) if len(valid) else 1e4
        if vmin >= vmax:
            vmin, vmax = 1.0, 1e4

        ax.set_facecolor("black")
        im = ax.pcolormesh(
            time_edges,
            y_edges,
            hist.T,
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap,
            shading="flat",
        )
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        ax.tick_params(colors="white", which="both")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.text(
            0.99,
            0.97,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
        pos = ax.get_position()
        cbar_ax = fig.add_axes([cbar_left, pos.y0 + 0.005, 0.015, pos.height - 0.01])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Counts", color="white", fontsize=8)
        cbar.ax.tick_params(colors="white", labelsize=7)

    @staticmethod
    def _build_time_bins(
        epoch_ns: np.ndarray, bin_width_s: float = 300
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return time bin edges in both datetime64 and int64 nanosecond formats.

        Parameters
        ----------
        epoch_ns : np.ndarray
            Event timestamps in int64 nanoseconds.
        bin_width_s : float
            Bin width in seconds.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(time_edges_dt, time_edges_ns)`` where the first is datetime64
            and the second is int64 nanoseconds.
        """
        bw_ns = int(bin_width_s * 1e9)
        t0, t1 = int(epoch_ns.min()), int(epoch_ns.max()) + bw_ns
        edges_ns = np.arange(t0, t1, bw_ns)
        return edges_ns.astype("datetime64[ns]"), edges_ns

    # ------------------------------------------------------------------ #
    # Plot: Priority event spectrogram (slides 3)                         #
    # ------------------------------------------------------------------ #

    def priority_event_spectrogram(self, priority: int = 1) -> None:
        """
        Generate priority event spectrograms for ULTRA (slide 3 layout).

        Requires ``self.data_set`` to be the L1A priority-N-de dataset and
        ``self.data_set_aux`` to be the L1A AUX dataset.

        Panel 1 — Spin phase spectrogram (time × phase bin 0–719)
        Panel 2 — Energy PH spectrogram  (time × energy_ph ADC value)
        Panel 3 — Start rates: left (type 1) and right (type 2) events

        Parameters
        ----------
        priority : int
            Priority level to label the plot (1 or 2).
        """
        if self.data_set is None:
            raise ValueError("Must load a priority DE dataset.")

        ds = self.data_set
        fill32 = 4_294_967_295  # 2^32 − 1

        epoch_de = convert_j2000_to_utc(ds["epoch"].values)
        t_ns = epoch_de.astype("datetime64[ns]").astype(np.int64)

        # Spin phase: raw bin 0–719 (0.5° per bin)
        phase_raw = ds["phase_angle"].values.astype(np.int64)
        phase_mask = phase_raw < fill32
        phase_bins = phase_raw[phase_mask]  # 0–719
        t_phase = t_ns[phase_mask]

        # Energy pulse height
        eph = ds["energy_ph"].values.astype(np.int64)
        eph_mask = eph < fill32
        eph_vals = eph[eph_mask].astype(float)
        t_eph = t_ns[eph_mask]

        # Start type for left (1) / right (2) rate split
        start_type = ds["start_type"].values
        left_mask = (start_type == 1) & phase_mask
        right_mask = (start_type == 2) & phase_mask

        # --- time & y bins ------------------------------------------------
        time_edges_dt, time_edges_ns = self._build_time_bins(t_ns, bin_width_s=300)
        bin_width_s = 300.0

        phase_edges = np.arange(0, 721)  # 0..720, covers all 720 bins
        eph_min = max(eph_vals.min(), 1) if len(eph_vals) else 500
        eph_max = eph_vals.max() if len(eph_vals) else 5000
        eph_edges = np.linspace(eph_min, eph_max, 72)

        # --- 2-D histograms -----------------------------------------------
        phase_hist, _, _ = np.histogram2d(
            t_phase, phase_bins, bins=[time_edges_ns, phase_edges]
        )
        eph_hist, _, _ = np.histogram2d(
            t_eph, eph_vals, bins=[time_edges_ns, eph_edges]
        )
        phase_hist[phase_hist == 0] = np.nan
        eph_hist[eph_hist == 0] = np.nan

        # --- start rates per time bin (counts / bin_width_s = CPS) ---------
        left_rate = np.histogram(t_ns[left_mask], bins=time_edges_ns)[0] / bin_width_s
        right_rate = np.histogram(t_ns[right_mask], bins=time_edges_ns)[0] / bin_width_s
        bin_centres = (
            time_edges_dt[:-1].astype(np.int64) + time_edges_dt[1:].astype(np.int64)
        ) // 2
        bin_centres = bin_centres.astype("datetime64[ns]")

        # --- figure -------------------------------------------------------
        fig, axes = plt.subplots(
            3, 1, figsize=(14, 10), sharex=False, constrained_layout=False
        )
        fig.patch.set_facecolor("black")
        fig.subplots_adjust(hspace=0.35, right=0.84, top=0.92, bottom=0.08)

        sensor = "45"
        date_str = str(epoch_de[0])[:10]

        self._spectrogram_panel(
            fig,
            axes[0],
            time_edges_dt,
            phase_edges,
            phase_hist,
            ylabel="Spin Phase [bin]",
            label=f"U{sensor} Priority {priority} — Phase Map",
            cbar_left=0.86,
            log_y=False,
        )
        axes[0].set_ylim(0, 720)

        self._spectrogram_panel(
            fig,
            axes[1],
            time_edges_dt,
            eph_edges,
            eph_hist,
            ylabel="Energy PH [ADC]",
            label=f"U{sensor} Priority {priority} — Energy PH",
            cbar_left=0.86,
            log_y=False,
        )

        # Rate panel
        axes[2].set_facecolor("black")
        for spine in axes[2].spines.values():
            spine.set_color("white")
        axes[2].tick_params(colors="white")
        axes[2].step(
            bin_centres,
            left_rate,
            where="mid",
            color="steelblue",
            linewidth=1.0,
            label="Left start (type 1)",
        )
        axes[2].step(
            bin_centres,
            right_rate,
            where="mid",
            color="tomato",
            linewidth=1.0,
            label="Right start (type 2)",
        )
        axes[2].set_ylabel("Start Rate\n[counts s⁻¹]", color="white", fontsize=9)
        axes[2].set_yscale("log")
        axes[2].legend(
            fontsize=8,
            loc="upper right",
            facecolor="black",
            edgecolor="white",
            labelcolor="white",
        )
        axes[2].set_xlabel("UTC Time", color="white", fontsize=10)
        axes[2].tick_params(axis="x", colors="white", labelsize=8)

        for ax in axes:
            ax.set_xlim(time_edges_dt[0], time_edges_dt[-1])
        axes[0].tick_params(axis="x", labelbottom=False)
        axes[1].tick_params(axis="x", labelbottom=False)

        fig.suptitle(
            f"ULTRA U{sensor} Priority {priority} Events — {date_str}",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        plt.show()

    # ------------------------------------------------------------------ #
    # Plot: TOF × spin-phase spectrograms side by side (slide 4)          #
    # ------------------------------------------------------------------ #

    def tof_spin_spectrogram(self) -> None:
        """
        Generate side-by-side TOF and spin-phase spectrograms from L1B DE.

        Left  — cTOF spectrogram  (time × tof_corrected in ns, log y)
        Right — Spin-phase spectrogram (time × phase bin 0–719, linear y)
        """
        if self.data_set is None:
            raise ValueError("Must load a L1B DE dataset.")

        ds = self.data_set
        fill = -1e31
        fill32 = 4_294_967_295

        epoch_de = convert_j2000_to_utc(ds["epoch"].values)
        t_ns = epoch_de.astype("datetime64[ns]").astype(np.int64)

        tof_raw = ds["tof_corrected"].values.astype(float)
        tof_ns = tof_raw / 10.0
        tof_mask = (tof_raw > fill * 0.99) & (tof_ns > 0)

        phase_raw = ds["phase_angle"].values.astype(np.int64)
        phase_mask = phase_raw < fill32

        time_edges_dt, time_edges_ns = self._build_time_bins(t_ns, bin_width_s=300)

        tof_edges = np.logspace(np.log10(0.5), np.log10(500), 71)
        phase_edges = np.arange(0, 721)

        tof_hist, _, _ = np.histogram2d(
            t_ns[tof_mask],
            tof_ns[tof_mask],
            bins=[time_edges_ns, tof_edges],
        )
        phase_hist, _, _ = np.histogram2d(
            t_ns[phase_mask],
            phase_raw[phase_mask],
            bins=[time_edges_ns, phase_edges],
        )
        tof_hist[tof_hist == 0] = np.nan
        phase_hist[phase_hist == 0] = np.nan

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=False)
        fig.patch.set_facecolor("black")
        fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.12, wspace=0.35)

        sensor = "45"
        date_str = str(epoch_de[0])[:10]

        self._spectrogram_panel(
            fig,
            axes[0],
            time_edges_dt,
            tof_edges,
            tof_hist,
            ylabel="cTOF [ns]",
            label=f"U{sensor} cTOF Spectrogram",
            cbar_left=0.435,
            log_y=True,
        )
        axes[0].set_xlabel("UTC Time", color="white", fontsize=9)
        axes[0].tick_params(axis="x", colors="white", labelsize=8)
        axes[0].set_xlim(time_edges_dt[0], time_edges_dt[-1])

        self._spectrogram_panel(
            fig,
            axes[1],
            time_edges_dt,
            phase_edges,
            phase_hist,
            ylabel="Spin Phase [bin]",
            label=f"U{sensor} Spin Phase Spectrogram",
            cbar_left=0.915,
            log_y=False,
        )
        axes[1].set_ylim(0, 720)
        axes[1].set_xlabel("UTC Time", color="white", fontsize=9)
        axes[1].tick_params(axis="x", colors="white", labelsize=8)
        axes[1].set_xlim(time_edges_dt[0], time_edges_dt[-1])

        fig.suptitle(
            f"ULTRA U{sensor} TOF & Spin Phase Spectrograms — {date_str}",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        plt.show()

    # ------------------------------------------------------------------ #
    # Plot: 1-D TOF spectrum (slide 4 diagnostic)                         #
    # ------------------------------------------------------------------ #

    def tof_spectrum(self) -> None:
        """
        Generate a 1-D TOF spectrum (counts vs cTOF in ns) from L1B DE.

        The ~10 ns dip diagnostic is annotated: a dip near 10 ns indicates
        clean ENA data; a dip shifted to >10 ns suggests SEP contamination.
        """
        if self.data_set is None:
            raise ValueError("Must load a L1B DE dataset.")

        ds = self.data_set
        fill = -1e31

        epoch_de = convert_j2000_to_utc(ds["epoch"].values)
        tof_raw = ds["tof_corrected"].values.astype(float)
        tof_ns = tof_raw / 10.0
        valid = (tof_raw > fill * 0.99) & (tof_ns > 0)
        tof_valid = tof_ns[valid]

        tof_edges = np.logspace(np.log10(0.5), np.log10(500), 150)
        counts, _ = np.histogram(tof_valid, bins=tof_edges)
        bin_centres = np.sqrt(tof_edges[:-1] * tof_edges[1:])  # geometric mean

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.plot(bin_centres, counts, color="white", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Corrected TOF [ns]", color="white", fontsize=10)
        ax.set_ylabel("Counts", color="white", fontsize=10)
        ax.tick_params(colors="white", which="both")
        for spine in ax.spines.values():
            spine.set_color("white")

        # Annotate the 10 ns dip diagnostic
        ax.axvline(10, color="tomato", linewidth=1.2, linestyle="--")
        ax.text(
            10.5,
            ax.get_ylim()[1] * 0.5,
            "Dip @ ~10 ns\n(ENA-clean if dip here)",
            color="tomato",
            fontsize=9,
            va="top",
        )

        sensor = "45"
        date_str = str(epoch_de[0])[:10]
        ax.set_title(
            f"ULTRA U{sensor} 1-D TOF Spectrum — {date_str}", color="white", fontsize=12
        )
        ax.set_xlim(tof_edges[0], tof_edges[-1])
        plt.tight_layout()
        plt.show()


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
            case "count rate line":
                self.swapi_count_rate_line()
            case "1d energy distribution":
                self.swapi_1d_energy_distribution()

    def swapi_count_rates(self) -> None:
        """Generate SWAPI plot of SW and PUI count rates per charge over time."""
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        # Time data
        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)

        # Energy axis: esa_energy is (time × 72) but constant across time
        energy = self.data_set["esa_energy"].values[0]  # shape (72,), eV/q

        # Get count rate data; mask fill values (-1e31)
        fill = -1e31

        def _mask_fill(arr: np.ndarray) -> np.ndarray:
            """
            Replace fill values with NaN.

            Parameters
            ----------
            arr : np.ndarray
                Input array potentially containing fill values (-1e31).

            Returns
            -------
            np.ndarray
                Float array with fill values replaced by NaN.
            """
            out = arr.astype(float).copy()
            out[out <= fill * 0.9] = np.nan
            return out

        pcem_rate = _mask_fill(self.data_set["swp_pcem_rate"].values)
        scem_rates = _mask_fill(self.data_set["swp_scem_rate"].values)
        coin_rates = _mask_fill(self.data_set["swp_coin_rate"].values)

        # Determine shared color scale from valid data
        all_rates = np.concatenate(
            [
                pcem_rate[np.isfinite(pcem_rate)],
                scem_rates[np.isfinite(scem_rates)],
                coin_rates[np.isfinite(coin_rates)],
            ]
        )
        vmin = (
            float(np.nanmin(all_rates[all_rates > 0])) if np.any(all_rates > 0) else 1.0
        )
        vmax = float(np.nanmax(all_rates)) if all_rates.size else 1e4
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)

        # Plot primary
        primary = axes[0].pcolormesh(
            epoch_dt, energy, pcem_rate.T, shading="auto", cmap="viridis", norm=norm
        )
        axes[0].set_title("SWAPI Primary Count Rate (PCEM)")
        axes[0].set_ylabel("Energy/charge [eV/q]")
        axes[0].set_yscale("log")
        fig.colorbar(primary, ax=axes[0], label="Count Rate [Hz]")

        # Plot secondary
        secondary = axes[1].pcolormesh(
            epoch_dt, energy, scem_rates.T, shading="auto", cmap="viridis", norm=norm
        )
        axes[1].set_title("SWAPI Secondary Count Rate (SCEM)")
        axes[1].set_ylabel("Energy/charge [eV/q]")
        axes[1].set_yscale("log")
        fig.colorbar(secondary, ax=axes[1], label="Count Rate [Hz]")

        # Plot coincidence
        coincidence = axes[2].pcolormesh(
            epoch_dt, energy, coin_rates.T, shading="auto", cmap="viridis", norm=norm
        )
        axes[2].set_title("SWAPI Coincidence Count Rate (COIN)")
        axes[2].set_xlabel("Time (UTC)")
        axes[2].set_ylabel("Energy/charge [eV/q]")
        axes[2].set_yscale("log")
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

        # Get count rate data; mask fill values (-1e31) before summing
        fill = -1e31
        ds = self.data_set

        def _masked_sum(var: str) -> xr.DataArray:
            """
            Return per-epoch sum over ESA steps with fill values excluded.

            Parameters
            ----------
            var : str
                Name of the dataset variable to sum.

            Returns
            -------
            xr.DataArray
                Array of shape (n_epoch,) with ESA-step-summed values.
            """
            return ds[var].where(ds[var] > fill * 0.9).sum(dim="esa_step", min_count=1)

        pcem_rate_sum = _masked_sum("swp_pcem_rate")
        scem_rate_sum = _masked_sum("swp_scem_rate")
        coin_rate_sum = _masked_sum("swp_coin_rate")

        denom = pcem_rate_sum * scem_rate_sum
        detection_efficiency = (coin_rate_sum**2) / denom
        detection_efficiency = detection_efficiency.where(denom > 0)

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
        FILL = -1e31

        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)  # np.datetime64 array

        # Coarse-step COIN rate: shape (n_epoch, 63); mask fill values
        coin_raw = (
            self.data_set["swp_coin_rate"]
            .isel(esa_step=COARSE_STEPS)
            .values.astype(float)
        )
        coin_raw[coin_raw <= FILL * 0.9] = np.nan

        # Actual ESA energy in eV/q — 2D (time × step), constant across time
        esa_energy = self.data_set["esa_energy"].values[0, :63]  # shape (63,)

        # Build 10-minute bins anchored to the first sample
        epoch_pd = pd.DatetimeIndex(epoch_dt.astype("datetime64[ns]"))
        t0 = epoch_pd[0]
        bin_edges = pd.date_range(
            start=t0,
            end=epoch_pd[-1] + pd.Timedelta("10min"),
            freq="10min",
        )

        n_bins = len(bin_edges) - 1
        cmap = plt.get_cmap("plasma", n_bins)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        plotted = 0
        for i in range(n_bins):
            mask = (epoch_pd >= bin_edges[i]) & (epoch_pd < bin_edges[i + 1])
            if not mask.any():
                continue
            bin_data = coin_raw[mask]  # shape: (n_in_bin, 63)
            mean_rate = np.nanmean(bin_data, axis=0)
            std_rate = np.nanstd(bin_data, axis=0)

            # Only plot steps where the mean is positive (log scale requirement)
            valid = mean_rate > 0
            color = cmap(i / max(n_bins - 1, 1))

            # Connecting line drawn first (underneath the dots)
            ax.plot(
                esa_energy[valid],
                mean_rate[valid],
                color=color,
                linewidth=0.8,
                alpha=0.6,
                zorder=1,
            )
            # Dots with uncapped error bars
            ax.errorbar(
                esa_energy[valid],
                mean_rate[valid],
                yerr=std_rate[valid],
                fmt=".",
                color=color,
                markersize=5,
                capsize=0,
                elinewidth=0.8,
                ecolor=color,
                zorder=2,
            )
            plotted += 1

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ESA energy (eV/q)")
        ax.set_ylabel("Count Rate (Hz)")
        ax.set_title(
            "SWAPI 1D Energy Distribution — 10-min averages (coarse steps 0–62)"
        )

        # Colorbar showing time progression
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=0, vmax=n_bins - 1),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("10-min interval index")
        tick_indices = np.linspace(0, n_bins - 1, min(6, n_bins)).astype(int)
        cbar.set_ticks(tick_indices)
        cbar.set_ticklabels([bin_edges[j].strftime("%H:%M") for j in tick_indices])

        # Box-style axes with inward ticks on all four sides
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")

        ax.tick_params(which="both", direction="in", top=True, right=True)
        plt.tight_layout()
        plt.show()

    def swapi_counts(self) -> None:
        """
        Generate SWAPI L1 count line plot (full sweep, ~12 s cadence, 5 sweeps/min).

        Each time point is one full 72-step sweep summed across all ESA steps.
        Fill values (-1e31) are masked before summing so a single bad step does
        not corrupt the sweep total.
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        # Time data
        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)

        # Mask fill values then sum over all ESA steps for full-sweep totals.
        # Using where() preserves NaN semantics so bad steps are skipped.
        fill = -1e31
        ds = self.data_set

        def _masked_sum(var: str) -> np.ndarray:
            """
            Return per-epoch ESA-step sum with fill values excluded.

            Parameters
            ----------
            var : str
                Name of the dataset variable to sum.

            Returns
            -------
            np.ndarray
                Array of shape (n_epoch,) with ESA-step-summed values.
            """
            return (
                ds[var]
                .where(ds[var] > fill * 0.9)
                .sum(dim="esa_step", min_count=1)
                .values
            )

        pcem_counts = _masked_sum("swp_pcem_counts")
        scem_counts = _masked_sum("swp_scem_counts")
        coin_counts = _masked_sum("swp_coin_counts")

        # sharey=False: each detector has a different count range
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=False)

        axes[0].plot(epoch_dt, pcem_counts, color="steelblue", linewidth=0.8)
        axes[0].set_title(
            "SWAPI PCEM Counts — L1 full sweep (~12 s cadence, 5 sweeps/min)"
        )
        axes[0].set_ylabel("Counts")

        axes[1].plot(epoch_dt, scem_counts, color="tomato", linewidth=0.8)
        axes[1].set_title("SWAPI SCEM Counts — L1 full sweep")
        axes[1].set_ylabel("Counts")

        axes[2].plot(epoch_dt, coin_counts, color="seagreen", linewidth=0.8)
        axes[2].set_title("SWAPI COIN Counts — L1 full sweep")
        axes[2].set_xlabel("Time (UTC)")
        axes[2].set_ylabel("Counts")

        plt.tight_layout()
        plt.show()

    def swapi_count_rate_line(self) -> None:
        """
        Generate SWAPI L2 count rate line plot (full sweep, ~12 s cadence, 5 sweeps/min).

        Each time point is one full 72-step sweep's count rate summed across all
        ESA energy steps.  Fill values (-1e31) are masked before summing.
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        epoch = self.data_set["epoch"].values
        epoch_dt = convert_j2000_to_utc(epoch)

        fill = -1e31
        ds = self.data_set

        def _masked_sum(var: str) -> np.ndarray:
            """
            Return per-epoch ESA-step sum with fill values excluded.

            Parameters
            ----------
            var : str
                Name of the dataset variable to sum.

            Returns
            -------
            np.ndarray
                Array of shape (n_epoch,) with ESA-step-summed values.
            """
            return (
                ds[var]
                .where(ds[var] > fill * 0.9)
                .sum(dim="esa_step", min_count=1)
                .values
            )

        pcem_rate = _masked_sum("swp_pcem_rate")
        scem_rate = _masked_sum("swp_scem_rate")
        coin_rate = _masked_sum("swp_coin_rate")

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=False)

        axes[0].plot(epoch_dt, pcem_rate, color="steelblue", linewidth=0.8)
        axes[0].set_title(
            "SWAPI PCEM Count Rate — L2 full sweep (~12 s cadence, 5 sweeps/min)"
        )
        axes[0].set_ylabel("Count Rate [Hz]")

        axes[1].plot(epoch_dt, scem_rate, color="tomato", linewidth=0.8)
        axes[1].set_title("SWAPI SCEM Count Rate — L2 full sweep")
        axes[1].set_ylabel("Count Rate [Hz]")

        axes[2].plot(epoch_dt, coin_rate, color="seagreen", linewidth=0.8)
        axes[2].set_title("SWAPI COIN Count Rate — L2 full sweep")
        axes[2].set_xlabel("Time (UTC)")
        axes[2].set_ylabel("Count Rate [Hz]")

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

    @staticmethod
    def _identify_columns(
        esa_steps: np.ndarray, epoch_dt: np.ndarray
    ) -> tuple[list[dict[int, int]], list[np.datetime64]]:
        """
        Group a sequence of ESA-step records into pointing columns.

        A new column starts at every ESA 1 record or any non-consecutive ESA jump.

        Parameters
        ----------
        esa_steps : np.ndarray
            ESA step value (1–9) for each record.
        epoch_dt : np.ndarray
            UTC timestamp for each record (same length as ``esa_steps``).

        Returns
        -------
        columns : list of dict
            Each entry maps ``esa_step → record_index`` for one column.
        col_times : list of np.datetime64
            Start time of each column (epoch of its first record).
        """
        columns: list[dict[int, int]] = []
        col_times: list[np.datetime64] = []
        current_col: dict[int, int] = {}
        col_start_time: np.datetime64 | None = None
        prev_esa: int | None = None

        for i in range(len(esa_steps)):
            esa = int(esa_steps[i])
            new_col = (prev_esa is None) or (esa == 1) or (esa != prev_esa + 1)

            if new_col:
                if current_col:
                    columns.append(current_col)
                    col_times.append(col_start_time)  # type: ignore[arg-type]
                current_col = {}
                col_start_time = epoch_dt[i]  # type: ignore[assignment]

            current_col[esa] = i
            prev_esa = esa

        if current_col:
            columns.append(current_col)
            col_times.append(col_start_time)  # type: ignore[arg-type]

        return columns, col_times

    @staticmethod
    def _plot_esa_grid(
        grid: np.ndarray,
        col_times: list[np.datetime64],
        title: str,
    ) -> None:
        """
        Render the stacked ESA-step pcolormesh plot.

        Parameters
        ----------
        grid : np.ndarray
            Counts array of shape ``(9, n_cols, 90)``.
        col_times : list of np.datetime64
            Start time of each column.
        title : str
            Figure title.
        """
        n_esa, n_cols, _ = grid.shape
        angle_edges = np.arange(0, 361, 4)

        rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, 256))
        rainbow_colors[0] = [1.0, 1.0, 1.0, 1.0]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rainbow_white", rainbow_colors
        )
        vmax = float(grid.max()) or 1.0

        col_times_arr = np.array(col_times, dtype="datetime64[ns]")
        if n_cols > 1:
            dt = (col_times_arr[-1] - col_times_arr[-2]).astype("timedelta64[ns]")
        else:
            dt = np.timedelta64(120, "s").astype("timedelta64[ns]")
        time_edges = np.append(col_times_arr, col_times_arr[-1] + dt)

        fig, axes = plt.subplots(
            n_esa, 1, figsize=(14, 2 * n_esa), sharex=True, sharey=True
        )
        fig.subplots_adjust(hspace=0, right=0.88)

        im = None
        for esa_i, ax in enumerate(axes):
            im = ax.pcolormesh(
                time_edges,
                angle_edges,
                grid[esa_i].T,
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                shading="flat",
            )
            ax.set_ylabel(f"ESA {esa_i + 1}\nPhase (°)", fontsize=8)
            ax.set_yticks([0, 90, 180, 270, 360])
            ax.tick_params(axis="y", labelsize=7)
            if esa_i < n_esa - 1:
                ax.tick_params(axis="x", labelbottom=False)

        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Counts")
        axes[0].set_title(title)
        axes[-1].set_xlabel("Time (UTC)")
        plt.show()

    def hi_histogram(self, histogram_types: list[str] | None = None) -> None:
        """
        Generate Hi L1A histogram quicklook plot for an entire pointing.

        Plots counts per 4-degree spin-phase bin vs. time, stacked by ESA step
        (ESA 1 at top, ESA 9 at bottom). A new column starts at each new ESA 1
        packet or any non-consecutive ESA jump. Missing ESA entries are zero.
        No interpolation or averaging is performed.

        Parameters
        ----------
        histogram_types : list of str, optional
            Names of dataset variables to sum for the displayed counts.
            Defaults to all ``*_qualified`` variables.
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        ds = self.data_set

        if histogram_types is None:
            histogram_types = [v for v in ds.data_vars if v.endswith("_qualified")]

        if not histogram_types:
            raise ValueError("No histogram type variables found in dataset.")

        # Sum selected types: shape (n_epoch, 90)
        counts = sum(ds[ht].values.astype(np.float64) for ht in histogram_types)

        epoch_dt = convert_j2000_to_utc(ds["epoch"].values)
        esa_steps = ds["esa_step"].values

        columns, col_times = self._identify_columns(esa_steps, epoch_dt)

        n_cols = len(columns)
        grid = np.zeros((9, n_cols, 90))
        for col_idx, col in enumerate(columns):
            for esa_step, pkt_idx in col.items():
                grid[esa_step - 1, col_idx, :] = counts[pkt_idx, :]

        self._plot_esa_grid(
            grid,
            col_times,
            f"Hi L1A Histogram — {self.instrument} — {'+'.join(histogram_types)}",
        )

    def de_histogram(self) -> None:
        """
        Generate Hi L1B DE histogram quicklook plot for an entire pointing.

        Direct events are binned into 4-degree spin-phase bins using the
        ``nominal_bin`` variable, grouped by ESA step and pointing column via
        ``ccsds_index``. Qualified coincidence types [3, 6, 7, 10, 11, 12, 14, 15]
        are summed. The layout mirrors the L1A histogram plot.
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        ds = self.data_set

        # --- Build ccsds_edges: boundaries where esa_step changes between packets ---
        esa_steps_pkt = ds["esa_step"].values  # shape (n_packets,)
        epoch_pkt = convert_j2000_to_utc(ds["epoch"].values)

        change_idxs = np.where(np.diff(esa_steps_pkt) != 0)[0] + 1
        ccsds_edges = np.concatenate([[0], change_idxs, [len(esa_steps_pkt)]])

        # ESA step and start time for each ccsds bin
        bin_esa = esa_steps_pkt[ccsds_edges[:-1]]
        bin_epoch = epoch_pkt[ccsds_edges[:-1]]

        # --- 3D histogram over (ccsds_bin, coincidence_type, nominal_bin) ---
        # Qualified coincidence types: [3, 6, 7, 10, 11, 12, 14, 15]
        coincidence_type_edges = np.array([3, 6, 7, 10, 11, 12, 14, 15, 16])
        bin_edges = np.arange(91)

        de_hist, _ = np.histogramdd(
            [
                ds["ccsds_index"].values,
                ds["coincidence_type"].values,
                ds["nominal_bin"].values,
            ],
            bins=(ccsds_edges, coincidence_type_edges, bin_edges),
        )
        # Sum over coincidence types → shape (n_ccsds_bins, 90)
        counts_2d = de_hist.sum(axis=1)

        # --- Build columns using same logic as hi_histogram ---
        columns, col_times = self._identify_columns(bin_esa, bin_epoch)

        n_cols = len(columns)
        grid = np.zeros((9, n_cols, 90))
        for col_idx, col in enumerate(columns):
            for esa_step, bin_idx in col.items():
                grid[esa_step - 1, col_idx, :] += counts_2d[bin_idx, :]

        self._plot_esa_grid(
            grid,
            col_times,
            f"Hi L1B DE Histogram — {self.instrument} — qualified coincidence types",
        )

    def de_tof_plot(self) -> None:
        """
        Generate Hi L1B DE TOF diagnostic quicklook plots.

        Produces three sets of figures per sensor:

        1. Four 1D log-scale TOF histograms (one figure per TOF pair):
           tB-tA, tC1-tA, tC1-tB (1 ns bins) and tC2-tC1 (0.5 ns bins).
           Each figure has 10 subplots — one per ESA step plus a total.
           A double-coincidence-only overplot is shown in orange.

        2. One figure of 2D histograms (tB-tA vs tC1-tA, 2 ns bins) with 10
           panels (per ESA step + total).  Log rainbow+white colorbar; black
           for zero-count bins.

        3. One 2D histogram (tB-tA vs tC2-tC1, 2 ns × 1 ns bins) summed over
           all ESA steps.  Same colormap.
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        ds = self.data_set
        fill = int(ds["tof_ab"].attrs.get("FILLVAL", -2147483648))

        # Per-event ESA step (via packet lookup)
        event_esa = ds["esa_step"].values[ds["ccsds_index"].values]

        tof_ab_v = ds["tof_ab"].values.astype(np.int32)
        tof_ac1_v = ds["tof_ac1"].values.astype(np.int32)
        tof_bc1_v = ds["tof_bc1"].values.astype(np.int32)
        tof_c1c2_v = ds["tof_c1c2"].values.astype(np.int32)

        def _valid(arr: np.ndarray, vmin: int = -512, vmax: int = 512) -> np.ndarray:
            """
            Return boolean mask selecting non-fill values within [vmin, vmax].

            Parameters
            ----------
            arr : np.ndarray
                Input array to check.
            vmin : int
                Minimum valid value (inclusive).
            vmax : int
                Maximum valid value (inclusive).

            Returns
            -------
            np.ndarray
                Boolean array of the same shape as ``arr``.
            """
            return (arr != fill) & (arr >= vmin) & (arr <= vmax)

        m_ab = _valid(tof_ab_v)
        m_ac1 = _valid(tof_ac1_v)
        m_bc1 = _valid(tof_bc1_v)
        m_c1c2 = _valid(tof_c1c2_v, vmin=0, vmax=511)

        # Double-coincidence-only masks: the one TOF valid, no other ABC hits
        dc_ab = m_ab & ~m_ac1 & ~m_bc1
        dc_ac1 = m_ac1 & ~m_ab & ~m_bc1
        dc_bc1 = m_bc1 & ~m_ab & ~m_ac1
        dc_c1c2 = m_c1c2 & ~m_ab & ~m_ac1 & ~m_bc1

        # Log rainbow+white colormap; black for empty 2D bins (NaN)
        rc = plt.cm.rainbow(np.linspace(0, 1, 255))
        log_cmap = mcolors.LinearSegmentedColormap.from_list(
            "log_rainbow_white", np.vstack([[[1, 1, 1, 1]], rc])
        )
        log_cmap.set_bad("black")

        # ------------------------------------------------------------------ #
        # Figure set 1 — 1D TOF histograms                                   #
        # ------------------------------------------------------------------ #
        tof_1d = [
            (tof_ab_v, m_ab, dc_ab, "tB - tA", np.arange(-512, 513, 1)),
            (tof_ac1_v, m_ac1, dc_ac1, "tC1 - tA", np.arange(-512, 513, 1)),
            (tof_bc1_v, m_bc1, dc_bc1, "tC1 - tB", np.arange(-512, 513, 1)),
            (tof_c1c2_v, m_c1c2, dc_c1c2, "tC2 - tC1", np.arange(0, 512, 0.5)),
        ]

        for tof_vals, valid, dc_only, label, edges in tof_1d:
            fig, axes = plt.subplots(
                10, 1, figsize=(10, 18), sharex=True, gridspec_kw={"hspace": 0}
            )

            esa_labels = [f"ESA {e}" for e in range(1, 10)] + ["All ESA"]
            esa_masks = [event_esa == e for e in range(1, 10)] + [
                np.ones(len(event_esa), dtype=bool)
            ]

            for ax, row_label, row_mask in zip(axes, esa_labels, esa_masks):
                all_vals = tof_vals[valid & row_mask]
                counts, _ = np.histogram(all_vals, bins=edges)
                ax.stairs(
                    counts,
                    edges,
                    fill=True,
                    color="steelblue",
                    alpha=0.7,
                    linewidth=0.5,
                )

                dc_vals = tof_vals[dc_only & row_mask]
                if dc_vals.size:
                    dc_counts, _ = np.histogram(dc_vals, bins=edges)
                    ax.stairs(
                        dc_counts, edges, color="darkorange", linewidth=1.0, alpha=0.9
                    )

                ax.set_yscale("log")
                ax.set_ylim(bottom=0.5)
                ax.set_ylabel(
                    row_label,
                    fontsize=7,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=50,
                )
                ax.tick_params(axis="both", labelsize=7)
                if ax is not axes[-1]:
                    ax.tick_params(axis="x", labelbottom=False)

            axes[-1].set_xlabel(f"{label} (ns)", fontsize=9)

            legend_handles = [
                plt.matplotlib.patches.Patch(
                    color="steelblue", alpha=0.7, label="All valid"
                ),
                plt.matplotlib.lines.Line2D(
                    [],
                    [],
                    color="darkorange",
                    linewidth=1.5,
                    label="Double-coincidence only",
                ),
            ]
            fig.legend(handles=legend_handles, loc="upper right", fontsize=8)
            fig.suptitle(
                f"Hi L1B TOF: {label} — {self.instrument}", fontsize=11, y=1.01
            )
            plt.show()

        # ------------------------------------------------------------------ #
        # Figure set 2 — 2D tB-tA vs tC1-tA (per ESA + total)               #
        # ------------------------------------------------------------------ #
        mask_abc = m_ab & m_ac1
        edges_2ns = np.arange(-512, 514, 2)

        fig2, axes2 = plt.subplots(5, 2, figsize=(13, 20), constrained_layout=True)
        axes2_flat = axes2.ravel()

        esa_labels_10 = [f"ESA {e}" for e in range(1, 10)] + ["All ESA"]
        esa_masks_10 = [event_esa == e for e in range(1, 10)] + [
            np.ones(len(event_esa), dtype=bool)
        ]

        # Compute global vmax for a consistent colorscale across panels
        h_total_2d, _, _ = np.histogram2d(
            tof_ab_v[mask_abc],
            tof_ac1_v[mask_abc],
            bins=(edges_2ns, edges_2ns),
        )
        global_vmax_abc = max(h_total_2d.max(), 1.0)

        for ax, row_label, row_mask in zip(axes2_flat, esa_labels_10, esa_masks_10):
            combined = mask_abc & row_mask
            ax.set_facecolor("black")
            ax.set_title(row_label, fontsize=9)
            ax.set_xlabel("tB - tA (ns)", fontsize=7)
            ax.set_ylabel("tC1 - tA (ns)", fontsize=7)
            ax.tick_params(labelsize=7)

            if combined.sum() == 0:
                continue

            h, xedges, yedges = np.histogram2d(
                tof_ab_v[combined],
                tof_ac1_v[combined],
                bins=(edges_2ns, edges_2ns),
            )
            h_plot = h.T.astype(float)
            h_plot[h_plot == 0] = np.nan
            norm = mcolors.LogNorm(vmin=0.5, vmax=global_vmax_abc)
            im2 = ax.pcolormesh(
                xedges, yedges, h_plot, cmap=log_cmap, norm=norm, shading="flat"
            )

        fig2.colorbar(im2, ax=axes2_flat[-1], label="Counts (log scale)", shrink=0.8)
        fig2.suptitle(
            f"Hi L1B 2D TOF: tB-tA vs tC1-tA (2 ns bins) — {self.instrument}",
            fontsize=11,
        )
        plt.show()

        # ------------------------------------------------------------------ #
        # Figure set 3 — 2D tB-tA vs tC2-tC1 (all ESA summed)               #
        # ------------------------------------------------------------------ #
        mask_abc2 = m_ab & m_c1c2
        edges_c1c2_1ns = np.arange(0, 513, 1)

        fig3, ax3 = plt.subplots(figsize=(11, 7))
        ax3.set_facecolor("black")

        if mask_abc2.sum() > 0:
            h3, xedges3, yedges3 = np.histogram2d(
                tof_ab_v[mask_abc2],
                tof_c1c2_v[mask_abc2],
                bins=(edges_2ns, edges_c1c2_1ns),
            )
            h3_plot = h3.T.astype(float)
            h3_plot[h3_plot == 0] = np.nan
            norm3 = mcolors.LogNorm(vmin=0.5, vmax=max(h3.max(), 1.0))
            im3 = ax3.pcolormesh(
                xedges3, yedges3, h3_plot, cmap=log_cmap, norm=norm3, shading="flat"
            )
            fig3.colorbar(im3, ax=ax3, label="Counts (log scale)")

        ax3.set_xlabel("tB - tA (ns)", fontsize=10)
        ax3.set_ylabel("tC2 - tC1 (ns)", fontsize=10)
        ax3.set_title(
            f"Hi L1B 2D TOF: tB-tA vs tC2-tC1 (all ESA) — {self.instrument}",
            fontsize=11,
        )
        plt.tight_layout()
        plt.show()


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
        """
        Generate Lo star sensor quicklook plot.

        Produces a two-panel figure:

        - **Top:** 2D pcolormesh with spin number on X, spin phase (°) on Y,
          and star sensor voltage (V) as color.  Black background; black →
          rainbow colormap (0 – 2.5 V).  Unscanned phase bins are masked black.
        - **Bottom:** star sensor voltage vs. spin phase averaged over all
          spins (blue line).  The simulated star sensor curve (green) requires
          SPICE attitude data not present in the L1A file and is omitted here.

        Data source: ``imap_lo_l1a_star`` CDF.
        ``data`` shape (n_spins, 720): 720 samples per spin at 0.5 °/sample.
        ``count`` gives the number of valid samples per spin packet.
        DN → Volts conversion assumes a 12-bit ADC with 2.5 V full-scale
        (scale factor 2.5 / 4096).
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        ds = self.data_set
        raw = ds["data"].values.astype(float)  # shape (n_spins, 720)
        counts = ds["count"].values  # valid samples per spin packet

        # Convert DN → Volts (12-bit ADC, 2.5 V full scale)
        DN_TO_VOLTS = 2.5 / 4096
        voltage = raw * DN_TO_VOLTS

        # Mask unscanned phase bins in partial-spin packets
        for i in range(len(counts)):
            if counts[i] < voltage.shape[1]:
                voltage[i, counts[i] :] = np.nan

        n_spins, n_phase = voltage.shape
        phase_edges = np.arange(n_phase + 1) * 0.5  # 0, 0.5, …, 360.0
        phase_centers = np.arange(n_phase) * 0.5
        spin_edges = np.arange(n_spins + 1)

        # Phase-averaged voltage for the bottom plot
        phase_means = np.nanmean(voltage, axis=0)  # shape (720,)

        # Black → rainbow colormap (0 V = black, 2.5 V = red)
        rc = plt.cm.rainbow(np.linspace(0, 1, 255))
        star_cmap = mcolors.LinearSegmentedColormap.from_list(
            "black_rainbow", np.vstack([[[0.0, 0.0, 0.0, 1.0]], rc])
        )
        star_cmap.set_bad("black")  # NaN (unscanned) → black
        vmax = float(np.nanmax(voltage)) or 2.5

        # --- Figure ---
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(14, 10), constrained_layout=True
        )
        fig.patch.set_facecolor("black")

        # Top panel — 2D pcolormesh
        ax_top.set_facecolor("black")
        im = ax_top.pcolormesh(
            spin_edges,
            phase_edges,
            voltage.T,
            cmap=star_cmap,
            vmin=0,
            vmax=vmax,
            shading="flat",
        )
        cbar = fig.colorbar(im, ax=ax_top, pad=0.01)
        cbar.set_label("Volts per 0.5° Bin", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax_top.set_xlabel("Spin Number", color="white")
        ax_top.set_ylabel("Spin Phase (°)", color="white")
        ax_top.set_ylim(0, 360)
        ax_top.set_yticks(range(0, 361, 45))
        ax_top.tick_params(colors="white")
        ax_top.set_title(
            f"Lo L1A Star Sensor — {self.instrument}", color="white", fontsize=11
        )
        for spine in ax_top.spines.values():
            spine.set_edgecolor("white")

        # Bottom panel — voltage vs spin phase
        ax_bot.plot(
            phase_centers,
            phase_means,
            color="royalblue",
            linewidth=1.0,
            label="IMAP-Lo Star Sensor",
        )
        # Simulated curve placeholder — requires SPICE attitude data:
        # ax_bot.plot(phase_centers, simulated_voltage, color="green",
        #             linewidth=1.0, label="Simulated (attitude)")
        ax_bot.set_xlabel("Spinphase (°)")
        ax_bot.set_ylabel("Star Sensor (V)")
        ax_bot.set_xlim(0, 360)
        ax_bot.set_xticks(range(0, 361, 25))
        ax_bot.legend(fontsize=9)
        ax_bot.set_ylim(bottom=0)

        plt.show()

    def histogram(self) -> None:
        """
        Generate Lo L1A histogram quicklook plot.

        Produces three stacked 2D pcolormesh panels (black background):

        - **Total TOF coincidences** — sum of tof0_tof1, tof0_tof2, tof1_tof2,
          and silver across all ESA steps.
        - **Hydrogen counts** — summed over all ESA steps.
        - **Oxygen counts** — summed over all ESA steps.

        X axis: time (UTC).  Y axis: spin phase 0°–360° (6° bins, 60 total).
        Colormap: black → rainbow, linear scale, per-panel autoscale.
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        ds = self.data_set
        fill = 4294967295  # FILLVAL for all count variables

        epoch_dt = convert_j2000_to_utc(ds["epoch"].values)

        # Spin-phase axis: azimuth_6 has 60 bins at 6° each (uint8 overflows,
        # so derive degrees directly rather than reading the coordinate values)
        az_deg_edges = np.arange(61) * 6.0  # 0, 6, 12, …, 360

        def _sum_over_esa(var_names: list[str]) -> np.ndarray:
            """
            Return counts summed over named variables and ESA steps.

            Parameters
            ----------
            var_names : list[str]
                Dataset variable names to accumulate.

            Returns
            -------
            np.ndarray
                Array of shape (n_epoch, 60) with ESA-step-summed counts.
            """
            total = np.zeros((len(epoch_dt), 60), dtype=np.float64)
            for name in var_names:
                arr = ds[name].values.astype(np.float64)
                arr[arr == fill] = 0.0
                total += arr.sum(axis=1)  # sum over esa_step dim
            return total

        panels = [
            (
                _sum_over_esa(["tof0_tof1", "tof0_tof2", "tof1_tof2", "silver"]),
                "Total TOF Coincidences\n(tof0_tof1 + tof0_tof2 + tof1_tof2 + silver)",
                "Counts",
            ),
            (
                _sum_over_esa(["hydrogen"]),
                "Hydrogen",
                "Counts",
            ),
            (
                _sum_over_esa(["oxygen"]),
                "Oxygen",
                "Counts",
            ),
        ]

        # Black → rainbow colormap
        rc = plt.cm.rainbow(np.linspace(0, 1, 255))
        hist_cmap = mcolors.LinearSegmentedColormap.from_list(
            "black_rainbow", np.vstack([[[0.0, 0.0, 0.0, 1.0]], rc])
        )
        hist_cmap.set_bad("black")

        # Time edges for pcolormesh (add trailing edge from last interval)
        epoch_ns = epoch_dt.astype("datetime64[ns]")
        if len(epoch_ns) > 1:
            dt = (epoch_ns[-1] - epoch_ns[-2]).astype("timedelta64[ns]")
        else:
            dt = np.timedelta64(int(4.32e11), "ns")  # ~7.2 min default
        time_edges = np.append(epoch_ns, epoch_ns[-1] + dt)

        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(14, 4 * len(panels)),
            sharex=True,
            constrained_layout=True,
        )
        fig.patch.set_facecolor("black")

        for ax, (counts, label, cbar_label) in zip(axes, panels):
            ax.set_facecolor("black")
            vmax = float(counts.max()) or 1.0

            im = ax.pcolormesh(
                time_edges,
                az_deg_edges,
                counts.T,
                cmap=hist_cmap,
                vmin=0,
                vmax=vmax,
                shading="flat",
            )
            cbar = fig.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label(cbar_label, color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
            cbar.ax.yaxis.set_tick_params(color="white")

            ax.set_ylabel(label, color="white", fontsize=9)
            ax.set_ylim(0, 360)
            ax.set_yticks(range(0, 361, 60))
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("white")

        axes[-1].set_xlabel("Time (UTC)", color="white")
        fig.suptitle(
            f"Lo L1A Histogram — {self.instrument} — summed over ESA steps",
            color="white",
            fontsize=11,
        )
        plt.show()

    def de_histogram(self) -> None:
        """
        Generate Lo L1B DE histogram quicklook plot.

        Direct events are binned into 6° spin-phase bins (60 total) using
        ``spin_bin`` (0–3599, 0.1°/bin → divide by 60).  The X axis is time,
        derived from the first event epoch in each unique ``spin_cycle``.

        Produces three stacked 2D pcolormesh panels (black background):

        - **Hydrogen (H)**
        - **Oxygen (O)**
        - **Unidentified (U)**

        Colormap: black → rainbow, linear per-panel autoscale.
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        ds = self.data_set

        spin_bin = ds["spin_bin"].values  # 0–3599
        spin_cycle = ds["spin_cycle"].values  # spin number (time proxy)
        species = ds["species"].values  # 'H', 'O', 'U'
        epoch_ns = ds["epoch"].values  # int64 J2000 ns

        # Convert 0.1° spin_bin to 6° azimuth bins (0–59)
        az_bin = spin_bin // 60

        # Map spin_cycle → sequential index and representative UTC time
        sc_unique, sc_first_idx = np.unique(spin_cycle, return_index=True)
        n_spins = len(sc_unique)
        sc_to_idx = np.empty(sc_unique[-1] - sc_unique[0] + 1, dtype=np.intp)
        sc_to_idx[sc_unique - sc_unique[0]] = np.arange(n_spins)
        spin_idx = sc_to_idx[spin_cycle - sc_unique[0]]

        spin_times = convert_j2000_to_utc(epoch_ns[sc_first_idx])

        az_deg_edges = np.arange(61) * 6.0  # 0, 6, …, 360
        # Time edges for pcolormesh
        spin_times_ns = spin_times.astype("datetime64[ns]")
        if n_spins > 1:
            dt = (spin_times_ns[-1] - spin_times_ns[-2]).astype("timedelta64[ns]")
        else:
            dt = np.timedelta64(int(15e9), "ns")  # 15 s default
        time_edges = np.append(spin_times_ns, spin_times_ns[-1] + dt)

        def _build_grid(mask: np.ndarray) -> np.ndarray:
            """
            Return 2D histogram (n_spins × 60) for a boolean event mask.

            Parameters
            ----------
            mask : np.ndarray
                Boolean array selecting events to bin.

            Returns
            -------
            np.ndarray
                Array of shape (n_spins, 60) with event counts per spin/azimuth bin.
            """
            grid = np.zeros((n_spins, 60), dtype=np.float64)
            np.add.at(grid, (spin_idx[mask], az_bin[mask]), 1)
            return grid

        panels = [
            (_build_grid(species == "H"), "Hydrogen (H)"),
            (_build_grid(species == "O"), "Oxygen (O)"),
            (_build_grid(species == "U"), "Unidentified (U)"),
        ]

        # Black → rainbow colormap
        rc = plt.cm.rainbow(np.linspace(0, 1, 255))
        de_cmap = mcolors.LinearSegmentedColormap.from_list(
            "black_rainbow", np.vstack([[[0.0, 0.0, 0.0, 1.0]], rc])
        )
        de_cmap.set_bad("black")

        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(14, 4 * len(panels)),
            sharex=True,
            constrained_layout=True,
        )
        fig.patch.set_facecolor("black")

        for ax, (grid, label) in zip(axes, panels):
            ax.set_facecolor("black")
            vmax = float(grid.max()) or 1.0

            im = ax.pcolormesh(
                time_edges,
                az_deg_edges,
                grid.T,
                cmap=de_cmap,
                vmin=0,
                vmax=vmax,
                shading="flat",
            )
            cbar = fig.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label("Counts", color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
            cbar.ax.yaxis.set_tick_params(color="white")

            ax.set_ylabel(label, color="white", fontsize=9)
            ax.set_ylim(0, 360)
            ax.set_yticks(range(0, 361, 60))
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("white")

        axes[-1].set_xlabel("Time (UTC)", color="white")
        fig.suptitle(
            f"Lo L1B DE Histogram — {self.instrument} — binned by spin phase",
            color="white",
            fontsize=11,
        )
        plt.show()

    def de_tof(self) -> None:
        """
        Generate Lo L1B DE TOF diagnostic quicklook plots.

        Produces two figures:

        **Figure 1 — 2D TOF histograms (3 rows × 2 columns):**

        Left column — long-range TOF pairs (2 ns bins on both axes):
          TOF0 vs TOF1 · TOF0 vs TOF2 · TOF1 vs TOF2

        Right column — short TOF3 (0.5 ns bins) vs each long TOF (2 ns bins):
          TOF3 vs TOF0 · TOF3 vs TOF1 · TOF3 vs TOF2

        Colormap: log rainbow+white; black for zero-count bins.

        **Figure 2 — 1D TOF histograms (4 stacked panels, log Y scale):**
          TOF0 · TOF1 · TOF2 · TOF3, all with 1 ns bins.
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        ds = self.data_set
        fill = float(ds["tof0"].attrs.get("FILLVAL", -1e31))

        tofs = {
            k: ds[k].values.astype(np.float64) for k in ["tof0", "tof1", "tof2", "tof3"]
        }
        valid = {k: (v > fill) & (v >= 0) for k, v in tofs.items()}

        # Log rainbow+white colormap; black for empty 2D bins (NaN)
        rc = plt.cm.rainbow(np.linspace(0, 1, 255))
        log_cmap = mcolors.LinearSegmentedColormap.from_list(
            "log_rainbow_white", np.vstack([[[1, 1, 1, 1]], rc])
        )
        log_cmap.set_bad("black")

        long_edges = np.arange(0, 352, 2)  # 2 ns bins, 0–350 ns
        short_edges = np.arange(0, 20.5, 0.5)  # 0.5 ns bins, 0–20 ns

        # ------------------------------------------------------------------ #
        # Figure 1 — 2D histograms                                            #
        # ------------------------------------------------------------------ #
        layout = [
            # (x_name, y_name, x_edges, y_edges)
            ("tof0", "tof1", long_edges, long_edges),
            ("tof0", "tof2", long_edges, long_edges),
            ("tof1", "tof2", long_edges, long_edges),
            ("tof3", "tof0", short_edges, long_edges),
            ("tof3", "tof1", short_edges, long_edges),
            ("tof3", "tof2", short_edges, long_edges),
        ]

        fig1, axes1 = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)
        # Reorder: left column = long pairs, right column = TOF3 pairs
        ax_order = [
            axes1[0, 0],
            axes1[1, 0],
            axes1[2, 0],
            axes1[0, 1],
            axes1[1, 1],
            axes1[2, 1],
        ]

        for ax, (xk, yk, xe, ye) in zip(ax_order, layout):
            mask = valid[xk] & valid[yk]
            ax.set_facecolor("black")
            ax.set_xlabel(f"TOF {xk[-1]} (ns)", fontsize=9)
            ax.set_ylabel(f"TOF {yk[-1]} (ns)", fontsize=9)
            ax.tick_params(labelsize=8)

            if mask.sum() == 0:
                continue

            h, xe_out, ye_out = np.histogram2d(
                tofs[xk][mask], tofs[yk][mask], bins=(xe, ye)
            )
            h_plot = h.T.astype(float)
            h_plot[h_plot == 0] = np.nan
            vmax = np.nanmax(h_plot) if np.any(~np.isnan(h_plot)) else 1.0
            norm = mcolors.LogNorm(vmin=0.5, vmax=vmax)
            im = ax.pcolormesh(
                xe_out,
                ye_out,
                h_plot,
                cmap=log_cmap,
                norm=norm,
                shading="flat",
            )
            fig1.colorbar(im, ax=ax, label="Counts", pad=0.02)

        fig1.suptitle(f"Lo L1B 2D TOF Histograms — {self.instrument}", fontsize=12)
        plt.show()

        # ------------------------------------------------------------------ #
        # Figure 2 — 1D histograms                                            #
        # ------------------------------------------------------------------ #
        tof_1d = [
            ("tof0", np.arange(0, 352, 1), "TOF 0"),
            ("tof1", np.arange(0, 322, 1), "TOF 1"),
            ("tof2", np.arange(0, 175, 1), "TOF 2"),
            ("tof3", np.arange(0, 20.5, 1), "TOF 3"),
        ]

        fig2, axes2 = plt.subplots(
            len(tof_1d),
            1,
            figsize=(10, 12),
            sharex=False,
            constrained_layout=True,
        )

        for ax, (key, edges, label) in zip(axes2, tof_1d):
            vals = tofs[key][valid[key]]
            counts, _ = np.histogram(vals, bins=edges)
            ax.stairs(counts, edges, fill=True, color="black", linewidth=0.5)
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)
            ax.set_ylabel(f"{label} counts", fontsize=9)
            ax.set_xlabel("TOF (ns)", fontsize=9)
            ax.tick_params(labelsize=8)

        fig2.suptitle(f"Lo L1B 1D TOF Histograms — {self.instrument}", fontsize=12)
        plt.show()


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
        """
        Generate a 10-panel GLOWS general quicklook figure.

        Left column (time series):
          - L1b histogram pcolormesh (epoch × spin angle)
          - Total photon counts per block
          - Quality flags heatmap
          - SWE electron count rate (stub if not loaded)
          - SWAPI proton moments (stub)

        Right column (spin-angle profiles + sky map):
          - F10.7 solar flux (stub)
          - Lyman-alpha flux (stub)
          - L1b histogram summed over epoch vs spin angle
          - L2 photon flux vs spin angle (with uncertainty band)
          - Sky map in ecliptic coordinates
        """
        if self.data_set is None:
            raise ValueError("Must load a dataset.")

        l1b = self.data_set
        l2: xr.Dataset | None = getattr(self, "data_set_l2", None)
        swe: xr.Dataset | None = getattr(self, "data_set_swe", None)

        epoch_dt = convert_j2000_to_utc(l1b["epoch"].values)
        spin_angle_bins = np.arange(3600) * 0.1  # 0.0 to 359.9 degrees

        fig = plt.figure(figsize=(20, 20))
        gs = GridSpec(
            5,
            2,
            figure=fig,
            width_ratios=[1.1, 0.9],
            height_ratios=[1, 1, 1, 1, 1.7],
            hspace=0.55,
            wspace=0.38,
        )

        # ── Left column: time-series panels ──────────────────────

        # Panel 1: L1b histogram pcolormesh (epoch × spin angle → counts)
        ax_hist = fig.add_subplot(gs[0, 0])
        hist_data = l1b["histogram"].values.astype(float)
        hist_data[hist_data == 0] = np.nan
        im_hist = ax_hist.pcolormesh(
            epoch_dt,
            spin_angle_bins,
            hist_data.T,
            norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(hist_data)),
            cmap="rainbow",
            shading="auto",
        )
        plt.colorbar(im_hist, ax=ax_hist, label="Counts")
        ax_hist.set_ylabel("Spin Angle [°]")
        ax_hist.set_ylim(0, 360)
        ax_hist.set_title("L1B Histogram")

        # Panel 2: Total photon counts per block
        ax_counts = fig.add_subplot(gs[1, 0])
        ax_counts.plot(epoch_dt, l1b["number_of_events"].values, color="steelblue")
        ax_counts.set_ylabel("Events per Block")
        ax_counts.set_xlim(epoch_dt[0], epoch_dt[-1])
        ax_counts.set_title("Total Photon Counts")

        # Panel 3: Quality flags heatmap (epoch × flag index)
        ax_flags = fig.add_subplot(gs[2, 0])
        flags = l1b["flags"].values  # (n_epoch, 17)
        im_flags = ax_flags.pcolormesh(
            epoch_dt,
            np.arange(flags.shape[1]),
            flags.T,
            cmap="Reds",
            vmin=0,
            vmax=1,
            shading="auto",
        )
        plt.colorbar(im_flags, ax=ax_flags, label="Flag set")
        ax_flags.set_ylabel("Flag Index")
        ax_flags.set_ylim(-0.5, flags.shape[1] - 0.5)
        ax_flags.set_title("Quality Flags")

        # Panel 4: SWE electron count rate
        ax_swe = fig.add_subplot(gs[3, 0])
        if swe is not None:
            swe_epoch_dt = convert_j2000_to_utc(swe["epoch"].values)
            # science_data: (epoch, esa_step, spin_sector, cem_id)
            # Sum over spin sectors and CEMs at ESA step 18 (~1 keV)
            sci = swe["science_data"].values.astype(float)
            fill = swe["science_data"].attrs.get("FILLVAL", None)
            if fill is not None:
                sci[sci == fill] = np.nan
            rate_1kev = np.nansum(sci[:, 18, :, :], axis=(1, 2))
            ax_swe.plot(swe_epoch_dt, rate_1kev, color="darkorange")
            ax_swe.set_ylabel("Counts")
            ax_swe.set_title("SWE Electron Count Rate (~1 keV)")
        else:
            ax_swe.text(
                0.5,
                0.5,
                "SWE data not available\nfor this time period",
                ha="center",
                va="center",
                transform=ax_swe.transAxes,
                fontsize=11,
                color="gray",
            )
            ax_swe.set_title("SWE Electron Count Rate (~1 keV)")
            ax_swe.set_xticks([])
            ax_swe.set_yticks([])

        # Panel 5: SWAPI proton moments (stub)
        ax_swapi = fig.add_subplot(gs[4, 0])
        ax_swapi.text(
            0.5,
            0.5,
            "SWAPI proton moments\nnot yet available",
            ha="center",
            va="center",
            transform=ax_swapi.transAxes,
            fontsize=11,
            color="gray",
        )
        ax_swapi.set_title("SWAPI Solar Wind (n_p, v_i)")
        ax_swapi.set_xticks([])
        ax_swapi.set_yticks([])

        # ── Right column: spin-angle profiles + sky map ───────────

        # Panel 6: F10.7 solar flux (stub)
        ax_f107 = fig.add_subplot(gs[0, 1])
        ax_f107.text(
            0.5,
            0.5,
            "F10.7 solar flux\nnot yet available",
            ha="center",
            va="center",
            transform=ax_f107.transAxes,
            fontsize=11,
            color="gray",
        )
        ax_f107.set_title("F10.7 Solar Flux")
        ax_f107.set_xticks([])
        ax_f107.set_yticks([])

        # Panel 7: Lyman-alpha flux (stub)
        ax_lya = fig.add_subplot(gs[1, 1])
        ax_lya.text(
            0.5,
            0.5,
            "Lyman-alpha flux\nnot yet available",
            ha="center",
            va="center",
            transform=ax_lya.transAxes,
            fontsize=11,
            color="gray",
        )
        ax_lya.set_title("Lyman-Alpha Flux")
        ax_lya.set_xticks([])
        ax_lya.set_yticks([])

        # Panel 8: L1b histogram summed over epoch vs spin angle
        ax_l1b_sum = fig.add_subplot(gs[2, 1])
        hist_sum = l1b["histogram"].values.sum(axis=0)  # (3600,)
        ax_l1b_sum.plot(spin_angle_bins, hist_sum, color="steelblue", linewidth=0.8)
        ax_l1b_sum.set_xlabel("Spin Angle [°]")
        ax_l1b_sum.set_ylabel("Total Counts")
        ax_l1b_sum.set_xlim(0, 360)
        ax_l1b_sum.set_title("L1B Histogram vs Spin Angle")

        # Panel 9: L2 photon flux vs spin angle (with uncertainty band)
        ax_flux = fig.add_subplot(gs[3, 1])
        if l2 is not None:
            spin_angle_l2 = l2["spin_angle"].values[0]  # (3600,)
            photon_flux = l2["photon_flux"].values[0]  # (3600,)
            # flux_uncertainties is stored as a coordinate in this file
            flux_unc = (
                l2["flux_uncertainties"].values[0]
                if "flux_uncertainties" in l2.coords
                else None
            )
            sort_idx = np.argsort(spin_angle_l2)
            spin_sorted = spin_angle_l2[sort_idx]
            flux_sorted = photon_flux[sort_idx]
            ax_flux.plot(spin_sorted, flux_sorted, color="steelblue", linewidth=0.8)
            if flux_unc is not None:
                unc_sorted = flux_unc[sort_idx]
                ax_flux.fill_between(
                    spin_sorted,
                    flux_sorted - unc_sorted,
                    flux_sorted + unc_sorted,
                    alpha=0.3,
                    color="steelblue",
                )
            ax_flux.set_xlabel("Spin Angle [°]")
            ax_flux.set_ylabel("Photon Flux [R]")
            ax_flux.set_xlim(0, 360)
            ax_flux.set_title("L2 Photon Flux vs Spin Angle")
        else:
            ax_flux.text(
                0.5,
                0.5,
                "L2 data not available",
                ha="center",
                va="center",
                transform=ax_flux.transAxes,
                fontsize=11,
                color="gray",
            )
            ax_flux.set_title("L2 Photon Flux vs Spin Angle")

        # Panel 10: Sky map in ecliptic coordinates
        ax_sky = fig.add_subplot(gs[4, 1])
        if l2 is not None:
            elon = l2["ecliptic_lon"].values[0]  # (3600,)
            elat = l2["ecliptic_lat"].values[0]  # (3600,)
            flux_sky = l2["photon_flux"].values[0]  # (3600,)
            sc = ax_sky.scatter(
                elon,
                elat,
                c=flux_sky,
                cmap="rainbow",
                s=3,
                linewidths=0,
            )
            plt.colorbar(
                sc,
                ax=ax_sky,
                label="Photon Flux [R]",
                location="bottom",
                pad=0.12,
                fraction=0.04,
            )
            # Mark galactic center (~ecliptic lon=266.4°, lat=-5.6°)
            ax_sky.plot(
                266.4, -5.6, "k*", markersize=10, label="Galactic Center", zorder=5
            )
            # Mark ecliptic-coordinate Sun (~lon=0°, lat=0°)
            ax_sky.plot(
                0, 0, "yo", markersize=8, markeredgecolor="k", label="Sun", zorder=5
            )
            ax_sky.set_xlabel("Ecliptic Longitude [°]")
            ax_sky.set_ylabel("Ecliptic Latitude [°]")
            ax_sky.set_xlim(0, 360)
            ax_sky.legend(loc="upper left", fontsize=8, markerscale=0.9)
            ax_sky.set_title("Sky Map (Ecliptic Coordinates)")
        else:
            ax_sky.text(
                0.5,
                0.5,
                "L2 data not available",
                ha="center",
                va="center",
                transform=ax_sky.transAxes,
                fontsize=11,
                color="gray",
            )
            ax_sky.set_title("Sky Map (Ecliptic Coordinates)")

        date_str = str(epoch_dt[0])[:10]
        fig.suptitle(f"GLOWS Quicklook — {date_str}", fontsize=14, fontweight="bold")
        plt.show()

    def ancillary_data(self) -> None:
        """
        Generate the GLOWS L1b ancillary data quicklook.

        Layout: 5 rows × 2 columns, each panel has dual y-axes.
        Left axis (red solid line) shows the averaged quantity;
        right axis (blue dotted markers) shows the corresponding std dev.

        Left column (top → bottom):
          - Total counts per L1b histogram | Spins per block
          - Filter temperature avg | std dev
          - HV voltage avg | std dev
          - Onboard spin period avg | std dev
          - Ground spin period avg | std dev

        Right column (top → bottom):
          - Spin (position-angle) offset avg | std dev
          - Pulse length avg | std dev
          - Spacecraft ecliptic longitude | std dev
          - Spacecraft ecliptic latitude | std dev
          - Number of bins per histogram | Spins per block
        """
        if self.data_set is None:
            raise ValueError("Must load a dataset.")

        ds = self.data_set
        epoch_dt = convert_j2000_to_utc(ds["epoch"].values)

        # Derive spacecraft ecliptic lon/lat from Cartesian location
        loc = ds["spacecraft_location_average"].values  # (N, 3) km
        loc_std = ds["spacecraft_location_std_dev"].values  # (N, 3) km
        x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        r_xy = np.sqrt(x**2 + y**2)
        sc_lon = np.degrees(np.arctan2(y, x)) % 360
        sc_lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
        # Angular uncertainties from Cartesian std devs (small-angle approximation)
        sc_lon_std = np.degrees(loc_std[:, 1] / np.where(r_xy > 0, r_xy, 1))
        sc_lat_std = np.degrees(loc_std[:, 2] / np.where(r > 0, r, 1))

        # Panel definitions: (left_data, right_data, left_label, right_label)
        left_panels = [
            (
                ds["number_of_events"].values,
                ds["number_of_spins_per_block"].values,
                "Total counts per\nL1b histogram [cts]",
                "Spins per block",
            ),
            (
                ds["filter_temperature_average"].values,
                ds["filter_temperature_std_dev"].values,
                "T avg [°C]",
                "T std dev [°C]",
            ),
            (
                ds["hv_voltage_average"].values,
                ds["hv_voltage_std_dev"].values,
                "HV avg [V]",
                "HV std dev [V]",
            ),
            (
                ds["spin_period_average"].values,
                ds["spin_period_std_dev"].values,
                "P onboard avg [s]",
                "P onboard std dev [s]",
            ),
            (
                ds["spin_period_ground_average"].values,
                ds["spin_period_ground_std_dev"].values,
                "P ground avg [s]",
                "P ground std dev [s]",
            ),
        ]
        right_panels = [
            (
                ds["position_angle_offset_average"].values,
                ds["position_angle_offset_std_dev"].values,
                "Spin offset avg [deg]",
                "Spin offset std dev [deg]",
            ),
            (
                ds["pulse_length_average"].values,
                ds["pulse_length_std_dev"].values,
                "Pulse avg [μs]",
                "Pulse std dev [μs]",
            ),
            (
                sc_lon,
                sc_lon_std,
                "Lon avg [deg]",
                "Lon std dev [deg]",
            ),
            (
                sc_lat,
                sc_lat_std,
                "Lat avg [deg]",
                "Lat std dev [deg]",
            ),
            (
                ds["number_of_bins_per_histogram"].values,
                ds["number_of_spins_per_block"].values,
                "Num of bins",
                "Spins per block",
            ),
        ]

        fig, axes = plt.subplots(
            5,
            2,
            figsize=(18, 16),
            sharex=True,
            constrained_layout=False,
        )
        fig.subplots_adjust(hspace=0.30, wspace=0.55)

        for row, (lp, rp) in enumerate(zip(left_panels, right_panels)):
            is_bottom = row == 4
            for panel, ax in zip([lp, rp], axes[row]):
                left_data, right_data, left_label, right_label = panel

                ax.plot(epoch_dt, left_data, color="red", linewidth=1.0)
                ax.set_ylabel(left_label, color="red", fontsize=8)
                ax.tick_params(axis="y", labelcolor="red", labelsize=7)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(epoch_dt[0], epoch_dt[-1])

                ax2 = ax.twinx()
                ax2.plot(epoch_dt, right_data, ".", color="blue", markersize=3)
                ax2.set_ylabel(right_label, color="blue", fontsize=8)
                ax2.tick_params(axis="y", labelcolor="blue", labelsize=7)

                if is_bottom:
                    ax.set_xlabel("UTC time", fontsize=9)

        start_utc = str(epoch_dt[0])[:19].replace("T", " ")
        fig.suptitle(
            f"IMAP/GLOWS QUICKLOOK PLOT: ANCILLARY DATA\n"
            f"L1b start time: {start_utc} UTC",
            fontsize=12,
            fontweight="bold",
        )
        plt.show()


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
        """
        Generate HIT ion flux spectrograms for H, He, O, and Fe.

        Four vertically stacked panels (no gap), one per species, each showing:
        - x: UTC time
        - y: Energy (MeV/nuc), log scale
        - color: Flux (cm⁻² sr⁻¹ s⁻¹ (MeV/nuc)⁻¹), log scale
        - Black background; fill/zero values shown as black.
        """
        if self.data_set is None:
            raise ValueError("Must load a dataset.")

        ds = self.data_set
        epoch_dt = convert_j2000_to_utc(ds["epoch"].values)

        # Build time edges: midpoints between consecutive epochs,
        # with ±30 s padding at the ends.
        ep_ns = epoch_dt.astype("datetime64[ns]").astype(np.int64)
        thirty_s_ns = int(30e9)
        mid_ns = (ep_ns[:-1] + ep_ns[1:]) // 2
        edge_ns = np.concatenate(
            [[ep_ns[0] - thirty_s_ns], mid_ns, [ep_ns[-1] + thirty_s_ns]]
        )
        time_edges = edge_ns.astype("datetime64[ns]")

        # Species: (flux_var, energy_mean_coord, dm_coord, dp_coord, label, energy_unit)
        species = [
            (
                "h_standard_intensity",
                "h_energy_mean",
                "h_energy_delta_minus",
                "h_energy_delta_plus",
                "H",
                "MeV",
            ),
            (
                "he_standard_intensity",
                "he_energy_mean",
                "he_energy_delta_minus",
                "he_energy_delta_plus",
                "He",
                "MeV/nuc",
            ),
            (
                "o_standard_intensity",
                "o_energy_mean",
                "o_energy_delta_minus",
                "o_energy_delta_plus",
                "O",
                "MeV/nuc",
            ),
            (
                "fe_standard_intensity",
                "fe_energy_mean",
                "fe_energy_delta_minus",
                "fe_energy_delta_plus",
                "Fe",
                "MeV/nuc",
            ),
        ]

        # Rainbow colormap with black for masked (fill / zero) values
        cmap = plt.get_cmap("rainbow").copy()
        cmap.set_bad("black")
        cmap.set_under("black")

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(14, 11),
            sharex=True,
            constrained_layout=False,
        )
        fig.patch.set_facecolor("black")
        fig.subplots_adjust(hspace=0, right=0.80, top=0.93, bottom=0.08)

        for ax, (flux_var, en_key, dm_key, dp_key, label, en_unit) in zip(
            axes, species
        ):
            flux = ds[flux_var].values.astype(float)  # (N_time, N_energy)
            energy_mean = ds[en_key].values
            dm = ds[dm_key].values
            dp = ds[dp_key].values

            # Energy bin edges: lower edge of each bin + upper edge of last bin
            energy_edges = np.append(energy_mean - dm, energy_mean[-1] + dp[-1])

            # Mask fill values (≤ 0 covers fill=-1e31 and genuine zero-count bins)
            flux[flux <= 0] = np.nan

            # Per-species colorbar range from valid data
            valid = flux[np.isfinite(flux)]
            vmin = float(valid.min()) if len(valid) else 1e-4
            vmax = float(valid.max()) if len(valid) else 1e4
            # Guard against vmin == vmax (all-zero or single-value file)
            if vmin >= vmax:
                vmin, vmax = 1e-4, 1e4

            ax.set_facecolor("black")
            im = ax.pcolormesh(
                time_edges,
                energy_edges,
                flux.T,
                norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=cmap,
                shading="flat",
            )
            ax.set_yscale("log")
            ax.set_ylabel(f"Energy\n({en_unit})", color="white", fontsize=9)
            ax.tick_params(colors="white", which="both")
            for spine in ax.spines.values():
                spine.set_color("white")

            # Species label — upper-right corner
            ax.text(
                0.99,
                0.97,
                label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="white",
                fontsize=11,
                fontweight="bold",
            )

            # Individual colorbar
            cbar_ax = fig.add_axes(
                [
                    0.82,  # left
                    ax.get_position().y0 + 0.005,  # bottom (with small padding)
                    0.015,  # width
                    ax.get_position().height - 0.01,  # height
                ]
            )
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(
                "Flux\n[cm⁻² sr⁻¹ s⁻¹\n(MeV/nuc)⁻¹]",
                color="white",
                fontsize=7,
            )
            cbar.ax.tick_params(colors="white", labelsize=7)
            cbar.ax.yaxis.set_tick_params(color="white")

        axes[-1].set_xlabel("UTC Time", color="white", fontsize=10)
        axes[-1].tick_params(axis="x", colors="white", labelsize=8)
        # Hide x-tick labels on non-bottom panels (sharex handles ticks,
        # but spines still need styling)
        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)

        date_str = str(epoch_dt[0])[:10]
        fig.suptitle(
            f"HIT Ion Flux Spectrograms — {date_str}",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        plt.show()

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
