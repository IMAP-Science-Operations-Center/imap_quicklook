"""ULTRA quicklook generator."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plotting.base_quicklook import QuicklookGenerator, convert_j2000_to_utc


@dataclass(init=False)
class UltraQuicklookGenerator(QuicklookGenerator):
    """ULTRA subclass for ULTRA quicklook plots."""

    data_set_aux: xr.Dataset | None = None

    def two_dimensional_plot(self, variable: str = "") -> None:
        """
        Lead to correct function that will generate the desired quicklook plot.

        Parameters
        ----------
        variable : str
            Variable to specify which quicklook plot to generate.
        """
        match variable:
            case "raw image events":  # Slide 2
                self.raw_image_events()
            case "priority 1 events":  # Slide 3
                self.priority_event_spectrogram(priority=1)
            case "priority 2 events":  # Slide 3
                self.priority_event_spectrogram(priority=2)
            case "tof spin spectrogram":  # Slide 4
                self.tof_spin_spectrogram()
            case "tof spectrum":  # Slide 4
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
