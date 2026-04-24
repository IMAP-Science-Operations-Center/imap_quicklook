"""Lo quicklook generator."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plotting.base_quicklook import QuicklookGenerator, convert_j2000_to_utc


class LoQuicklookGenerator(QuicklookGenerator):
    """Lo subclass for Lo quicklook plots."""

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
          and star sensor voltage (V) as color.  White for unscanned bins.
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
        counts = ds["count"].values

        DN_TO_VOLTS = 2.5 / 4096
        voltage = raw * DN_TO_VOLTS

        for i in range(len(counts)):
            if counts[i] < voltage.shape[1]:
                voltage[i, counts[i] :] = np.nan

        n_spins, n_phase = voltage.shape
        phase_edges = np.arange(n_phase + 1) * 0.5
        phase_centers = np.arange(n_phase) * 0.5
        spin_edges = np.arange(n_spins + 1)

        phase_means = np.nanmean(voltage, axis=0)

        cmap = plt.get_cmap("plasma").copy()
        cmap.set_bad("whitesmoke")
        vmax = float(np.nanmax(voltage)) or 2.5

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(14, 10), constrained_layout=True
        )

        im = ax_top.pcolormesh(
            spin_edges,
            phase_edges,
            voltage.T,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            shading="flat",
        )
        cbar = fig.colorbar(im, ax=ax_top, pad=0.01)
        cbar.set_label("Volts per 0.5° Bin", fontsize=10)

        ax_top.set_xlabel("Spin Number", fontsize=10)
        ax_top.set_ylabel("Spin Phase (°)", fontsize=10)
        ax_top.set_ylim(0, 360)
        ax_top.set_yticks(range(0, 361, 45))
        ax_top.set_title(
            f"Lo L1A Star Sensor — {self.instrument}", fontsize=12, fontweight="bold"
        )

        ax_bot.plot(
            phase_centers,
            phase_means,
            color="steelblue",
            linewidth=1.0,
            label="IMAP-Lo Star Sensor",
        )
        # Simulated curve placeholder — requires SPICE attitude data:
        # ax_bot.plot(phase_centers, simulated_voltage, color="seagreen",
        #             linewidth=1.0, label="Simulated (attitude)")
        ax_bot.set_xlabel("Spinphase (°)", fontsize=10)
        ax_bot.set_ylabel("Star Sensor (V)", fontsize=10)
        ax_bot.set_xlim(0, 360)
        ax_bot.set_xticks(range(0, 361, 25))
        ax_bot.legend(fontsize=9)
        ax_bot.set_ylim(bottom=0)

        plt.show()

    def histogram(self) -> None:
        """
        Generate Lo L1A histogram quicklook plot.

        Produces three stacked 2D pcolormesh panels:

        - **Total TOF coincidences** — sum of tof0_tof1, tof0_tof2, tof1_tof2,
          and silver across all ESA steps.
        - **Hydrogen counts** — summed over all ESA steps.
        - **Oxygen counts** — summed over all ESA steps.

        X axis: time (UTC).  Y axis: spin phase 0°–360° (6° bins, 60 total).
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        ds = self.data_set
        fill = 4294967295

        epoch_dt = convert_j2000_to_utc(ds["epoch"].values)
        az_deg_edges = np.arange(61) * 6.0

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
                total += arr.sum(axis=1)
            return total

        panels = [
            (
                _sum_over_esa(["tof0_tof1", "tof0_tof2", "tof1_tof2", "silver"]),
                "Total TOF Coincidences\n(tof0_tof1 + tof0_tof2 + tof1_tof2 + silver)",
                "Counts",
            ),
            (_sum_over_esa(["hydrogen"]), "Hydrogen", "Counts"),
            (_sum_over_esa(["oxygen"]), "Oxygen", "Counts"),
        ]

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("whitesmoke")

        epoch_ns = epoch_dt.astype("datetime64[ns]")
        if len(epoch_ns) > 1:
            dt = (epoch_ns[-1] - epoch_ns[-2]).astype("timedelta64[ns]")
        else:
            dt = np.timedelta64(int(4.32e11), "ns")
        time_edges = np.append(epoch_ns, epoch_ns[-1] + dt)

        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(14, 4 * len(panels)),
            sharex=True,
            constrained_layout=True,
        )

        for ax, (counts, label, cbar_label) in zip(axes, panels):
            counts_plot = counts.copy()
            counts_plot[counts_plot == 0] = np.nan
            vmax = (
                float(np.nanmax(counts_plot)) if np.any(~np.isnan(counts_plot)) else 1.0
            )

            im = ax.pcolormesh(
                time_edges,
                az_deg_edges,
                counts_plot.T,
                cmap=cmap,
                vmin=0.5,
                vmax=vmax,
                shading="flat",
            )
            cbar = fig.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label(cbar_label, fontsize=9)

            ax.set_ylabel(label, fontsize=9)
            ax.set_ylim(0, 360)
            ax.set_yticks(range(0, 361, 60))

        axes[-1].set_xlabel("Time (UTC)", fontsize=10)
        fig.suptitle(
            f"Lo L1A Histogram — {self.instrument} — summed over ESA steps",
            fontsize=12,
            fontweight="bold",
        )
        plt.show()

    def de_histogram(self) -> None:
        """
        Generate Lo L1B DE histogram quicklook plot.

        Direct events are binned into 6° spin-phase bins (60 total) using
        ``spin_bin`` (0–3599, 0.1°/bin → divide by 60).  The X axis is time,
        derived from the first event epoch in each unique ``spin_cycle``.

        Produces three stacked 2D pcolormesh panels:

        - **Hydrogen (H)**
        - **Oxygen (O)**
        - **Unidentified (U)**
        """
        if self.data_set is None:
            raise ValueError("Must load in a dataset.")

        ds = self.data_set

        spin_bin = ds["spin_bin"].values
        spin_cycle = ds["spin_cycle"].values
        species = ds["species"].values
        epoch_ns = ds["epoch"].values

        az_bin = spin_bin // 60

        sc_unique, sc_first_idx = np.unique(spin_cycle, return_index=True)
        n_spins = len(sc_unique)
        sc_to_idx = np.empty(sc_unique[-1] - sc_unique[0] + 1, dtype=np.intp)
        sc_to_idx[sc_unique - sc_unique[0]] = np.arange(n_spins)
        spin_idx = sc_to_idx[spin_cycle - sc_unique[0]]

        spin_times = convert_j2000_to_utc(epoch_ns[sc_first_idx])

        az_deg_edges = np.arange(61) * 6.0
        spin_times_ns = spin_times.astype("datetime64[ns]")
        if n_spins > 1:
            dt = (spin_times_ns[-1] - spin_times_ns[-2]).astype("timedelta64[ns]")
        else:
            dt = np.timedelta64(int(15e9), "ns")
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

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("whitesmoke")

        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(14, 4 * len(panels)),
            sharex=True,
            constrained_layout=True,
        )

        for ax, (grid, label) in zip(axes, panels):
            grid_plot = grid.copy()
            grid_plot[grid_plot == 0] = np.nan
            vmax = float(np.nanmax(grid_plot)) if np.any(~np.isnan(grid_plot)) else 1.0

            im = ax.pcolormesh(
                time_edges,
                az_deg_edges,
                grid_plot.T,
                cmap=cmap,
                vmin=0.5,
                vmax=vmax,
                shading="flat",
            )
            cbar = fig.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label("Counts", fontsize=9)

            ax.set_ylabel(label, fontsize=9)
            ax.set_ylim(0, 360)
            ax.set_yticks(range(0, 361, 60))

        axes[-1].set_xlabel("Time (UTC)", fontsize=10)
        fig.suptitle(
            f"Lo L1B DE Histogram — {self.instrument} — binned by spin phase",
            fontsize=12,
            fontweight="bold",
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

        # Log rainbow+white colormap for 2D histograms
        rc = plt.cm.viridis(np.linspace(0, 1, 255))
        log_cmap = mcolors.LinearSegmentedColormap.from_list("log_viridis", rc)
        log_cmap.set_bad("whitesmoke")

        long_edges = np.arange(0, 352, 2)
        short_edges = np.arange(0, 20.5, 0.5)

        # ------------------------------------------------------------------ #
        # Figure 1 — 2D histograms                                            #
        # ------------------------------------------------------------------ #
        layout = [
            ("tof0", "tof1", long_edges, long_edges),
            ("tof0", "tof2", long_edges, long_edges),
            ("tof1", "tof2", long_edges, long_edges),
            ("tof3", "tof0", short_edges, long_edges),
            ("tof3", "tof1", short_edges, long_edges),
            ("tof3", "tof2", short_edges, long_edges),
        ]

        fig1, axes1 = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)
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
                xe_out, ye_out, h_plot, cmap=log_cmap, norm=norm, shading="flat"
            )
            fig1.colorbar(im, ax=ax, label="Counts", pad=0.02)

        fig1.suptitle(
            f"Lo L1B 2D TOF Histograms — {self.instrument}",
            fontsize=12,
            fontweight="bold",
        )
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
            ax.stairs(
                counts, edges, fill=True, color="steelblue", alpha=0.7, linewidth=0.5
            )
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)
            ax.set_ylabel(f"{label} counts", fontsize=9)
            ax.set_xlabel("TOF (ns)", fontsize=9)
            ax.tick_params(labelsize=8)

        fig2.suptitle(
            f"Lo L1B 1D TOF Histograms — {self.instrument}",
            fontsize=12,
            fontweight="bold",
        )
        plt.show()
