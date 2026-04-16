"""Hi quicklook generator."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plotting.base_quicklook import QuicklookGenerator, convert_j2000_to_utc


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
