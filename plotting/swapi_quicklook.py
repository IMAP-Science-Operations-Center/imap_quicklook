"""SWAPI quicklook generator."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting.base_quicklook import QuicklookGenerator, convert_j2000_to_utc


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

    def swapi_1d_energy_distribution(self) -> None:
        """
        Generate SWAPI 1D energy distribution plot.

        Averages the coincidence count rate into 10-minute bins (starting from
        the first timestamp) and plots mean vs ESA energy for each interval.
        Only coarse ESA steps (indices 0–62) are included. Error bars show the
        standard deviation of samples within each bin.
        """
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
