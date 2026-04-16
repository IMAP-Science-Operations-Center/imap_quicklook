"""HIT quicklook generator."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plotting.base_quicklook import QuicklookGenerator, convert_j2000_to_utc


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
