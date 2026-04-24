"""HIT quicklook generator."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plotting.base_quicklook import QuicklookGenerator, convert_j2000_to_utc


class HitQuicklookGenerator(QuicklookGenerator):
    """HIT subclass for HIT quicklook plots."""

    data_set_ialirt: xr.Dataset | None = None

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

        ep_ns = epoch_dt.astype("datetime64[ns]").astype(np.int64)
        thirty_s_ns = int(30e9)
        mid_ns = (ep_ns[:-1] + ep_ns[1:]) // 2
        edge_ns = np.concatenate(
            [[ep_ns[0] - thirty_s_ns], mid_ns, [ep_ns[-1] + thirty_s_ns]]
        )
        time_edges = edge_ns.astype("datetime64[ns]")

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

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("whitesmoke")

        fig, axes = plt.subplots(
            4, 1, figsize=(14, 11), sharex=True, constrained_layout=False
        )
        fig.subplots_adjust(hspace=0, right=0.80, top=0.93, bottom=0.08)

        for ax, (flux_var, en_key, dm_key, dp_key, label, en_unit) in zip(
            axes, species
        ):
            flux = ds[flux_var].values.astype(float)
            energy_mean = ds[en_key].values
            dm = ds[dm_key].values
            dp = ds[dp_key].values

            energy_edges = np.append(energy_mean - dm, energy_mean[-1] + dp[-1])
            flux[flux <= 0] = np.nan

            valid = flux[np.isfinite(flux)]
            vmin = float(valid.min()) if len(valid) else 1e-4
            vmax = float(valid.max()) if len(valid) else 1e4
            if vmin >= vmax:
                vmin, vmax = 1e-4, 1e4

            im = ax.pcolormesh(
                time_edges,
                energy_edges,
                flux.T,
                norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=cmap,
                shading="flat",
            )
            ax.set_yscale("log")
            ax.set_ylabel(f"Energy\n({en_unit})", fontsize=9)

            ax.text(
                0.99,
                0.97,
                label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="bold",
            )

            cbar_ax = fig.add_axes(
                [
                    0.82,
                    ax.get_position().y0 + 0.005,
                    0.015,
                    ax.get_position().height - 0.01,
                ]
            )
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Flux\n[cm⁻² sr⁻¹ s⁻¹\n(MeV/nuc)⁻¹]", fontsize=7)
            cbar.ax.tick_params(labelsize=7)

        axes[-1].set_xlabel("UTC Time", fontsize=10)
        axes[-1].tick_params(axis="x", labelsize=8)
        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)

        date_str = str(epoch_dt[0])[:10]
        fig.suptitle(
            f"HIT Ion Flux Spectrograms — {date_str}", fontsize=12, fontweight="bold"
        )
        plt.show()

    def electron_count_rate(self) -> None:
        """
        Generate HIT i-ALiRT electron count rate plot.

        Two vertically stacked line plots showing electron count rates over
        time: B-side (sunward) on top, A-side (anti-sunward) on bottom.

        X = Time (UTC)
        Y = Count Rate (s⁻¹)
        """
        ialirt: xr.Dataset | None = getattr(self, "data_set_ialirt", None)
        if ialirt is None:
            raise ValueError("Must load i-ALiRT dataset.")

        epoch_dt = convert_j2000_to_utc(ialirt["hit_epoch"].values)
        fill = -1e31

        def _mask(arr: np.ndarray) -> np.ndarray:
            """
            Replace fill-value entries with NaN.

            Parameters
            ----------
            arr : np.ndarray
                Input array that may contain fill values.

            Returns
            -------
            np.ndarray
                Float copy of ``arr`` with fill values replaced by NaN.
            """
            out = arr.astype(float).copy()
            out[out <= fill * 0.9] = np.nan
            return out

        b_side = _mask(ialirt["hit_e_b_side_med_en"].values)
        a_side = _mask(ialirt["hit_e_a_side_med_en"].values)

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        axes[0].plot(epoch_dt, b_side, color="steelblue", linewidth=0.8)
        axes[0].set_ylabel("Count Rate [s⁻¹]")
        axes[0].set_title("HIT e⁻ <1 MeV — B Side (Sunward)")

        axes[1].plot(epoch_dt, a_side, color="tomato", linewidth=0.8)
        axes[1].set_ylabel("Count Rate [s⁻¹]")
        axes[1].set_xlabel("Time (UTC)")
        axes[1].set_title("HIT e⁻ <1 MeV — A Side (Anti-Sunward)")

        plt.tight_layout()
        plt.show()
