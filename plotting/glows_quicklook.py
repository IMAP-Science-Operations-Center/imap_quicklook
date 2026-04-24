"""GLOWS quicklook generator."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec

from plotting.base_quicklook import QuicklookGenerator, convert_j2000_to_utc


class GlowsQuicklookGenerator(QuicklookGenerator):
    """GLOWS subclass for GLOWS quicklook plots."""

    data_set_l2: xr.Dataset | None = None
    data_set_swe: xr.Dataset | None = None
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
          - SWE electron normalized counts from i-ALiRT (spectrogram)
          - SWAPI proton moments from i-ALiRT (density + speed)

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
        ialirt: xr.Dataset | None = getattr(self, "data_set_ialirt", None)

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
        ax_hist.set_ylabel("Spin Angle [°]", fontsize=10)
        ax_hist.set_ylim(0, 360)
        ax_hist.set_title("L1B Histogram", fontsize=11)

        # Panel 2: Total photon counts per block
        ax_counts = fig.add_subplot(gs[1, 0])
        ax_counts.plot(epoch_dt, l1b["number_of_events"].values, color="steelblue")
        ax_counts.set_ylabel("Events per Block", fontsize=10)
        ax_counts.set_xlim(epoch_dt[0], epoch_dt[-1])
        ax_counts.set_title("Total Photon Counts", fontsize=11)

        # Panel 3: Quality flags heatmap (epoch × flag index)
        ax_flags = fig.add_subplot(gs[2, 0])
        flags = l1b["flags"].values
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
        ax_flags.set_ylabel("Flag Index", fontsize=10)
        ax_flags.set_ylim(-0.5, flags.shape[1] - 0.5)
        ax_flags.set_title("Quality Flags", fontsize=11)

        # Panel 4: SWE electron normalized counts from i-ALiRT (spectrogram)
        ax_swe = fig.add_subplot(gs[3, 0])
        fill = -1e31
        if ialirt is not None and "swe_normalized_counts" in ialirt:
            swe_epoch_dt = convert_j2000_to_utc(ialirt["swe_epoch"].values)
            swe_counts = ialirt["swe_normalized_counts"].values.astype(float)
            swe_energy = ialirt["swe_electron_energy"].values
            swe_counts[swe_counts <= fill * 0.9] = np.nan
            valid = swe_counts[np.isfinite(swe_counts)]
            vmin = float(valid.min()) if len(valid) else 1.0
            vmax = float(valid.max()) if len(valid) else 1e4
            im_swe = ax_swe.pcolormesh(
                swe_epoch_dt,
                swe_energy,
                swe_counts.T,
                shading="auto",
                cmap="viridis",
                norm=mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
            )
            plt.colorbar(im_swe, ax=ax_swe, label="Normalized Counts")
            ax_swe.set_ylabel("Energy [eV]")
            ax_swe.set_title("SWE Electron Normalized Counts (i-ALiRT)")
        elif swe is not None:
            swe_epoch_dt = convert_j2000_to_utc(swe["epoch"].values)
            sci = swe["science_data"].values.astype(float)
            swe_fill = swe["science_data"].attrs.get("FILLVAL", None)
            if swe_fill is not None:
                sci[sci == swe_fill] = np.nan
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
            ax_swe.set_title("SWE Electron Normalized Counts (i-ALiRT)")
            ax_swe.set_xticks([])
            ax_swe.set_yticks([])

        # Panel 5: SWAPI proton moments from i-ALiRT
        ax_swapi = fig.add_subplot(gs[4, 0])
        if ialirt is not None and "swapi_pseudo_proton_density" in ialirt:
            swapi_epoch_dt = convert_j2000_to_utc(ialirt["swapi_epoch"].values)
            density = ialirt["swapi_pseudo_proton_density"].values.astype(float)
            speed = ialirt["swapi_pseudo_proton_speed"].values.astype(float)
            density[density <= fill * 0.9] = np.nan
            speed[speed <= fill * 0.9] = np.nan
            ax_swapi.plot(swapi_epoch_dt, density, color="steelblue", linewidth=0.8)
            ax_swapi.set_ylabel("Density [cm⁻³]", color="steelblue")
            ax_swapi.tick_params(axis="y", labelcolor="steelblue")
            ax2_swapi = ax_swapi.twinx()
            ax2_swapi.plot(swapi_epoch_dt, speed, color="tomato", linewidth=0.8)
            ax2_swapi.set_ylabel("Speed [km/s]", color="tomato")
            ax2_swapi.tick_params(axis="y", labelcolor="tomato")
            ax_swapi.set_title("SWAPI Solar Wind (n_p, v_i) — i-ALiRT")
        else:
            ax_swapi.text(
                0.5,
                0.5,
                "SWAPI i-ALiRT data not available\nfor this time period",
                ha="center",
                va="center",
                transform=ax_swapi.transAxes,
                fontsize=11,
                color="gray",
            )
            ax_swapi.set_title("SWAPI Solar Wind (n_p, v_i) — i-ALiRT")
            ax_swapi.set_xticks([])
            ax_swapi.set_yticks([])

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
        hist_sum = l1b["histogram"].values.sum(axis=0)
        ax_l1b_sum.plot(spin_angle_bins, hist_sum, color="steelblue", linewidth=0.8)
        ax_l1b_sum.set_xlabel("Spin Angle [°]")
        ax_l1b_sum.set_ylabel("Total Counts")
        ax_l1b_sum.set_xlim(0, 360)
        ax_l1b_sum.set_title("L1B Histogram vs Spin Angle")

        # Panel 9: L2 photon flux vs spin angle (with uncertainty band)
        ax_flux = fig.add_subplot(gs[3, 1])
        if l2 is not None:
            spin_angle_l2 = l2["spin_angle"].values[0]
            photon_flux = l2["photon_flux"].values[0]
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
            elon = l2["ecliptic_lon"].values[0]
            elat = l2["ecliptic_lat"].values[0]
            flux_sky = l2["photon_flux"].values[0]
            sc = ax_sky.scatter(
                elon, elat, c=flux_sky, cmap="rainbow", s=3, linewidths=0
            )
            plt.colorbar(
                sc,
                ax=ax_sky,
                label="Photon Flux [R]",
                location="bottom",
                pad=0.12,
                fraction=0.04,
            )
            ax_sky.plot(
                266.4, -5.6, "k*", markersize=10, label="Galactic Center", zorder=5
            )
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
        fig.suptitle(f"GLOWS Quicklook — {date_str}", fontsize=13, fontweight="bold")
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

        loc = ds["spacecraft_location_average"].values
        loc_std = ds["spacecraft_location_std_dev"].values
        x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        r_xy = np.sqrt(x**2 + y**2)
        sc_lon = np.degrees(np.arctan2(y, x)) % 360
        sc_lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))
        sc_lon_std = np.degrees(loc_std[:, 1] / np.where(r_xy > 0, r_xy, 1))
        sc_lat_std = np.degrees(loc_std[:, 2] / np.where(r > 0, r, 1))

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
            (sc_lon, sc_lon_std, "Lon avg [deg]", "Lon std dev [deg]"),
            (sc_lat, sc_lat_std, "Lat avg [deg]", "Lat std dev [deg]"),
            (
                ds["number_of_bins_per_histogram"].values,
                ds["number_of_spins_per_block"].values,
                "Num of bins",
                "Spins per block",
            ),
        ]

        fig, axes = plt.subplots(
            5, 2, figsize=(18, 16), sharex=True, constrained_layout=False
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
