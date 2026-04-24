"""IDEX quicklook generator."""

from __future__ import annotations

import matplotlib.pyplot as plt

from plotting.base_quicklook import QuicklookGenerator


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

        axes[-1].set_xlabel("Time (μs)", fontsize=10)
        fig.suptitle("IDEX L1A Dust Impact Waveforms", fontsize=12, fontweight="bold")
        plt.show()
