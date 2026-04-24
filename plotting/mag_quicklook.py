"""MAG quicklook generator."""

from __future__ import annotations

import matplotlib.pyplot as plt

from plotting.base_quicklook import QuicklookGenerator, convert_j2000_to_utc


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

        axes[-1].set_xlabel("Time (UTC)", fontsize=10)
        fig.suptitle(
            "MAG Sensor Coordinates (X, Y, Z) [nT]", fontsize=12, fontweight="bold"
        )
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
        axes[3].set_xlabel("Time (UTC)", fontsize=10)

        fig.suptitle(
            "MAG GSE Coordinates (X, Y, Z) + Magnitude [nT]",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()
