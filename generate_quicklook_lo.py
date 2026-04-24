"""Generate quicklook plots for all Lo CDF files found in the data directory."""

from __future__ import annotations

import logging
from pathlib import Path

from plotting.cdf.cdf_utils import load_cdf
from plotting.quicklook_generator import LoQuicklookGenerator
from plotting.save_utils import capture_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "plotting" / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

# Maps descriptor substring to the plot types that require it.
DESCRIPTOR_PLOT_MAP: dict[str, list[str]] = {
    "star": ["star sensor"],
    "histogram": ["histogram"],
    "_de_": ["DE histogram", "DE tof"],
}


def find_lo_files(data_dir: Path, descriptor: str) -> list[Path]:
    """
    Return all Lo CDF files whose filename contains ``descriptor``.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``lo/<level>/`` subdirectories.
    descriptor : str
        Substring to match against filenames (e.g. ``"star"`` or ``"-de"``).

    Returns
    -------
    list[Path]
        Sorted list of matching CDF file paths.
    """
    lo_dir = data_dir / "lo"
    if not lo_dir.exists():
        return []
    return sorted(f for f in lo_dir.rglob("*.cdf") if descriptor in f.name)


def generate_lo_quicklooks(data_dir: Path) -> None:
    """
    Generate all implemented Lo quicklook plots for every CDF file found.

    Plot types are matched to files by descriptor:

    - star files      → star sensor
    - histogram files → histogram
    - de files        → DE histogram, DE tof

    Parameters
    ----------
    data_dir : Path
        Root directory containing ``lo/<level>/`` subdirectories.
    """
    for descriptor, plot_types in DESCRIPTOR_PLOT_MAP.items():
        cdf_files = find_lo_files(data_dir, descriptor)

        if not cdf_files:
            logger.warning(
                "No Lo files matching '%s' found — skipping: %s",
                descriptor,
                plot_types,
            )
            continue

        logger.info(
            "Found %d Lo file(s) matching '%s' for %s:",
            len(cdf_files),
            descriptor,
            plot_types,
        )
        for f in cdf_files:
            logger.info("  %s", f.name)

        for cdf_file in cdf_files:
            logger.info("Processing %s", cdf_file.name)
            dataset = load_cdf(cdf_file)

            gen = LoQuicklookGenerator.__new__(LoQuicklookGenerator)
            gen.data_set = dataset
            gen.instrument = "lo"

            for plot_type in plot_types:
                logger.info("  Generating '%s'", plot_type)
                stem = f"{cdf_file.stem}_{plot_type.replace(' ', '_')}"
                with capture_plots(OUTPUT_DIR / "lo", stem):
                    gen.two_dimensional_plot(plot_type)


if __name__ == "__main__":
    generate_lo_quicklooks(DATA_DIR)
