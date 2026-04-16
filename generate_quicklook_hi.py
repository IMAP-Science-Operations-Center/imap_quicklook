"""Generate quicklook plots for all Hi CDF files found in the data directory."""

from __future__ import annotations

import logging
from pathlib import Path

from plotting.cdf.cdf_utils import load_cdf
from plotting.quicklook_generator import HiQuicklookGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "plotting" / "data"

# Maps descriptor substring to the plot types that require it.
DESCRIPTOR_PLOT_MAP: dict[str, list[str]] = {
    "hist": ["HI histogram"],
    "-de": ["DE Hisogram", "DE TOF Plots"],
}


def find_hi_files(data_dir: Path, descriptor: str) -> list[Path]:
    """
    Return all Hi CDF files whose filename contains ``descriptor``.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``hi/<level>/`` subdirectories.
    descriptor : str
        Substring to match against filenames (e.g. ``"hist"`` or ``"-de"``).

    Returns
    -------
    list[Path]
        Sorted list of matching CDF file paths.
    """
    hi_dir = data_dir / "hi"
    if not hi_dir.exists():
        return []
    return sorted(f for f in hi_dir.rglob("*.cdf") if descriptor in f.name)


def generate_hi_quicklooks(data_dir: Path) -> None:
    """
    Generate all implemented Hi quicklook plots for every CDF file found.

    Plot types are matched to files by descriptor:

    - hist files → HI histogram
    - de files   → DE Hisogram, DE TOF Plots

    Parameters
    ----------
    data_dir : Path
        Root directory containing ``hi/<level>/`` subdirectories.
    """
    for descriptor, plot_types in DESCRIPTOR_PLOT_MAP.items():
        cdf_files = find_hi_files(data_dir, descriptor)

        if not cdf_files:
            logger.warning(
                "No Hi files matching '%s' found — skipping: %s",
                descriptor,
                plot_types,
            )
            continue

        logger.info(
            "Found %d Hi file(s) matching '%s' for %s:",
            len(cdf_files),
            descriptor,
            plot_types,
        )
        for f in cdf_files:
            logger.info("  %s", f.name)

        for cdf_file in cdf_files:
            logger.info("Processing %s", cdf_file.name)
            dataset = load_cdf(cdf_file)

            gen = HiQuicklookGenerator.__new__(HiQuicklookGenerator)
            gen.data_set = dataset
            gen.instrument = cdf_file.name.split("_")[2]  # e.g. "45sensor"

            for plot_type in plot_types:
                logger.info("  Generating '%s'", plot_type)
                gen.two_dimensional_plot(plot_type)


if __name__ == "__main__":
    generate_hi_quicklooks(DATA_DIR)
