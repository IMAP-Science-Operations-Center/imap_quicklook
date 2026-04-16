"""Generate quicklook plots for all MAG CDF files found in the data directory."""

from __future__ import annotations

import logging
from pathlib import Path

from plotting.cdf.cdf_utils import load_cdf
from plotting.quicklook_generator import MagQuicklookGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "plotting" / "data"

# Maps data level prefix to the plot types that require it.
LEVEL_PLOT_MAP: dict[str, list[str]] = {
    "l1a": ["mag sensor co-ord"],
    "l1d": ["gse"],
}


def find_mag_files_for_level(data_dir: Path, level_prefix: str) -> list[Path]:
    """
    Return all MAG CDF files whose data level directory starts with ``level_prefix``.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``mag/<level>/`` subdirectories.
    level_prefix : str
        Data level prefix to match (e.g. ``"l1a"`` or ``"l1d"``).

    Returns
    -------
    list[Path]
        Sorted list of matching CDF file paths.
    """
    mag_dir = data_dir / "mag"
    if not mag_dir.exists():
        return []
    return sorted(
        cdf_file
        for level_dir in mag_dir.iterdir()
        if level_dir.is_dir() and level_dir.name.startswith(level_prefix)
        for cdf_file in level_dir.rglob("*.cdf")
    )


def generate_mag_quicklooks(data_dir: Path) -> None:
    """
    Generate all implemented MAG quicklook plots for every CDF file found.

    Plot types are matched to files by data level:

    - l1a files → mag sensor co-ord (MAGo and MAGi)
    - l1d files → gse

    Parameters
    ----------
    data_dir : Path
        Root directory containing ``mag/<level>/`` subdirectories.
    """
    for level_prefix, plot_types in LEVEL_PLOT_MAP.items():
        cdf_files = find_mag_files_for_level(data_dir, level_prefix)

        if not cdf_files:
            logger.warning(
                "No MAG %s* files found — skipping: %s", level_prefix, plot_types
            )
            continue

        logger.info(
            "Found %d MAG %s* file(s) for %s:",
            len(cdf_files),
            level_prefix,
            plot_types,
        )
        for f in cdf_files:
            logger.info("  %s", f.name)

        for cdf_file in cdf_files:
            logger.info("Processing %s", cdf_file.name)
            dataset = load_cdf(cdf_file)

            gen = MagQuicklookGenerator.__new__(MagQuicklookGenerator)
            gen.data_set = dataset
            gen.instrument = "mag"

            for plot_type in plot_types:
                logger.info("  Generating '%s'", plot_type)
                gen.two_dimensional_plot(plot_type)


if __name__ == "__main__":
    generate_mag_quicklooks(DATA_DIR)
