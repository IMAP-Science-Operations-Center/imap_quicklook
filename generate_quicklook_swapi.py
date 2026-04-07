"""Generate quicklook plots for all SWAPI CDF files found in the data directory."""

from __future__ import annotations

import logging
from pathlib import Path

from plotting.cdf.cdf_utils import load_cdf
from plotting.quicklook_generator import SwapiQuicklookGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "plotting" / "data"

# Maps data level prefix to the plot types that require it.
# A prefix of "l1" matches any l1 variant (l1a, l1b, l1c, l1d).
LEVEL_PLOT_MAP: dict[str, list[str]] = {
    "l2": [
        "count rates",
        "count rate line",
        "absolute detection efficiency",
        "1d energy distribution",
    ],
    "l1": ["count line"],
}


def find_files_for_level(
    data_dir: Path, instrument: str, level_prefix: str
) -> list[Path]:
    """
    Return all CDF files for an instrument whose data level starts with ``level_prefix``.

    Parameters
    ----------
    data_dir : Path
        Root data directory (contains ``<instrument>/<level>/`` subdirectories).
    instrument : str
        Instrument name (used as the first subdirectory).
    level_prefix : str
        Data level or prefix to match (e.g. ``"l2"`` or ``"l1"``).

    Returns
    -------
    list[Path]
        Sorted list of matching CDF file paths.
    """
    instrument_dir = data_dir / instrument
    if not instrument_dir.exists():
        return []
    return sorted(
        cdf_file
        for level_dir in instrument_dir.iterdir()
        if level_dir.is_dir() and level_dir.name.startswith(level_prefix)
        for cdf_file in level_dir.rglob("*.cdf")
    )


def generate_swapi_quicklooks(data_dir: Path) -> None:
    """
    Generate all implemented SWAPI quicklook plots for every CDF file found.

    Plot types are matched to files by data level:

    - l2 files  → count rates, count rate line, absolute detection efficiency, 1d energy distribution
    - l1* files → count line

    Parameters
    ----------
    data_dir : Path
        Root directory containing ``swapi/<level>/`` subdirectories.
    """
    for level_prefix, plot_types in LEVEL_PLOT_MAP.items():
        cdf_files = find_files_for_level(data_dir, "swapi", level_prefix)

        if not cdf_files:
            logger.warning(
                "No SWAPI %s* files found — skipping: %s", level_prefix, plot_types
            )
            continue

        logger.info(
            "Found %d SWAPI %s* file(s) for %s:",
            len(cdf_files),
            level_prefix,
            plot_types,
        )
        for f in cdf_files:
            logger.info("  %s", f.name)

        for cdf_file in cdf_files:
            logger.info("Processing %s", cdf_file.name)
            dataset = load_cdf(cdf_file)

            # Construct generator directly so we can pass a full path instead of
            # relying on dataset_into_xarray's directory resolution.
            gen = SwapiQuicklookGenerator.__new__(SwapiQuicklookGenerator)
            gen.data_set = dataset
            gen.instrument = "swapi"

            for plot_type in plot_types:
                logger.info("  Generating '%s'", plot_type)
                gen.two_dimensional_plot(plot_type)


if __name__ == "__main__":
    generate_swapi_quicklooks(DATA_DIR)
