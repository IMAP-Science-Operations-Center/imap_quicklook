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

# Only plot types with working implementations (rtn and gse are not yet implemented)
MAG_PLOT_TYPES = ["mag sensor co-ord"]


def generate_mag_quicklooks(data_dir: Path) -> None:
    """Generate all implemented MAG quicklook plots for every CDF file found.

    Parameters
    ----------
    data_dir : Path
        Root directory to search for MAG CDF files.
    """
    cdf_files = sorted(data_dir.rglob("imap_mag_*.cdf"))

    if not cdf_files:
        logger.warning("No MAG CDF files found under %s", data_dir)
        return

    logger.info("Found %d MAG file(s):", len(cdf_files))
    for f in cdf_files:
        logger.info("  %s", f.name)

    for cdf_file in cdf_files:
        logger.info("Processing %s", cdf_file.name)
        dataset = load_cdf(cdf_file)

        # Construct generator directly so we can pass a full path instead of
        # relying on dataset_into_xarray's nested directory resolution.
        gen = MagQuicklookGenerator.__new__(MagQuicklookGenerator)
        gen.data_set = dataset
        gen.instrument = "mag"

        for plot_type in MAG_PLOT_TYPES:
            logger.info("  Generating '%s'", plot_type)
            gen.two_dimensional_plot(plot_type)


if __name__ == "__main__":
    generate_mag_quicklooks(DATA_DIR)
