"""Generate quicklook plots for all IDEX CDF files found in the data directory."""

from __future__ import annotations

import logging
from pathlib import Path

from plotting.cdf.cdf_utils import load_cdf
from plotting.quicklook_generator import IdexQuicklookGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "plotting" / "data"


def generate_idex_quicklooks(data_dir: Path) -> None:
    """Generate IDEX quicklook plots for every CDF file found.

    One plot is produced per dust impact event (epoch) in each file.

    Parameters
    ----------
    data_dir : Path
        Root directory to search for IDEX CDF files.
    """
    cdf_files = sorted(data_dir.rglob("imap_idex_*.cdf"))

    if not cdf_files:
        logger.warning("No IDEX CDF files found under %s", data_dir)
        return

    logger.info("Found %d IDEX file(s):", len(cdf_files))
    for f in cdf_files:
        logger.info("  %s", f.name)

    for cdf_file in cdf_files:
        logger.info("Processing %s", cdf_file.name)
        dataset = load_cdf(cdf_file)

        num_events = dataset.sizes["epoch"]
        logger.info("  %d event(s) found", num_events)

        # Construct generator directly so we can pass a full path instead of
        # relying on dataset_into_xarray's directory resolution.
        gen = IdexQuicklookGenerator.__new__(IdexQuicklookGenerator)
        gen.data_set = dataset
        gen.instrument = "idex"

        for time_index in range(num_events):
            logger.info("  Generating waveform plot for event %d", time_index)
            gen.idex_quicklook(time_index=time_index)


if __name__ == "__main__":
    generate_idex_quicklooks(DATA_DIR)
