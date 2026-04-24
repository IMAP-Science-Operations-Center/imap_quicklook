"""Generate quicklook plots for all HIT CDF files found in the data directory."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from plotting.cdf.cdf_utils import load_cdf
from plotting.quicklook_generator import HitQuicklookGenerator
from plotting.save_utils import capture_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "plotting" / "data"
OUTPUT_DIR = Path(__file__).parent / "output"


_DATE_RE = re.compile(r"_(\d{8})")


def find_ialirt_file(data_dir: Path, cdf_file: Path) -> Path | None:
    """
    Return the matching i-ALiRT CDF file for a given instrument CDF file.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing the ``ialirt/`` subdirectory.
    cdf_file : Path
        Path to the instrument CDF file whose date tag will be matched.

    Returns
    -------
    Path or None
        Path to the matching i-ALiRT file, or ``None`` if not found.
    """
    m = _DATE_RE.search(cdf_file.name)
    if not m:
        return None
    date = m.group(1)
    candidates = sorted(
        (data_dir / "ialirt").glob(f"imap_ialirt_l1_realtime_{date}_*.cdf")
    )
    return candidates[-1] if candidates else None


def find_hit_files(data_dir: Path, level_prefix: str) -> list[Path]:
    """
    Return all HIT CDF files whose data level directory starts with ``level_prefix``.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``hit/<level>/`` subdirectories.
    level_prefix : str
        Data level prefix to match (e.g. ``"l2"``).

    Returns
    -------
    list[Path]
        Sorted list of matching CDF file paths.
    """
    hit_dir = data_dir / "hit"
    if not hit_dir.exists():
        return []
    return sorted(
        cdf_file
        for level_dir in hit_dir.iterdir()
        if level_dir.is_dir() and level_dir.name.startswith(level_prefix)
        for cdf_file in level_dir.rglob("*.cdf")
    )


def generate_hit_quicklooks(data_dir: Path) -> None:
    """
    Generate all implemented HIT quicklook plots for every CDF file found.

    Plot types are matched to files by data level:

    - l2 files → hit ion flux
    - l2 files + matching i-ALiRT file → electron count rate

    Parameters
    ----------
    data_dir : Path
        Root directory containing ``hit/<level>/`` subdirectories.
    """
    cdf_files = find_hit_files(data_dir, "l2")

    if not cdf_files:
        logger.warning("No HIT l2 files found under %s", data_dir)
        return

    logger.info("Found %d HIT l2 file(s):", len(cdf_files))
    for f in cdf_files:
        logger.info("  %s", f.name)

    for cdf_file in cdf_files:
        logger.info("Processing %s", cdf_file.name)
        dataset = load_cdf(cdf_file)

        gen = HitQuicklookGenerator.__new__(HitQuicklookGenerator)
        gen.data_set = dataset
        gen.instrument = "hit"

        logger.info("  Generating 'hit ion flux'")
        with capture_plots(OUTPUT_DIR / "hit", f"{cdf_file.stem}_hit_ion_flux"):
            gen.two_dimensional_plot("hit ion flux")

        ialirt_file = find_ialirt_file(data_dir, cdf_file)
        if ialirt_file:
            logger.info("  Loading i-ALiRT: %s", ialirt_file.name)
            gen.data_set_ialirt = load_cdf(ialirt_file)
            logger.info("  Generating 'electron count rate'")
            with capture_plots(
                OUTPUT_DIR / "hit", f"{cdf_file.stem}_electron_count_rate"
            ):
                gen.two_dimensional_plot("electron count rate")
        else:
            logger.warning(
                "  No i-ALiRT file found for %s — skipping electron count rate",
                cdf_file.name,
            )


if __name__ == "__main__":
    generate_hit_quicklooks(DATA_DIR)
