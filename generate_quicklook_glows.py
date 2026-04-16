"""Generate quicklook plots for all GLOWS CDF files found in the data directory."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from plotting.cdf.cdf_utils import load_cdf
from plotting.quicklook_generator import GlowsQuicklookGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "plotting" / "data"

_DATE_RE = re.compile(r"_(\d{8}-repoint\d+|\d{8})_")


def _extract_date_tag(path: Path) -> str | None:
    """
    Return the date (+ optional repoint) tag from an IMAP filename.

    Parameters
    ----------
    path : Path
        Path to an IMAP CDF file.

    Returns
    -------
    str or None
        The date tag (e.g. ``"20260309"`` or ``"20260309-repoint00180"``)
        or ``None`` if no tag is found.
    """
    m = _DATE_RE.search(path.name)
    return m.group(1) if m else None


def find_glows_files(data_dir: Path, level: str, descriptor: str) -> dict[str, Path]:
    """
    Return the latest GLOWS CDF file per date tag matching level and descriptor.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``glows/<level>/`` subdirectories.
    level : str
        Data level directory name (e.g. ``"l1b"`` or ``"l2"``).
    descriptor : str
        Substring to match against filenames (e.g. ``"hist"``).

    Returns
    -------
    dict[str, Path]
        Mapping of date tag → latest matching CDF file.
    """
    level_dir = data_dir / "glows" / level
    if not level_dir.exists():
        return {}
    seen: dict[str, Path] = {}
    for f in level_dir.rglob("*.cdf"):
        if descriptor in f.name:
            tag = _extract_date_tag(f)
            if tag and (tag not in seen or f.name > seen[tag].name):
                seen[tag] = f
    return seen


def generate_glows_quicklooks(data_dir: Path) -> None:
    """
    Generate all implemented GLOWS quicklook plots for every CDF file found.

    For each L1B hist file:
      1. general quicklook — 10-panel overview (pairs with matching L2 file if available)
      2. ancillary data    — 5×2 ancillary parameter panel

    Parameters
    ----------
    data_dir : Path
        Root directory containing ``glows/l1b/`` and ``glows/l2/`` subdirectories.
    """
    l1b_files = find_glows_files(data_dir, "l1b", "hist")
    l2_files = find_glows_files(data_dir, "l2", "hist")

    if not l1b_files:
        logger.warning("No GLOWS l1b hist files found under %s", data_dir)
        return

    logger.info("Found %d GLOWS l1b hist file(s):", len(l1b_files))
    for tag, f in sorted(l1b_files.items()):
        logger.info("  [%s] %s", tag, f.name)

    for date_tag, l1b_file in sorted(l1b_files.items()):
        logger.info("Processing %s", l1b_file.name)

        gen = GlowsQuicklookGenerator.__new__(GlowsQuicklookGenerator)
        gen.data_set = load_cdf(l1b_file)
        gen.instrument = "glows"

        # Attach matching L2 file if available
        if date_tag in l2_files:
            logger.info("  Loading L2: %s", l2_files[date_tag].name)
            gen.data_set_l2 = load_cdf(l2_files[date_tag])
        else:
            logger.warning(
                "  No matching L2 file for %s — sky map panel will be empty.", date_tag
            )
            gen.data_set_l2 = None

        for plot_type in ("general quicklook", "ancillary data"):
            logger.info("  Generating '%s'", plot_type)
            gen.two_dimensional_plot(plot_type)


if __name__ == "__main__":
    generate_glows_quicklooks(DATA_DIR)
