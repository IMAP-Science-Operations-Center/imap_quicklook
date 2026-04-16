"""Generate quicklook plots for all ULTRA CDF files found in the data directory."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from plotting.cdf.cdf_utils import load_cdf
from plotting.quicklook_generator import UltraQuicklookGenerator

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
        The date tag (e.g. ``"20260107"`` or ``"20260107-repoint00131"``)
        or ``None`` if no tag is found.
    """
    m = _DATE_RE.search(path.name)
    return m.group(1) if m else None


def _latest_files(directory: Path, descriptor: str) -> dict[str, Path]:
    """
    Return the highest-version CDF file per date tag matching descriptor.

    Parameters
    ----------
    directory : Path
        Directory to search recursively for CDF files.
    descriptor : str
        Substring to match against filenames.

    Returns
    -------
    dict[str, Path]
        Mapping of date tag → highest-version matching CDF file.
    """
    seen: dict[str, Path] = {}
    for f in directory.rglob("*.cdf"):
        if descriptor in f.name:
            tag = _extract_date_tag(f)
            if tag and (tag not in seen or f.name > seen[tag].name):
                seen[tag] = f
    return seen


def find_ultra_files(data_dir: Path, level: str, descriptor: str) -> dict[str, Path]:
    """
    Return the latest CDF file per date tag matching level and descriptor.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``ultra/<level>/`` subdirectories.
    level : str
        Data level directory name (e.g. ``"l1a"`` or ``"l1b"``).
    descriptor : str
        Substring to match against filenames.

    Returns
    -------
    dict[str, Path]
        Mapping of date tag → latest matching CDF file.
    """
    level_dir = data_dir / "ultra" / level
    if not level_dir.exists():
        return {}
    return _latest_files(level_dir, descriptor)


def find_ultra_de_files(data_dir: Path, sensor: str = "45sensor") -> dict[str, Path]:
    """
    Return the latest L1B DE CDF file per date tag for ``sensor``.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``ultra/l1b/`` subdirectories.
    sensor : str
        Sensor identifier (e.g. ``"45sensor"``).

    Returns
    -------
    dict[str, Path]
        Mapping of date tag → latest L1B DE CDF file.
    """
    return find_ultra_files(data_dir, "l1b", f"{sensor}-de")


def find_ultra_aux_file(
    data_dir: Path, date_tag: str, sensor: str = "45sensor"
) -> Path | None:
    """
    Return the latest L1A AUX CDF file matching ``date_tag``.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``ultra/l1a/`` subdirectories.
    date_tag : str
        Date tag to match against filenames.
    sensor : str
        Sensor identifier (e.g. ``"45sensor"``).

    Returns
    -------
    Path or None
        Path to the matching AUX file, or ``None`` if not found.
    """
    candidates = sorted(find_ultra_files(data_dir, "l1a", f"{sensor}-aux").values())
    matches = [f for f in candidates if date_tag in f.name]
    return matches[-1] if matches else None


def generate_ultra_quicklooks(data_dir: Path) -> None:
    """
    Generate all implemented ULTRA quicklook plots.

    For each L1B DE file found:
      1. raw image events   — cTOF + energy spectrograms + voltage state (Slide 2)
      2. tof spin spectrogram — side-by-side cTOF and spin-phase spectrograms (Slide 4)
      3. tof spectrum         — 1-D TOF histogram with 10 ns dip annotation (Slide 4)

    For each L1A priority-1-de file found:
      4. priority 1 events  — spin phase + energy PH spectrograms + start rates (Slide 3)

    For each L1A priority-2-de file found:
      5. priority 2 events  — same layout as priority 1 (Slide 3)

    Parameters
    ----------
    data_dir : Path
        Root directory containing ``ultra/l1b/`` and ``ultra/l1a/``
        subdirectories.
    """
    sensor = "45sensor"

    # ── L1B DE plots ──────────────────────────────────────────────────
    de_files = find_ultra_de_files(data_dir, sensor)
    if not de_files:
        logger.warning("No ULTRA L1B %s-de files found.", sensor)
    else:
        logger.info("Found %d ULTRA L1B DE file(s):", len(de_files))
        for tag, f in sorted(de_files.items()):
            logger.info("  [%s] %s", tag, f.name)

        for date_tag, de_file in sorted(de_files.items()):
            logger.info("Processing L1B DE: %s", de_file.name)
            de_ds = load_cdf(de_file)

            gen = UltraQuicklookGenerator.__new__(UltraQuicklookGenerator)
            gen.data_set = de_ds
            gen.instrument = "ultra"

            aux_file = find_ultra_aux_file(data_dir, date_tag, sensor)
            if aux_file:
                logger.info("  Loading L1A AUX: %s", aux_file.name)
                gen.data_set_aux = load_cdf(aux_file)  # type: ignore[attr-defined]
            else:
                logger.warning("  No L1A AUX file — voltage panel will be empty.")
                gen.data_set_aux = None  # type: ignore[attr-defined]

            for plot_type in (
                "raw image events",
                "tof spin spectrogram",
                "tof spectrum",
            ):
                logger.info("  Generating '%s'", plot_type)
                gen.two_dimensional_plot(plot_type)

    # ── Priority event plots ───────────────────────────────────────────
    for priority, descriptor in [
        (1, f"{sensor}-priority-1-de"),
        (2, f"{sensor}-priority-2-de"),
    ]:
        pri_files = find_ultra_files(data_dir, "l1a", descriptor)
        if not pri_files:
            logger.warning("No ULTRA L1A %s files found.", descriptor)
            continue

        logger.info("Found %d ULTRA L1A priority-%d file(s):", len(pri_files), priority)
        for tag, f in sorted(pri_files.items()):
            logger.info("  [%s] %s", tag, f.name)

        for date_tag, pri_file in sorted(pri_files.items()):
            logger.info("Processing priority-%d: %s", priority, pri_file.name)
            gen = UltraQuicklookGenerator.__new__(UltraQuicklookGenerator)
            gen.data_set = load_cdf(pri_file)
            gen.instrument = "ultra"

            logger.info("  Generating 'priority %d events'", priority)
            gen.two_dimensional_plot(f"priority {priority} events")


if __name__ == "__main__":
    generate_ultra_quicklooks(DATA_DIR)
