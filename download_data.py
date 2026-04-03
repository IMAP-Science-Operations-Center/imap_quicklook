"""Download one CDF file per IMAP instrument for a given date.

Files are saved under plotting/data/ so that the existing
``dataset_into_xarray`` loader in plotting/cdf/cdf_utils.py can find them
without any path changes.

Usage
-----
# Download data for today (default)
python download_data.py

# Download data for a specific date
python download_data.py --date 20251017

# Download only specific instruments
python download_data.py --date 20251017 --instruments mag swapi idex

# Set a global default data level (used for all instruments)
python download_data.py --date 20251017 --data-level l1a

# Set per-instrument data levels using instrument:level syntax
python download_data.py --date 20251017 --data-level mag:l1a swapi:l2 idex:l1b

# Mix: per-instrument overrides with a global fallback for everything else
python download_data.py --date 20251017 --data-level l1a mag:l2

# See all options
python download_data.py --help
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import date
from pathlib import Path

import imap_data_access

logger = logging.getLogger(__name__)

# All IMAP science instruments available from the SDC
INSTRUMENTS = [
    "codice",
    "glows",
    "hi",
    "hit",
    "idex",
    "lo",
    "mag",
    "swapi",
    "swe",
    "ultra",
]

# Data levels tried in order of preference (highest processing level first)
DATA_LEVELS = ["l2", "l1c", "l1b", "l1a", "l1", "l0"]

# Default data directory: plotting/data/ (matches path expected by cdf_utils.py)
DEFAULT_DATA_DIR = Path(__file__).parent / "plotting" / "data"


def parse_data_levels(values: list[str]) -> tuple[str | None, dict[str, str]]:
    """Parse ``--data-level`` values into a global default and per-instrument overrides.

    Each value is either a plain level (``"l1a"``) treated as the global
    default, or an ``instrument:level`` pair (``"mag:l1a"``) treated as an
    override for that instrument.  If multiple plain levels are given the last
    one wins.

    Parameters
    ----------
    values : list[str]
        Raw values from the ``--data-level`` argument.

    Returns
    -------
    tuple[str or None, dict[str, str]]
        ``(global_default, per_instrument)`` where ``global_default`` is
        ``None`` if no plain level was supplied.
    """
    global_default: str | None = None
    per_instrument: dict[str, str] = {}

    for value in values:
        if ":" in value:
            instrument, level = value.split(":", 1)
            per_instrument[instrument.strip()] = level.strip()
        else:
            global_default = value.strip()

    return global_default, per_instrument


def _move_to_data_dir(staged: Path, instrument: str, data_level: str, data_dir: Path) -> Path:
    """Move a staged download into the ``data_dir/<instrument>/<data_level>/`` structure.

    Parameters
    ----------
    staged : Path
        Path where imap_data_access placed the downloaded file.
    instrument : str
        Instrument name (used as the first subdirectory).
    data_level : str
        Data level (used as the second subdirectory).
    data_dir : Path
        Root data directory for this repository.

    Returns
    -------
    Path
        Final location of the file.
    """
    dest = data_dir / instrument / data_level / staged.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), dest)
    return dest


def download_instrument_file(
    instrument: str,
    target_date: str,
    data_dir: Path,
    data_level: str | None = None,
    fallback: bool = True,
) -> Path | None:
    """Download one CDF file for a given instrument and date.

    Tries data levels from highest to lowest (l2 → l0) unless a specific
    level is requested. If no file exists for ``target_date`` and
    ``fallback`` is ``True``, downloads the most recent available file
    instead.

    Downloaded files are moved into ``data_dir/<instrument>/<data_level>/``.

    Parameters
    ----------
    instrument : str
        IMAP instrument identifier (e.g. ``"mag"``, ``"swapi"``).
    target_date : str
        Date in ``YYYYMMDD`` format.
    data_dir : Path
        Root data directory for this repository.
    data_level : str or None
        If provided, only query this data level. Otherwise try all levels
        in order from ``DATA_LEVELS``.
    fallback : bool
        If ``True`` and no file exists for ``target_date``, download the
        most recent available file.

    Returns
    -------
    Path or None
        Local path of the downloaded file, or ``None`` if nothing was found.
    """
    levels_to_try = [data_level] if data_level else DATA_LEVELS

    for level in levels_to_try:
        logger.debug("Querying %s %s on %s", instrument, level, target_date)
        results = imap_data_access.query(
            instrument=instrument,
            data_level=level,
            start_date=target_date,
            end_date=target_date,
            version="latest",
            extension="cdf",
        )
        if results:
            result = results[0]
            logger.info("Downloading %s", result["file_path"])
            staged = imap_data_access.download(result["file_path"])
            dest = _move_to_data_dir(staged, result["instrument"], result["data_level"], data_dir)
            logger.info("Saved to %s", dest)
            return dest

    if not fallback:
        logger.warning("No CDF files found for %s on %s", instrument, target_date)
        return None

    # Fallback: find the most recent file at the highest available level
    logger.warning(
        "No data for %s on %s — falling back to most recent available file.",
        instrument,
        target_date,
    )
    for level in levels_to_try:
        results = imap_data_access.query(
            instrument=instrument,
            data_level=level,
            version="latest",
            extension="cdf",
        )
        if results:
            results.sort(key=lambda r: r["start_date"], reverse=True)
            result = results[0]
            logger.info("Most recent %s %s: %s", instrument, level, result["file_path"])
            staged = imap_data_access.download(result["file_path"])
            dest = _move_to_data_dir(staged, result["instrument"], result["data_level"], data_dir)
            logger.info("Saved to %s", dest)
            return dest

    logger.warning("No CDF files found for %s at any level", instrument)
    return None


def main() -> None:
    """Parse arguments and download one file per instrument."""
    parser = argparse.ArgumentParser(
        description="Download one CDF file per IMAP instrument for a given date.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date",
        default=date.today().strftime("%Y%m%d"),
        help="Date to download data for, in YYYYMMDD format.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Root directory for downloaded files.",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=INSTRUMENTS,
        metavar="INSTRUMENT",
        help="One or more instrument names to download.",
    )
    parser.add_argument(
        "--data-level",
        nargs="+",
        default=[],
        metavar="LEVEL",
        help=(
            "Data level(s) to download. Accepts a global default (e.g. l1a), "
            "per-instrument overrides (e.g. mag:l1a swapi:l2), or both "
            "(e.g. l1a mag:l2). Without this flag the highest available level "
            "is used for every instrument."
        ),
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help=(
            "Disable fallback behaviour. By default, if no file exists for "
            "the requested date the most recent available file is downloaded instead."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # imap_data_access downloads into its own nested structure; use a staging
    # subdirectory and move files into data/<instrument>/<data_level>/ afterwards.
    staging_dir = data_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    imap_data_access.config["DATA_DIR"] = staging_dir

    logger.info("Downloading data for %s → %s", args.date, data_dir)

    global_level, per_instrument_levels = parse_data_levels(args.data_level)

    downloaded: list[tuple[str, Path]] = []
    not_found: list[str] = []

    try:
        for instrument in args.instruments:
            level = per_instrument_levels.get(instrument, global_level)
            path = download_instrument_file(
                instrument, args.date, data_dir, level, fallback=not args.no_fallback
            )
            if path:
                downloaded.append((instrument, path))
            else:
                not_found.append(instrument)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"\nResults for {args.date}:")
    print(f"  Downloaded : {len(downloaded)}")
    for instrument, path in downloaded:
        print(f"    {instrument:12s}  {path}")
    if not_found:
        print(f"  Not found  : {', '.join(not_found)}")


if __name__ == "__main__":
    main()
