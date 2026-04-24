"""Class for abstracting and organizing quicklook plots."""

from __future__ import annotations

from enum import Enum

from imap_data_access import ScienceFilePath

from plotting.base_quicklook import QuicklookGenerator, convert_j2000_to_utc  # noqa: F401
from plotting.glows_quicklook import GlowsQuicklookGenerator
from plotting.hi_quicklook import HiQuicklookGenerator
from plotting.hit_quicklook import HitQuicklookGenerator
from plotting.idex_quicklook import IdexQuicklookGenerator
from plotting.lo_quicklook import LoQuicklookGenerator
from plotting.mag_quicklook import MagQuicklookGenerator
from plotting.swapi_quicklook import SwapiQuicklookGenerator
from plotting.ultra_quicklook import UltraQuicklookGenerator

__all__ = [
    "QuicklookGenerator",
    "convert_j2000_to_utc",
    "GlowsQuicklookGenerator",
    "HiQuicklookGenerator",
    "HitQuicklookGenerator",
    "IdexQuicklookGenerator",
    "LoQuicklookGenerator",
    "MagQuicklookGenerator",
    "SwapiQuicklookGenerator",
    "UltraQuicklookGenerator",
    "QuicklookGeneratorType",
    "get_instrument_quicklook",
]


class QuicklookGeneratorType(Enum):
    """Map instrument to correct dataclass."""

    MAG = MagQuicklookGenerator
    IDEX = IdexQuicklookGenerator
    ULTRA = UltraQuicklookGenerator
    SWAPI = SwapiQuicklookGenerator
    HI = HiQuicklookGenerator
    LO = LoQuicklookGenerator
    GLOWS = GlowsQuicklookGenerator
    HIT = HitQuicklookGenerator


def get_instrument_quicklook(filename: str) -> QuicklookGenerator:
    """
    Determine which abstract class to use for a given file.

    Parameters
    ----------
    filename : str
        Desired file to generate quicklook from.

    Returns
    -------
    QuicklookGenerator
        Proper abstract class for file.
    """
    file_name_dict = ScienceFilePath.extract_filename_components(filename)
    gen_cls = QuicklookGeneratorType[file_name_dict["instrument"].upper()]
    return gen_cls.value(filename)
