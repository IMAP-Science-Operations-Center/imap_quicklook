"""Test quicklook_generator."""

from plotting.quicklook_generator import (
    IdexQuicklookGenerator,
    MagQuicklookGenerator,
    SwapiQuicklookGenerator,
    get_instrument_quicklook,
)


def test_generate_quicklook_instance():
    """Test if the correct class is generated."""
    mag_file = "imap_mag_l1a_norm-magi_20251017_v001.cdf"
    idex_file = "imap_idex_l1a_sci-1week_20231018_v006.cdf"
    swapi_file = "imap_swapi_l2_quicklook_00000000_v000.cdf"

    assert isinstance(get_instrument_quicklook(mag_file), MagQuicklookGenerator)
    assert isinstance(get_instrument_quicklook(idex_file), IdexQuicklookGenerator)
    assert isinstance(get_instrument_quicklook(swapi_file), SwapiQuicklookGenerator)
