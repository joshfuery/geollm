import numpy as np
import pytest

from geoagentbench.terrain_encode import (
    decode_terrarium,
    encode_terrarium,
    write_terrain_png,
    write_terrain_meta,
)


def test_encode_decode_roundtrip_flat():
    z = np.full((8, 8), 1234.5)
    rgb = encode_terrarium(z)
    back = decode_terrarium(rgb)
    assert np.allclose(back, z, atol=1 / 256)


def test_encode_decode_handles_negative_and_high():
    z = np.array([[-200.0, 0.0, 8848.0]])
    rgb = encode_terrarium(z)
    back = decode_terrarium(rgb)
    assert np.allclose(back, z, atol=1 / 256)


def test_encode_clamps_nan_and_inf():
    z = np.array([[np.nan, np.inf, -np.inf, 500.0]])
    rgb = encode_terrarium(z)
    back = decode_terrarium(rgb)
    assert back[0, 3] == pytest.approx(500.0, abs=1 / 256)
    assert np.isfinite(back).all()


def test_write_files_produces_valid_png(tmp_path):
    z = np.linspace(0, 3000, 32 * 32).reshape(32, 32)
    png = write_terrain_png(z, tmp_path / "t.png")
    meta = write_terrain_meta(
        tmp_path / "t.meta.json",
        bbox=(-119.6, 37.7, -119.5, 37.8),
        elev_min_m=float(z.min()),
        elev_max_m=float(z.max()),
        pixel_size_m=30.0,
        shape=z.shape,
        source="test",
    )
    assert png.exists() and png.stat().st_size > 0
    assert meta.exists() and meta.stat().st_size > 0
