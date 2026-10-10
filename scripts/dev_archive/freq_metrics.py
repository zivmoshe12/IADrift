"""Frequency-resolved reconstruction quality for 256x256 RGB images in [-1, 1].

Band SNR (dB) = 10 log10( |HR content in band|^2 / |(reconstruction - HR) in band|^2 ),
from Hann-windowed 2D FFTs summed over RGB channels. Higher is better; 0 dB means
the error in that band is as large as the content itself (e.g. leaving the band
empty); negative means the added detail is worse than none.

Frequencies are radial, in cycles per HR pixel (0 to ~0.71). A 64x64 LR image
(SR x4) can represent at most LR_NYQUIST = 0.125 cycles/pixel, so "low" and
"mid" are constrained by the measurement and "high"/"very high" come from the prior.
"""

import numpy as np

SIZE = 256
LR_NYQUIST = 0.125
BANDS = (
    ("low", 0.0, 0.0625),
    ("mid", 0.0625, 0.125),
    ("high", 0.125, 0.25),
    ("very high", 0.25, np.inf),
)

_fy = np.fft.fftfreq(SIZE)[:, None]
_fx = np.fft.fftfreq(SIZE)[None, :]
RADIUS = np.sqrt(_fx ** 2 + _fy ** 2)
WINDOW = np.outer(np.hanning(SIZE), np.hanning(SIZE))[..., None]


def _power(image):
    """|FFT|^2 of a Hann-windowed [256, 256, 3] array, summed over channels."""
    image = np.asarray(image, dtype=np.float64)
    assert image.shape == (SIZE, SIZE, 3)
    spectrum = np.fft.fft2(image * WINDOW, axes=(0, 1))
    return np.sum(np.abs(spectrum) ** 2, axis=-1)


def _snr_db(signal, error):
    return float(10 * np.log10(signal.sum() / error.sum()))


def band_snr_db(reconstruction, reference):
    """{band name: SNR in dB} for the four BANDS."""
    signal = _power(reference)
    error = _power(np.asarray(reconstruction, dtype=np.float64) - reference)
    return {
        name: _snr_db(signal[(RADIUS >= lo) & (RADIUS < hi)], error[(RADIUS >= lo) & (RADIUS < hi)])
        for name, lo, hi in BANDS
    }


_PAD = SIZE // 2
_fp = np.fft.fftfreq(SIZE + 2 * _PAD)
_RADIUS_PADDED = np.sqrt(_fp[None, :] ** 2 + _fp[:, None] ** 2)


def band_filter(image, lo, hi):
    """Keep radial frequencies in [lo, hi) cycles/pixel (ideal filter).

    The image is mirror-padded to 512x512 first so the FFT does not wrap the
    left/right and top/bottom borders into each other.
    """
    padded = np.pad(np.asarray(image, dtype=np.float64), ((_PAD, _PAD), (_PAD, _PAD), (0, 0)), mode="symmetric")
    mask = ((_RADIUS_PADDED >= lo) & (_RADIUS_PADDED < hi))[..., None]
    filtered = np.real(np.fft.ifft2(np.fft.fft2(padded, axes=(0, 1)) * mask, axes=(0, 1)))
    return filtered[_PAD:_PAD + SIZE, _PAD:_PAD + SIZE].astype(np.float32)


def fuse_with_bicubic(reconstruction, bicubic, lo=0.0625, hi=LR_NYQUIST):
    """Bicubic outside [lo, hi), the reconstruction inside it (the mid band)."""
    return band_filter(bicubic, 0.0, lo) + band_filter(reconstruction, lo, hi) + band_filter(bicubic, hi, np.inf)


def radial_snr_db(reconstruction, reference, num_bins=48):
    """(bin centers, SNR dB per bin) from 0 to 0.5 cycles/pixel, for plotting."""
    signal = _power(reference)
    error = _power(np.asarray(reconstruction, dtype=np.float64) - reference)
    edges = np.linspace(0.0, 0.5, num_bins + 1)
    snr = [
        _snr_db(signal[(RADIUS >= lo) & (RADIUS < hi)], error[(RADIUS >= lo) & (RADIUS < hi)])
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    return (edges[:-1] + edges[1:]) / 2, np.asarray(snr)


def blockiness(image, period=16, phase=4):
    """Tile-border jump ratio for the generator's 16-pixel patch grid; about 1 = no tiling.

    Mean |pixel jump| across the borders between `period`-pixel tiles (between pixels
    16k-1 and 16k, both axes, averaged over channels) divided by the mean |jump| at the
    other positions that are aligned to `phase` pixels (4k-1 | 4k, not on a tile border).
    Comparing only against 4-pixel-aligned positions cancels the period-4 structure that
    x4 bicubic upsampling itself creates, so HR and bicubic both sit near 1.
    """
    x = np.asarray(image, dtype=np.float64)
    assert x.shape == (SIZE, SIZE, 3)
    jumps = np.abs(np.diff(x, axis=1)).mean(axis=(0, 2)) + np.abs(np.diff(x, axis=0)).mean(axis=(1, 2))
    position = np.arange(1, SIZE)  # jumps[k] is between pixels k and k+1, i.e. before pixel k+1
    border = position % period == 0
    control = (position % phase == 0) & ~border
    return float(jumps[border].mean() / jumps[control].mean())
