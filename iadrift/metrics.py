"""Image-quality metrics, comparing a reconstruction with the HR image; both [256, 256, 3] in [-1, 1].

    psnr            data range 2, unclipped; psnr_of_mean_mse gives IAGAN's averaging convention
    lpips_dist      LPIPS with the AlexNet backbone (lpips 0.1.4), inputs clipped to [-1, 1]
    blockiness      pixel jump across the generator's 16-pixel tile borders relative to other 4-aligned positions
    band_snr_db     SNR in four radial frequency bands, two below the LR limit 0.125 and two above
    grid_score      spectral power on the 16-pixel patch-grid lines relative to neighbouring lines

No JAX import; torch is imported on the first LPIPS call.
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


# ---------------------------------------------------------------- pixel metrics
def psnr(original, reconstructed):
    """PSNR (dB) for images in [-1, 1] (data range 2)."""
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    assert original.shape == reconstructed.shape
    mse = np.mean(((original - reconstructed) / 2.0) ** 2)
    return float("inf") if mse == 0 else float(-10 * np.log10(mse))


def psnr_of_mean_mse(psnrs):
    """PSNR of the MSE averaged over images (IAGAN's reporting convention)."""
    return float(-10 * np.log10(np.mean(10 ** (-np.asarray(psnrs) / 10))))


_LPIPS_NET = None


def lpips_dist(x, ref):
    """LPIPS (AlexNet) between two [256, 256, 3] images in [-1, 1]; lower is better."""
    global _LPIPS_NET
    import torch
    if _LPIPS_NET is None:
        import lpips
        _LPIPS_NET = lpips.LPIPS(net="alex", verbose=False).eval()
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(np.clip(a, -1, 1).transpose(2, 0, 1)[None])).float()
    with torch.no_grad():
        return float(_LPIPS_NET(to_t(x), to_t(ref)))


# ---------------------------------------------------------------- tile seams
def blockiness(image, period=16, phase=4):
    """Tile-border jump ratio for the generator's 16-pixel patch grid; about 1 means no visible tiling.

    Mean |pixel jump| across the borders between `period`-pixel tiles (both axes, averaged over channels) divided by
    the mean |jump| at the other `phase`-aligned positions. Using only 4-aligned positions as the control cancels the
    period-4 pattern of x4 bicubic upsampling, so HR and bicubic images both score about 1.
    """
    x = np.asarray(image, dtype=np.float64)
    assert x.shape == (SIZE, SIZE, 3)
    jumps = np.abs(np.diff(x, axis=1)).mean(axis=(0, 2)) + np.abs(np.diff(x, axis=0)).mean(axis=(1, 2))
    position = np.arange(1, SIZE)  # jumps[k] lies between pixels k and k+1
    border = position % period == 0
    control = (position % phase == 0) & ~border
    return float(jumps[border].mean() / jumps[control].mean())


def grid_score(x):
    """Spectral power on the patch-grid lines k/16 (k = 1..7, both axes) over power on k/16 +- 2/256; ~1 = no grid."""
    x = np.asarray(x, np.float64)
    p = np.sum(np.abs(np.fft.fft2((x - x.mean(axis=(0, 1))) * WINDOW, axes=(0, 1))) ** 2, axis=-1)
    on = np.mean([p[:, 16 * k].mean() + p[16 * k, :].mean() for k in range(1, 8)])
    off = np.mean([p[:, 16 * k + s].mean() + p[16 * k + s, :].mean() for k in range(1, 8) for s in (-2, 2)])
    return float(on / off)


# ---------------------------------------------------------------- frequency-resolved error
def _power(image):
    """|FFT|^2 of a Hann-windowed [256, 256, 3] array, summed over channels."""
    image = np.asarray(image, dtype=np.float64)
    assert image.shape == (SIZE, SIZE, 3)
    return np.sum(np.abs(np.fft.fft2(image * WINDOW, axes=(0, 1))) ** 2, axis=-1)


def _snr_db(signal, error):
    return float(10 * np.log10(signal.sum() / error.sum()))


def band_snr_db(reconstruction, reference):
    """{band: SNR in dB} for the four BANDS. 0 dB is as good as leaving the band empty."""
    signal = _power(reference)
    error = _power(np.asarray(reconstruction, dtype=np.float64) - reference)
    return {name: _snr_db(signal[(RADIUS >= lo) & (RADIUS < hi)], error[(RADIUS >= lo) & (RADIUS < hi)])
            for name, lo, hi in BANDS}


_PAD = SIZE // 2
_fp = np.fft.fftfreq(SIZE + 2 * _PAD)
_RADIUS_PADDED = np.sqrt(_fp[None, :] ** 2 + _fp[:, None] ** 2)


def band_filter(image, lo, hi):
    """Ideal radial band-pass keeping [lo, hi) cycles/pixel; the image is mirror-padded to 512x512 first."""
    padded = np.pad(np.asarray(image, dtype=np.float64), ((_PAD, _PAD), (_PAD, _PAD), (0, 0)), mode="symmetric")
    mask = ((_RADIUS_PADDED >= lo) & (_RADIUS_PADDED < hi))[..., None]
    filtered = np.real(np.fft.ifft2(np.fft.fft2(padded, axes=(0, 1)) * mask, axes=(0, 1)))
    return filtered[_PAD:_PAD + SIZE, _PAD:_PAD + SIZE].astype(np.float32)


def evaluate(x, hr, y=None, with_lpips=True):
    """The metric dict stored for each reconstruction in the experiment caches."""
    from .operators import lr_loss  # imports JAX
    m = {"psnr": psnr(hr, x), "psnr_clipped": psnr(hr, np.clip(x, -1, 1)), "band_snr_db": band_snr_db(x, hr),
         "blockiness": blockiness(x)}
    if with_lpips:
        m["lpips"] = lpips_dist(x, hr)
    if y is not None:
        m["lr_loss"] = lr_loss(x, y)
    return m
