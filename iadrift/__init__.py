"""IADrift: CSGM- and IAGAN-style inversion of a drifting model (Deng et al., 2026) for super-resolution x4.

Package layout (import order matters: call ``iadrift.runtime.setup`` before anything that imports JAX):

    runtime      environment set-up (allocator, upstream code path, Hugging Face cache); no JAX import
    paths        data / upstream / cache locations, from environment variables; no JAX import
    data         ImageNet-V2 loading, the 80/20 split, the fixed test sample, wrong labels, measurements
    operators    the SR x4 operator A, bicubic upsampling, back-projection, LR loss
    generator    loading pixel_B_sota and the generator calls used by every method
    csgm         CSGM: invert the frozen generator's noise input (the frozen benchmark loop and a generic loop)
    adaptation   IADrift: joint (theta, eps) adaptation, and theta-only adaptation
    ensemble     shifted CSGM members and the (weighted) posterior-mean ensemble
    metrics      PSNR, LPIPS, blockiness, band / radial SNR, patch-grid score
    stats        paired statistics and Holm's correction (the analysis of docs/PROTOCOL.md, Section 6)
    checkpoint   resumable, hashed, locked experiment checkpoints (JSONL + .npz)
"""

__version__ = "1.0.0"
