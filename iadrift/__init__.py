"""IADrift: CSGM- and IAGAN-style inversion of a drifting model (Deng et al., 2026) for super-resolution x4.

Call ``iadrift.runtime.setup`` before importing anything that imports JAX.

    runtime      environment set-up (allocator, upstream code path, Hugging Face cache); no JAX
    paths        data, upstream and cache locations from environment variables; no JAX
    data         ImageNet-V2 loading, 80/20 split, fixed test sample, wrong labels, measurements
    operators    SR x4 operator A, bicubic upsampling, back-projection, LR loss
    generator    loads pixel_B_sota; the generator calls shared by all methods
    csgm         CSGM: optimize the noise input of the frozen generator
    adaptation   IADrift: joint (theta, eps) adaptation; theta-only adaptation
    ensemble     shifted CSGM members and the posterior-mean ensembles
    metrics      PSNR, LPIPS, blockiness, band / radial SNR, patch-grid score
    stats        paired statistics and Holm's correction (docs/PROTOCOL.md, Section 6)
    checkpoint   resumable, hashed experiment checkpoints (JSONL + .npz)
"""

__version__ = "1.0.0"
