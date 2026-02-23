# Synthetic aperture imaging through foreground obstacles: a comprehensive algorithm survey

**A 4×4 camera array with 120–150 mm effective aperture can suppress foreground fences with remarkable effectiveness, but the choice of compositing algorithm determines whether residual ghosting, background blur, or computational cost dominates.** The core principle is simple: when 16 cameras are shifted to align background content, foreground objects scatter across the frame due to parallax, becoming statistical outliers that robust estimators can reject. This survey covers the full algorithmic landscape—from classical shift-and-add methods through robust statistical compositing to neural radiance fields and 3D Gaussian splatting—evaluating each against the specific geometry of closely-spaced cameras imaging through near-field obstructions. The system under consideration (Raspberry Pi 4B + IMX296 global shutter monochrome sensors, **1440×1080 resolution at 3.45 µm pixel pitch**, 40–50 mm inter-camera spacing) creates a scenario where foreground disparity exceeds **1000 pixels** between adjacent cameras while background disparity remains around 130 pixels—a ratio that strongly favors parallax-based separation.

---

## Optical geometry of the 4×4 array establishes the physical basis

Before examining algorithms, the system's optical geometry determines what is theoretically achievable. The 4×4 grid with 40–50 mm spacing yields a total synthetic aperture of **120–150 mm per axis** (three inter-camera gaps), with diagonal extent of approximately 210 mm. With a typical f ≈ 6 mm lens on the IMX296's 4.97 × 3.73 mm sensor, the disparity between adjacent cameras (B = 45 mm baseline) at different depths reveals the separation power:

| Scene element | Depth z | Disparity per 45 mm baseline | Disparity in pixels |
|---|---|---|---|
| Fence (near) | 75 mm | 3.60 mm | **1043 px** |
| Fence (far) | 100 mm | 2.70 mm | 783 px |
| Background (near) | 500 mm | 0.54 mm | 157 px |
| Background (mid) | 600 mm | 0.45 mm | 130 px |
| Background (far) | 700 mm | 0.39 mm | 112 px |

The foreground-to-background disparity ratio of roughly **8:1** is the foundation for all subsequent algorithms. For the maximum baseline in the array (corner-to-corner = 135 mm), foreground disparity reaches approximately 3130 pixels—exceeding the sensor width itself. This means extreme cameras see entirely different segments of fence wire, guaranteeing that most background pixels are visible in at least some subset of views.

The synthetic aperture's **depth of field** at the 600 mm focus plane is extraordinarily shallow: DOF ≈ 2z²c/(fD) ≈ 2 × 600² × 0.00345/(6 × 150) ≈ **2.76 mm**, compared to ~197 mm for a single f/2.8 lens. This razor-thin DOF means objects even slightly in front of or behind the focus plane are strongly blurred in the composite, which is precisely the mechanism that suppresses foreground obstacles.

The **circle of confusion** for a fence wire at 75 mm when focused at 600 mm corresponds to a blur spread of 40–100+ pixels across the composite (depending on the effective aperture used), sufficient to render thin wires invisible against the sharpened background.

---

## Classical shift-and-add and light field refocusing

### The shift-and-add integral

The fundamental synthetic aperture imaging operation aligns N camera images to a target depth plane and averages them. Given cameras at positions (s_i, t_i) with images I_i, the refocused image at depth parameter α is:

**I_α(u′, v′) = (1/N) Σᵢ I_i(u′ + (1−α)(s_i − s₀), v′ + (1−α)(t_i − t₀))**

where (s₀, t₀) is the reference camera position and α = z_ref/z_target controls the focus depth. Objects at the target depth align coherently and appear sharp; objects at other depths are displaced differently in each view and blur in the average. This formulation is mathematically identical to the "photography operator" in Ren Ng's Fourier Slice Photography framework, which showed that refocusing in the Fourier domain is a 2D slice through the 4D light field spectrum.

Vaish et al. introduced the **shear-warp factorization** that makes this computationally efficient: the mapping from each camera to the reference frame factors into an initial homography H_{i,ref} (applied once per camera during preprocessing) and a depth-dependent shift G_i(z) that reduces to a pure translation for fronto-parallel planes. This means refocusing through different depths requires only shifts and additions—trivially parallelizable on a GPU.

The principal limitation is **ghosting**: with only 16 cameras (a sparse aperture), out-of-focus foreground objects produce 16 discrete ghost images rather than the smooth blur of a continuous aperture. For a fence wire, this manifests as 16 faint, spatially separated wire copies superimposed on the background. Simple averaging preserves the total foreground energy—it is merely redistributed. This motivates the robust compositing methods discussed below. Vaish's Stanford thesis (2007) provides the definitive treatment of SAI geometry for camera arrays, building on the Stanford 100-camera array work by Wilburn et al. (SIGGRAPH 2005) that demonstrated real-time synthetic aperture focusing and seeing through foliage.

### 4D light field parameterization and EPI analysis

The two-plane parameterization L(s, t, u, v) introduced by Levoy and Hanrahan (SIGGRAPH 1996) and Gortler et al. ("The Lumigraph," SIGGRAPH 1996) provides the theoretical framework for understanding multi-camera refocusing. The (s, t) coordinates index camera position; (u, v) index pixel position. Digital refocusing is integration over the aperture plane, and the refocusing parameter α selects the depth of the focal plane.

**Epipolar Plane Images (EPIs)**, introduced by Bolles, Baker, and Marimont (IJCV 1987), offer an elegant visualization: fixing one spatial coordinate and one angular coordinate produces a 2D slice where scene points trace lines whose slopes encode depth. Closer objects produce steeper slopes; background objects produce shallower ones. For the 4×4 array, horizontal EPIs contain only 4 samples across the angular dimension—too sparse for reliable slope estimation via the structure tensor methods of Wanner and Goldluecke (CVPR 2012, TPAMI 2013), which require ≥9 angular samples per direction. However, EPIs remain useful for qualitative depth-layer visualization and for CNN-based depth estimation (EPINET) that can learn from sparse angular data.

### Focal stack synthesis and depth from defocus

A focal stack—a set of SAI images focused at different depths—enables all-in-focus compositing by selecting the sharpest value at each pixel: d*(x,y) = argmax_k Sharpness(S_{d_k}, x, y), using sharpness measures such as Laplacian variance or gradient magnitude. However, Vaish et al. (CVPR 2006) proved an important theoretical result: **shape-from-focus requires 3rd-order spatial intensity gradients**, compared to only 1st-order gradients for stereo matching. This means focus-based depth estimation fails on smooth or low-texture regions, a significant limitation for scenes viewed through fences where background texture may be limited.

---

## Foreground occlusion removal through robust compositing

This section addresses the core algorithmic challenge: given 16 images aligned to the background plane, how to reject foreground fence pixels and recover a clean background. The methods range from simple median filtering to sophisticated energy-minimization frameworks.

### Median and trimmed-mean compositing

The most practical and effective baseline method exploits the statistical nature of foreground contamination. After warping all 16 views to align the background, each pixel position has 16 intensity samples. Background pixels agree (low variance); fence pixels, displaced by parallax, appear at different positions in each view and contaminate only a subset of the 16 samples at any given pixel. **The per-pixel median across views rejects foreground as statistical outliers.**

For a fence occluding fraction α of each view, approximately α × 16 samples per pixel are contaminated. The median selects the central value, succeeding when fewer than 8 of 16 views show fence at that pixel. Quantitatively: at **30% fence coverage**, ~5 of 16 views are contaminated—the median robustly selects from the 11 clean samples. At **45% coverage**, ~7 contaminated—still works (9 clean vs. 7 contaminated). At **50%**, the method reaches its breakdown point.

The **trimmed mean** (discarding the k highest and k lowest values before averaging) offers a tunable alternative. With k = 3, the trimmed mean averages the central 10 of 16 values, tolerating up to ~19% contamination from each tail. For dark fence wires against a brighter background, the optimal **percentile filter** selects slightly above the contaminated fraction: for 30% dark-fence coverage, choosing the 35th percentile avoids the fence-contaminated lower values.

Implementation is trivial in NumPy: `np.median(aligned_stack, axis=0)` processes the entire 1440×1080×16 stack in ~100–400 ms on CPU, ~10–80 ms on GPU.

### Entropy-based depth and compositing

Vaish, Szeliski, Zitnick, Kang, and Levoy (CVPR 2006)—the single most important paper for this application—compared four cost functions for reconstructing occluded surfaces. **Shannon entropy** of the pixel intensity histogram across aligned views proved most robust:

**H(x, y, d) = −Σ_k (b_k/N) log(b_k/N)**

where b_k counts the 16 intensity values falling in bin k (K = 16 bins for 8-bit data). At the correct background depth, unoccluded rays converge to a narrow intensity range (peaked histogram, low entropy); foreground-contaminated values scatter across bins (high entropy) but, critically, do not increase the modal peak. Unlike variance (which is inflated by outliers), entropy penalizes dispersion without proportionally weighting extreme values—making it inherently robust to foreground contamination.

Vaish et al. demonstrated near-perfect background reconstruction up to **65% occlusion** with 81 cameras, and effective performance up to ~45–50% occlusion with 16 cameras. Entropy outperformed stereo (variance), focus (sharpness), and median measures in heavily occluded scenarios.

### Robust M-estimators: Huber, Tukey biweight, and IRLS

Formulating background estimation as a robust regression problem—finding μ* = argmin_μ Σᵢ ρ(I_i − μ) for robust loss ρ—provides theoretically grounded outlier rejection. **Tukey's biweight** is particularly suited to fence removal because observations with residuals exceeding threshold c receive exactly zero weight, completely eliminating foreground contributions:

- ρ_Tukey(r) = (c²/6)[1 − (1 − (r/c)²)³] for |r| ≤ c; c²/6 otherwise

The scale parameter is estimated via the **Median Absolute Deviation**: σ̂ = 1.4826 × median(|I_i − median(I)|), which has a **50% breakdown point**. The Iteratively Reweighted Least Squares (IRLS) algorithm converges in 5–10 iterations: initialize with the median, compute residuals, assign weights via the influence function ψ = ρ′, update the weighted mean, and repeat. For 16 views at 1440×1080, IRLS requires approximately 2 seconds on a GPU.

The Huber estimator provides a smoother transition between quadratic (inlier) and linear (outlier) behavior, appropriate when the fence-background intensity difference is moderate. In practice, Tukey's complete rejection of large residuals makes it preferable for high-contrast fence wires.

### Visibility map estimation and iterative refinement

More sophisticated approaches explicitly estimate which pixels in each view are occluded by the foreground, constructing binary or continuous visibility masks M_i(x, y). **Wilburn et al. (SIGGRAPH 2005)** introduced the matted synthetic aperture: I_matted = Σᵢ M_i × I_{i,d} / Σᵢ M_i, averaging only unoccluded rays. Pei, Chen, and Yang (IEEE TCSVT 2018) used image matting techniques—first creating an SAI focused on the foreground to detect fence pixels, then using these as scribbles for alpha matte estimation via the closed-form matting solution of Levin, Lischinski, and Weiss (TPAMI 2008).

**Pei, Zhang, Chen, and Yang (Pattern Recognition 2013)** formulated visibility as an MRF labeling problem solved via graph cuts: E(L) = Σ_p D_p(l_p) + λ Σ_{(p,q)} V_{pq}(l_p, l_q), where the data term D uses SAI variance and the smoothness term V enforces spatial coherence. This approach achieves clean separation but requires the foreground depth range as prior input.

An effective practical pipeline combines these approaches iteratively: (1) compute median composite as initial background B₀, (2) detect foreground in each view as |I_i − B₀| > threshold, (3) create visibility masks, (4) recompute background as the visibility-weighted average, and (5) iterate until convergence. This typically stabilizes in 3–5 iterations and handles up to ~40–45% fence coverage.

---

## Registration and homographic alignment are the critical preprocessing step

### Feature matching and homography estimation

All compositing methods depend on accurate inter-view alignment. For the 4×4 array, the depth-dependent homography mapping camera i to the reference view at depth z is:

**H_i(z) = K_ref · (R_i + t_i · nᵀ/z) · K_i⁻¹**

where K is the camera intrinsic matrix, R_i and t_i are the relative rotation and translation, and n = [0,0,1]ᵀ for fronto-parallel planes. For a well-calibrated near-coplanar array where R_i ≈ I, this simplifies to a depth-dependent translation—exactly the "shift" in shift-and-add.

Feature-based registration uses **SIFT** (Lowe, IJCV 2004) for highest accuracy (~0.3 pixel keypoint localization), **ORB** (Rublee et al., ICCV 2011) for speed (10–100× faster, ~10–50 ms per 1440×1080 image), or **AKAZE** (Alcantarilla et al., BMVC 2013) for the best balance—its nonlinear diffusion scale space preserves the fine edge structures of fence wires that provide distinctive matching features. Repeating fence patterns create ambiguous matches; mitigation requires strict ratio-test thresholds (~0.6) and RANSAC with geometric verification.

**Sub-pixel accuracy is essential**: registration error should be ≤0.1 pixel for sharp SAI results. The recommended pipeline is ORB/AKAZE for coarse matching → RANSAC homography estimation → **ECC (Enhanced Correlation Coefficient)** refinement. The ECC method of Evangelidis and Psarakis (IEEE TPAMI 2008) maximizes a photometric-invariant correlation coefficient, achieving ~0.05 pixel accuracy and inherent robustness to inter-camera brightness differences. OpenCV provides `findTransformECC()` supporting full homography motion models. Phase correlation offers an alternative for initial translation estimation, with sub-pixel accuracy of ~0.02 pixels via the Foroosh et al. (IEEE TIP 2002) polyphase method.

### Plane-sweep stereo and its dual role

Collins's space-sweep algorithm (CVPR 1996) sweeps a virtual plane through the depth range, warping all views via H_i(z_d) at each depth hypothesis and computing photometric matching costs. The resulting cost volume (1440 × 1080 × D) simultaneously provides depth estimation and a set of refocused images—**each depth slice IS a synthetic aperture image focused at that depth**. Cost functions include SSD (simplest), NCC (robust to brightness variation), and the Census transform (robust to monotonic illumination changes, preferred for structured scenes). At 1440×1080 resolution with D = 100 depth planes, the cost volume occupies ~622 MB—manageable in modern GPU memory—and computes in **0.5–2 seconds** on an RTX-class GPU.

### Calibration requirements

Zhang's flexible calibration method (IEEE TPAMI 2000) using a planar checkerboard provides intrinsic parameters via OpenCV's `calibrateCamera()`. For 16 cameras, pairwise stereo calibration followed by global bundle adjustment achieves typical reprojection errors of 0.1–0.3 pixels. A large checkerboard (300+ mm) should be imaged at 15–20 poses within the 500–700 mm operating range. Calibration errors propagate directly to alignment errors; with careful calibration (~0.2 pixel reprojection error) plus ECC refinement (~0.05 pixel), total alignment error of ~0.1 pixel is achievable.

---

## Neural and learning-based approaches offer higher quality at computational cost

### NeRF and its occlusion-aware variants

The original Neural Radiance Field (Mildenhall et al., ECCV 2020) represents scenes as continuous functions F_θ(x, y, z, θ, φ) → (c, σ) mapping 3D position and viewing direction to color and density, rendered via volume integration. While NeRF itself does not inherently separate foreground from background, several variants directly address obstacle removal.

**OCC-NeRF** (Zhu et al., CVPR 2023) is the most directly relevant paper for this system. It specifically targets occlusion removal from multi-view images using depth-constrained selective supervision: the volume rendering integral is split into foreground and background ranges, and only the background contribution is supervised. The foreground density is suppressed during training by excluding near-range samples. This requires no external occlusion-free supervision and jointly optimizes camera parameters.

**DeclutterNeRF** (arXiv 2025) substantially improves on OCC-NeRF, achieving PSNR of **23.71 dB on chain fence scenes** compared to OCC-NeRF's 11.90 dB. Its key innovation is the S3IM (Stochastic Structural Similarity) loss that provides global structural supervision, plus joint multi-view camera parameter optimization. **NeRF in the Wild** (Martin-Brualla et al., CVPR 2021) introduces per-image transient embeddings that can capture view-dependent fence appearance, though the fence is persistent rather than truly transient.

**RobustNeRF** (Sabour et al., CVPR 2023) uses trimmed robust estimation to ignore distractors, applying iteratively reweighted least squares with a smoothness prior on the outlier process. However, it was designed for objects absent from some views entirely, rather than parallax-shifted persistent structures. **Instant-NGP** (Müller et al., SIGGRAPH 2022) reduces training from hours to minutes via multi-resolution hash encoding, making per-scene optimization practical.

With only 16 views, few-shot NeRF variants become relevant: **pixelNeRF** (Yu et al., CVPR 2021) conditions on CNN-extracted image features for cross-scene generalization, while **RegNeRF** (Niemeyer et al., CVPR 2022) regularizes unobserved viewpoints. Sixteen well-distributed views substantially exceed the extreme few-shot setting (1–3 views), and most NeRF variants produce reasonable results at this count.

### Multiplane image decomposition

The MPI representation—a set of D fronto-parallel RGBA planes at discrete depths—is arguably the most natural fit for the fence removal problem. Each plane stores color C_d and opacity α_d, composed via back-to-front alpha blending:

**C(p) = Σ_d T_d · α_d · C_d, where T_d = Π_{d′<d} (1 − α_{d′})**

Foreground obstacles occupy the near-depth planes; background occupies the far planes. **Simply zeroing the alpha of near-depth planes and re-rendering yields a clean background**—this is the most intuitive and direct approach.

**Stereo Magnification** (Zhou et al., SIGGRAPH 2018) predicts MPIs from stereo pairs, while **LLFF (Local Light Field Fusion)** (Mildenhall et al., SIGGRAPH 2019) expands each sampled view into a local MPI and blends adjacent local light fields. LLFF derives plenoptic sampling bounds showing that 16 views with 40–50 mm spacing at the relevant depth range fall well within practical sampling requirements. MPI inference runs in ~100 ms per view on a GPU—**1.6 seconds total for all 16 views**—followed by real-time rendering via homography warping and alpha compositing. Pre-trained models are available and adaptable to monochrome input with minor architectural changes.

### 3D Gaussian splatting

Kerbl et al. (SIGGRAPH 2023) represent scenes as collections of 3D Gaussians, each with position, covariance, opacity, and spherical harmonic color coefficients, rendered via differentiable rasterization at **100+ FPS**. Training takes 5–30 minutes on a modern GPU. For obstacle removal, the strategy is: reconstruct the scene as Gaussians, filter by depth to remove those at z < 100 mm, and re-render the background only.

Challenges include COLMAP initialization struggling with 16 fenced images (fence features contaminate structure-from-motion), and insufficient Gaussian density in heavily occluded background regions. **WeatherGS** (arXiv 2024) demonstrates an effective preprocessing paradigm: detect obstruction masks, exclude masked areas from training data, then train 3DGS on the remaining clean observations. This "detect → mask → train" approach maps directly to fence removal.

---

## Application-specific fence removal literature spans two decades

### Foundational single-image and video methods

The de-fencing problem was formalized by **Liu, Belkina, Hays, and Lublinerman (CVPR 2008)**, who introduced three-phase processing: automatic lattice detection of the fence skeleton as a deformed near-regular texture, foreground/background separation using translational symmetry as the primary cue, and occluded region inpainting. This handled up to 53% occlusion. **Park, Brocklehurst, Collins, and Liu (ACCV 2010)** extended this with improved deformable lattice detection via mean-shift belief propagation and introduced multi-view inpainting using parallax from additional frames.

**Yi, Wang, and Tan (CVPR 2016)** achieved fully automatic video fence segmentation through optical flow clustering and graph-cut optimization, exploiting the observation that fence pixels exhibit peaked gradient orientation distributions. **Mu, Liu, and Yan (IEEE TCSVT 2014)** exploited parallax-aware alignment: when aligning backgrounds across frames, the fence fails to align, enabling detection and removal.

### Multi-frame methods directly applicable to camera arrays

The most relevant work for the 4×4 array is **Xue, Rubinstein, Liu, and Freeman, "A Computational Approach for Obstruction-Free Photography" (SIGGRAPH 2015)**. This captures a short image sequence while slightly moving the camera, exploits motion parallax to separate foreground and background layers via edge-based optical flow, and performs iterative coarse-to-fine layer decomposition. The slight camera motion is directly analogous to the different viewpoints in the 4×4 array—each camera replaces a temporal frame.

**Liu, Lai, Yang, Chuang, and Huang, "Learning to See Through Obstructions" (CVPR 2020)** advanced this with deep learning: alternating between optical flow estimation for each layer and CNN-based layer reconstruction, trained on synthetic data that transfers well to real images. It processes 5 input frames and produces state-of-the-art results (PSNR ~30–35 dB on synthetic data, SSIM > 0.95). **Tsogkas, Zhang, Jepson, and Levinshtein (WACV 2023)**, from Samsung AI, presented the most recent practical multi-frame method, achieving real-time de-fencing from 5-frame bursts using simplified flow estimation and lightweight CNNs.

**Jonna, Satapathy, and Sahay (ICASSP 2017)** directly addressed stereo de-fencing, exploiting disparity for fence/background separation—the closest prior work to the camera array configuration. Deep learning approaches include **Du et al. (ICME 2018)** with FCN-based fence segmentation plus occlusion-aware flow, **Matsui and Ikehara (IEEE Access 2020)** with end-to-end U-Net fence detection and ResNet inpainting, and GAN-based methods achieving **16 ms inference** for single-image de-fencing.

### Frequency-domain and blind deconvolution approaches

Regular fence patterns create peaks in the 2D Fourier transform at the fence's fundamental spatial frequency and harmonics. **Notch filtering** can suppress these peaks, but risks destroying background content at the same frequencies—a fundamental limitation. Adaptive Gaussian notch filters (Aizenberg and Butakoff, Image and Vision Computing 2008) automatically detect noise spike positions, but cannot distinguish fence frequency components from background texture at overlapping frequencies. These methods are most useful as supplementary post-processing for removing residual periodic artifacts after multi-view compositing.

The blind deconvolution formulation I = α·F + (1−α)·B treats fence removal as joint estimation of background B, fence pattern F, and alpha matte α. MAP estimation with alternating minimization (estimate B given F, estimate F given B) using total variation and sparsity (ℓ₁) priors can be solved via FISTA or Split Bregman methods. Jonna et al. (JOSA A 2016) formalized this optimization across multiple frames with TV regularization.

---

## Evaluation metrics should separate occluded from non-occluded regions

### Standard reference-based metrics

**PSNR** (10·log₁₀(MAX²/MSE)) is universally reported but correlates poorly with perceptual quality. For fence removal, PSNR values of **28–35 dB** indicate good reconstruction. **SSIM** (Wang et al., IEEE TIP 2004) captures structural fidelity through luminance, contrast, and structure comparisons computed in sliding windows: values above **0.90** indicate good preservation, above **0.95** excellent. Multi-scale SSIM (MS-SSIM) evaluates across scales, better capturing both fine detail and global structure.

**LPIPS** (Zhang et al., CVPR 2018) measures perceptual distance using deep network feature differences: LPIPS = Σ_l w_l · ‖φ_l(x) − φ_l(y)‖², where φ_l are normalized activations from a pre-trained VGG or AlexNet. Lower values indicate greater similarity. LPIPS correlates substantially better with human judgment than pixel-level metrics and is **the single most appropriate metric** for fence removal evaluation, because ghosting and blurring artifacts are perceptually salient but may not dominate MSE.

### Fence-removal-specific evaluation

The recommended protocol evaluates metrics separately for previously-occluded regions (testing reconstruction/inpainting quality) and non-occluded regions (testing background preservation). The **foreground suppression ratio** FSR = E_fence_before/E_fence_after measures how much fence energy is reduced, computed via gradient magnitude or edge response in fence regions. Ground truth can be obtained by capturing scenes with and without the fence (physically removing it) or by synthetic overlay of rendered fence patterns on clean backgrounds, as used by Liu et al. (CVPR 2020) and Tsogkas et al. (WACV 2023). For no-reference evaluation of real captures, NIQE (Mittal et al., IEEE SPL 2013) and BRISQUE (Mittal et al., IEEE TIP 2012) assess naturalness without ground truth.

---

## A recommended processing pipeline and the feasibility frontier

For the 4×4 IMX296 array, the following pipeline balances quality and speed:

- **Offline calibration** (once): Zhang's method per camera, pairwise stereo calibration, global bundle adjustment. ~30 minutes of human effort.
- **Capture**: Hardware-triggered simultaneous acquisition. 16 × 1440×1080 × 10-bit = ~50 MB per capture.
- **Undistort and warp**: Apply lens distortion correction and depth-dependent homographies H_i(z_target) to align all views at background depth. ~50 ms on GPU.
- **Robust composite**: Pixel-wise Tukey biweight M-estimation across 16 aligned views (5 IRLS iterations). ~200 ms on GPU. Alternative: median for simplicity (~80 ms) or entropy for maximum robustness (~500 ms).
- **Post-processing**: Optional spatial regularization (bilateral filter) to fill residual holes. ~20 ms.
- **Total latency: ~300–800 ms on a GPU**, achieving near-real-time operation at 1440×1080.

For higher quality with offline processing, MPI decomposition (LLFF) provides natural layer separation in ~2 seconds total, while DeclutterNeRF achieves the best reported results on chain fence scenes (PSNR 23.71) but requires 10–30 minutes of per-scene optimization. 3D Gaussian splatting offers a middle ground: 5–30 minutes training with real-time rendering thereafter.

| Method | Latency (GPU) | Max fence coverage | Quality ceiling |
|---|---|---|---|
| Mean SAI | 50 ms | ~15% (visible ghosting) | Low |
| Median composite | 80 ms | ~50% | Good |
| Tukey M-estimator | 200 ms | ~50% | Very good |
| Entropy + spatial prior | 500 ms | ~50% | Very good |
| Visibility-weighted SAI | 1–3 s | ~60% | Excellent |
| MPI (LLFF) | 2 s | ~70% (depth-based) | Excellent |
| DeclutterNeRF | 10–30 min | ~80% | Highest |
| 3D Gaussian Splatting | 5–30 min | ~70% | Excellent |

The critical tradeoff: robust statistical methods (median, Tukey) are **100–1000× faster** than neural methods and require no training data or GPU-intensive optimization, making them the clear choice for real-time or near-real-time operation. Neural methods provide superior quality for offline processing, especially in heavily occluded scenarios where the fence coverage approaches or exceeds 50%.

---

## Conclusion: key insights and the algorithmic frontier

The 4×4 camera array at 40–50 mm spacing creates an optical geometry strongly favorable to foreground rejection: the 8:1 disparity ratio between fence and background ensures that robust compositing has ample statistical signal for separation. **Three findings emerge from this survey that may not be obvious a priori.** First, simple median compositing of 16 aligned views handles up to ~45% fence coverage with negligible computational cost—beating far more sophisticated algorithms in the speed-quality Pareto frontier for moderate occlusion. Second, the entropy cost function of Vaish et al. (CVPR 2006) remains the strongest classical method 20 years after publication, with no pixel-level compositing method demonstrably surpassing it under equivalent occlusion conditions. Third, MPI decomposition offers the most architecturally natural fit for this problem: the discrete depth-plane representation maps exactly to the foreground/background separation task, with the additional advantage that pre-trained LLFF models exist and inference is fast.

The sensor correction (IMX296 is **1440×1080**, not 4096×3000) substantially relaxes computational requirements—16 images total only ~25–50 MB, and the full plane-sweep cost volume fits in under 1 GB of GPU memory. This makes even the plane-sweep approach feasible in seconds rather than minutes. The most productive area for future development is likely **combining robust statistical compositing for speed with neural refinement for quality**: using median or Tukey compositing as a fast initial estimate, then applying a lightweight CNN for residual artifact removal and background inpainting in regions where all 16 views were occluded—a hybrid approach not yet well-explored in the literature but directly motivated by the system's specific geometry and constraints.