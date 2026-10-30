"""
p26lib.py -- shared code for CSE 681 Project 26:
"Compact Deep Feature Descriptors vs. Classical SIFT/ORB"

Every notebook imports this one file, so all three students use the SAME data loader,
the SAME matcher and the SAME metrics. That is what makes the comparison fair.

Ownership (who maintains which section -- everyone must understand all of it):
    Section 1  Config & utilities ............ everyone
    Section 2  HPatches data ................. Student A (Data & Baseline)
    Section 3  Feature extractors ............ Student A (classical) / Student B (learned)
    Section 4  Patch extraction .............. Student B
    Section 5  Compact CNN descriptor ........ Student B (Model)
    Section 6  Training data + training loop . Student B
    Section 7  Matching + metrics + timing ... Student C (Evaluation)
    Section 8  Benchmark runner + reporting .. Student C
    Section 9  Registration demo ............. Student C
"""
from __future__ import annotations

import glob
import math
import os
import platform
import random
import tarfile
import time
import urllib.request
import zipfile

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================================
# 1. CONFIG & UTILITIES
# =====================================================================================
SEED = 42
HPATCHES_URL = "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz"
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"

# 8 high-resolution sequences excluded by the standard D2-Net HPatches protocol
# (Dusmanu et al., CVPR 2019) -> 108 sequences = 52 illumination + 56 viewpoint.
D2NET_EXCLUDED = [
    "i_contruction", "i_crownnight", "i_dc", "i_pencils", "i_whitebuilding",
    "v_artisans", "v_astronautis", "v_talent",
]
MMA_THRESHOLDS = list(range(1, 11))      # pixels
HOMOG_THRESHOLDS = [1, 3, 5]             # pixels (mean corner error)
RANSAC_THRESHOLD = 3.0                   # pixels
PATCH_SIZE = 32                          # CNN input patch side (pixels)
PATCH_MAG = 6.0                          # patch side = PATCH_MAG * cv2.KeyPoint.size (same window as SIFT)


# Evaluation protocols.
#   MAIN       -> our project protocol (full resolution, D2-Net 108 sequences, 2000 kp)
#   SUPERPOINT -> setup of DeTone et al. 2018 (Table 3), used ONLY for the reproduction check:
#                 images resized to 480x640, 1000 keypoints, all 116 sequences, OpenCV default
#                 RANSAC (3 px, 2000 iters, 0.995 conf), H-acc = mean corner error <= eps.
#                 (Verify these settings against Section 7 of the paper before citing.)
PROTOCOLS = {
    "main": {"resize": None, "n_features": 2000, "exclude": D2NET_EXCLUDED,
             "ransac_thresh": 3.0, "ransac_iters": 5000, "ransac_conf": 0.999},
    "superpoint": {"resize": (480, 640), "n_features": 1000, "exclude": [],
                   "ransac_thresh": 3.0, "ransac_iters": 2000, "ransac_conf": 0.995},
}


def set_seed(seed: int = SEED):
    """Fix every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_threads(n: int = 1):
    """Use the same number of CPU threads for OpenCV and PyTorch (fair timing)."""
    cv2.setNumThreads(n)
    torch.set_num_threads(n)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_dirs(proj: str) -> dict:
    """Create the shared folder layout on Google Drive and return the paths."""
    d = {k: os.path.join(proj, k) for k in ["results", "tables", "figures", "checkpoints"]}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    d["root"] = proj
    return d


def env_info() -> dict:
    """Hardware/software description to paste into the Feasibility / Setup sections."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    return info


# =====================================================================================
# 2. HPATCHES DATA  (Student A)
# =====================================================================================
def _download(url: str, dst: str):
    print(f"Downloading {url}\n  -> {dst}")

    def hook(count, block, total):
        if total > 0 and count % 500 == 0:
            print(f"  {100 * count * block / total:5.1f}%", end="\r")

    urllib.request.urlretrieve(url, dst, hook)
    print("  done.            ")


# Where the team keeps its own copy of the HPatches archive on the shared Drive folder.
DRIVE_DATA_DIR = "/content/drive/MyDrive/DIP_P26/data"
_ARCHIVE_EXT = (".zip", ".tar.gz", ".tgz", ".tar")


def _find_hpatches_root(base: str, min_seqs: int = 10):
    """Folder under `base` that directly contains the i_*/v_* sequence folders (any zip layout)."""
    if not os.path.isdir(base):
        return None
    best, best_n = None, 0
    for cur, dirs, _ in os.walk(base):
        if "__MACOSX" in cur or cur[len(base):].count(os.sep) > 3:
            dirs[:] = []
            continue
        n = sum(1 for d in dirs if d[:2] in ("i_", "v_"))
        if n > best_n:
            best, best_n = cur, n
    return best if best_n >= min_seqs else None


def _extract(archive: str, dst: str):
    print(f"Extracting {os.path.basename(archive)} -> {dst} ...")
    if archive.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dst)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dst)


def download_hpatches(data_dir: str = "/content/data", search_dirs=(DRIVE_DATA_DIR,)) -> str:
    """Get HPatches *sequences* (full images + GT homographies) onto the local disk.

    Order of preference:
      1. already extracted in `data_dir`                -> use it
      2. an archive (.zip / .tar.gz) with "hpatches" in its name in `data_dir` or in the
         shared Drive folder DIP_P26/data/              -> copy to local disk + extract
      3. download from the official server
    Returns the folder that contains the sequence folders (i_* and v_*).
    Extracting on the local Colab disk (not on Drive) keeps image loading fast.
    """
    import shutil
    os.makedirs(data_dir, exist_ok=True)
    extract_dir = os.path.join(data_dir, "hpatches")
    root = _find_hpatches_root(extract_dir) or _find_hpatches_root(data_dir)
    if root:
        return root
    archives = []
    for d in [data_dir, *search_dirs]:
        if d and os.path.isdir(d):
            archives += sorted(os.path.join(d, f) for f in os.listdir(d)
                               if "hpatches" in f.lower() and f.lower().endswith(_ARCHIVE_EXT))
    if archives:
        src = archives[0]
        local = os.path.join(data_dir, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(local):
            print(f"Copying {src} -> local disk ...")
            shutil.copy(src, local)
        _extract(local, extract_dir)
    else:
        local = os.path.join(data_dir, "hpatches-sequences-release.tar.gz")
        _download(HPATCHES_URL, local)
        _extract(local, extract_dir)
    root = _find_hpatches_root(extract_dir)
    if root is None:
        raise RuntimeError(f"No i_*/v_* sequence folders found inside {local}. "
                           "Is this the HPatches *sequences* release (not the patches release)?")
    print(f"HPatches ready: {root} ({len(glob.glob(os.path.join(root, '[iv]_*')))} sequences)")
    return root


def list_sequences(root: str, exclude=D2NET_EXCLUDED, max_per_kind: int | None = None,
                   seed: int = SEED) -> list[str]:
    """Sorted list of sequence folders. `max_per_kind` makes a balanced, seeded subsample
    (e.g. 10 illumination + 10 viewpoint) for quick experiments."""
    seqs = sorted(d for d in glob.glob(os.path.join(root, "[iv]_*")) if os.path.isdir(d))
    seqs = [s for s in seqs if os.path.basename(s) not in set(exclude or [])]
    if max_per_kind:
        rng = random.Random(seed)
        out = []
        for k in ["i", "v"]:
            ks = [s for s in seqs if os.path.basename(s).startswith(k + "_")]
            out += sorted(rng.sample(ks, min(max_per_kind, len(ks))))
        seqs = out
    return seqs


def seq_kind(seq_dir: str) -> str:
    return "illumination" if os.path.basename(seq_dir).startswith("i_") else "viewpoint"


def load_sequence(seq_dir: str, max_side: int | None = None, resize: tuple | None = None):
    """Load the 6 grayscale images and the 5 GT homographies H_1_i (maps image 1 -> image i).

    max_side: downscale keeping the aspect ratio.
    resize=(H, W): resize every image to exactly HxW (e.g. (480, 640) as in SuperPoint).
    Homographies are corrected for the scaling:  H' = S_i @ H @ inv(S_1),  S = diag(sx, sy, 1)
    """
    imgs, scales = [], []
    for i in range(1, 7):
        im = cv2.imread(os.path.join(seq_dir, f"{i}.ppm"), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(os.path.join(seq_dir, f"{i}.ppm"))
        sx = sy = 1.0
        if resize:
            h, w = im.shape
            sx, sy = resize[1] / w, resize[0] / h
            im = cv2.resize(im, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)
        elif max_side and max(im.shape) > max_side:
            sx = sy = max_side / max(im.shape)
            im = cv2.resize(im, None, fx=sx, fy=sy, interpolation=cv2.INTER_AREA)
        imgs.append(im)
        scales.append((sx, sy))
    Hs = {}
    for i in range(2, 7):
        H = np.loadtxt(os.path.join(seq_dir, f"H_1_{i}"))
        S1 = np.diag([scales[0][0], scales[0][1], 1.0])
        Si = np.diag([scales[i - 1][0], scales[i - 1][1], 1.0])
        Hs[i] = Si @ H @ np.linalg.inv(S1)
    return imgs, Hs


def random_homography(h: int, w: int, rng: np.random.Generator, max_rot_deg=30.0,
                      scale_range=(0.7, 1.3), persp=0.15, max_shift=0.1) -> np.ndarray:
    """Random but well-conditioned homography that keeps most of the image in view."""
    c = np.array([w / 2, h / 2])
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    a = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
    s = rng.uniform(*scale_range)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * s
    dst = (src - c) @ R.T + c
    dst += rng.uniform(-persp, persp, size=(4, 2)) * np.array([w, h]) / 2
    dst += rng.uniform(-max_shift, max_shift, size=2) * np.array([w, h])
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)


def photometric_change(img: np.ndarray, rng: np.random.Generator, strength=1.0) -> np.ndarray:
    """Gamma / gain / bias / noise change (illumination-like)."""
    x = img.astype(np.float32) / 255.0
    x = x ** rng.uniform(1 / (1 + 1.2 * strength), 1 + 1.2 * strength)
    x = x * rng.uniform(1 - 0.4 * strength, 1 + 0.3 * strength) + rng.uniform(-0.1, 0.1) * strength
    x = x + rng.normal(0, 0.01 * strength, x.shape)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def _sample_images():
    """Built-in scikit-image pictures (no download) used for smoke tests."""
    from skimage import data
    names = ["astronaut", "camera", "coffee", "chelsea", "rocket", "brick", "grass",
             "gravel", "coins", "moon", "page", "text", "hubble_deep_field", "immunohistochemistry"]
    out = []
    for n in names:
        try:
            im = getattr(data, n)()
        except Exception:
            continue
        if im.ndim == 3:
            im = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2GRAY)
        im = im.astype(np.uint8)
        if max(im.shape) < 400:
            im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        out.append(im)
    return out


def make_synthetic_hpatches(out_dir: str, n_per_kind: int = 2, seed: int = SEED) -> str:
    """Tiny fake HPatches (same folder/file format) for SMOKE TESTS only. Never report these numbers."""
    rng = np.random.default_rng(seed)
    imgs = _sample_images()
    os.makedirs(out_dir, exist_ok=True)
    k = 0
    for kind in ["i", "v"]:
        for j in range(n_per_kind):
            base = imgs[k % len(imgs)]
            k += 1
            d = os.path.join(out_dir, f"{kind}_synthetic{j}")
            os.makedirs(d, exist_ok=True)
            h, w = base.shape
            cv2.imwrite(os.path.join(d, "1.ppm"), cv2.cvtColor(base, cv2.COLOR_GRAY2BGR))
            for i in range(2, 7):
                if kind == "i":
                    H = np.eye(3)
                    im = photometric_change(base, rng, strength=0.3 * (i - 1))
                else:
                    H = random_homography(h, w, rng, max_rot_deg=6 * (i - 1), persp=0.03 * (i - 1))
                    im = cv2.warpPerspective(base, H, (w, h))
                cv2.imwrite(os.path.join(d, f"{i}.ppm"), cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
                np.savetxt(os.path.join(d, f"H_1_{i}"), H)
    return out_dir


# =====================================================================================
# 3. FEATURE EXTRACTORS
# =====================================================================================
class Features:
    """Keypoints + descriptors for one image."""

    def __init__(self, kps, desc, norm):
        self.kps = list(kps) if kps is not None else []
        self.pts = np.array([k.pt for k in self.kps], dtype=np.float32).reshape(-1, 2)
        self.desc = desc
        self.norm = norm  # cv2.NORM_L2 or cv2.NORM_HAMMING

    def __len__(self):
        return len(self.kps)


class Extractor:
    """Base class: `ext(img)` returns Features; `ext.last_times` holds detect/describe ms."""
    name = "base"
    norm = cv2.NORM_L2
    dim = 0
    bytes_per_desc = 0

    def detect(self, img):
        raise NotImplementedError

    def describe(self, img, kps):
        raise NotImplementedError

    def __call__(self, img) -> Features:
        t0 = time.perf_counter()
        kps = self.detect(img)
        t1 = time.perf_counter()
        kps, desc = self.describe(img, kps)
        t2 = time.perf_counter()
        self.last_times = {"detect_ms": 1e3 * (t1 - t0), "describe_ms": 1e3 * (t2 - t1)}
        return Features(kps, desc, self.norm)


class SIFTExtractor(Extractor):
    """OpenCV SIFT (Lowe 2004). root=True gives RootSIFT (Arandjelovic & Zisserman 2012)."""

    def __init__(self, n_features=2000, root=False):
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.root = root
        self.name = "RootSIFT" if root else "SIFT"
        self.dim, self.bytes_per_desc = 128, 128 * 4

    def detect(self, img):
        return self.sift.detect(img, None)

    def describe(self, img, kps):
        kps, d = self.sift.compute(img, kps)
        if d is None:
            return [], np.zeros((0, 128), np.float32)
        if self.root:
            d = d / (np.abs(d).sum(1, keepdims=True) + 1e-7)
            d = np.sqrt(d)
        return kps, d.astype(np.float32)

    def __call__(self, img):
        # detectAndCompute is faster than detect()+compute() (scale space built once)
        t0 = time.perf_counter()
        kps, d = self.sift.detectAndCompute(img, None)
        if d is None:
            kps, d = [], np.zeros((0, 128), np.float32)
        if self.root and len(kps):
            d = np.sqrt(d / (np.abs(d).sum(1, keepdims=True) + 1e-7))
        t1 = time.perf_counter()
        self.last_times = {"detect_ms": float("nan"), "describe_ms": float("nan"),
                           "total_ms": 1e3 * (t1 - t0)}
        return Features(kps, d.astype(np.float32), self.norm)


class ORBExtractor(Extractor):
    """OpenCV ORB (Rublee et al. 2011): FAST + oriented BRIEF, 256-bit binary descriptor."""
    norm = cv2.NORM_HAMMING

    def __init__(self, n_features=2000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.name = "ORB"
        self.dim, self.bytes_per_desc = 256, 32

    def detect(self, img):
        return self.orb.detect(img, None)

    def describe(self, img, kps):
        kps, d = self.orb.compute(img, kps)
        if d is None:
            return [], np.zeros((0, 32), np.uint8)
        return kps, d

    def __call__(self, img):
        t0 = time.perf_counter()
        kps, d = self.orb.detectAndCompute(img, None)
        if d is None:
            kps, d = [], np.zeros((0, 32), np.uint8)
        t1 = time.perf_counter()
        self.last_times = {"detect_ms": float("nan"), "describe_ms": float("nan"),
                           "total_ms": 1e3 * (t1 - t0)}
        return Features(kps, d, self.norm)


class HarrisSIFTExtractor(Extractor):
    """Week-9 style: Harris corners (single scale, upright) + SIFT descriptor."""

    def __init__(self, n_features=2000, kp_size=12.0, quality=0.001, min_dist=4):
        self.n, self.kp_size, self.quality, self.min_dist = n_features, kp_size, quality, min_dist
        self.sift = cv2.SIFT_create()
        self.name = "Harris+SIFT"
        self.dim, self.bytes_per_desc = 128, 512

    def detect(self, img):
        c = cv2.goodFeaturesToTrack(img, self.n, self.quality, self.min_dist,
                                    useHarrisDetector=True, k=0.04)
        if c is None:
            return []
        return [cv2.KeyPoint(float(x), float(y), self.kp_size, 0.0) for x, y in c.reshape(-1, 2)]

    def describe(self, img, kps):
        if len(kps) == 0:
            return [], np.zeros((0, 128), np.float32)
        kps, d = self.sift.compute(img, kps)
        return kps, d.astype(np.float32)


class LearnedExtractor(Extractor):
    """SIFT keypoints (location, scale, orientation) + our compact CNN descriptor.

    Using the SAME detector as SIFT isolates the effect of the descriptor alone.
    binary=True -> sign(descriptor) packed into bits, matched with Hamming distance.
    """

    def __init__(self, model, n_features=2000, device=None, binary=False,
                 mag=PATCH_MAG, batch=2048, name=None):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.detector = cv2.SIFT_create(nfeatures=n_features)
        self.binary, self.mag, self.batch = binary, mag, batch
        d = model.dim
        if binary:
            self.norm, self.dim, self.bytes_per_desc = cv2.NORM_HAMMING, d, int(math.ceil(d / 8))
        else:
            self.norm, self.dim, self.bytes_per_desc = cv2.NORM_L2, d, d * 4
        self.name = name or f"Ours-{'bin' if binary else 'f'}{d}"

    def detect(self, img):
        return self.detector.detect(img, None)

    def describe(self, img, kps):
        if len(kps) == 0:
            return [], np.zeros((0, self.bytes_per_desc if self.binary else self.dim),
                                np.uint8 if self.binary else np.float32)
        centers, frames = keypoint_frames(kps, self.mag)
        patches = extract_patches(img, centers, frames, PATCH_SIZE)
        d = describe_patches(self.model, patches, self.device, self.batch)
        if self.binary:
            d = binarize(d)
        return kps, d


def binarize(desc: np.ndarray) -> np.ndarray:
    """float descriptor -> packed bits (1 bit per dimension)."""
    return np.packbits((desc > 0).astype(np.uint8), axis=1)


# =====================================================================================
# 4. PATCH EXTRACTION  (Student B)
# =====================================================================================
def keypoint_frames(kps, mag=PATCH_MAG):
    """Centers (N,2) and 2x2 frames (N,2,2) mapping patch coords [-1,1]^2 to image offsets.

    Half-side = mag * kp.size / 2. Orientation = kp.angle in image coordinates (y down);
    this sign convention was verified empirically: patches from an image and from its
    rotated copy correlate at NCC ~0.95 (vs ~0.07 with the opposite sign).
    """
    centers = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
    half = np.array([mag * k.size / 2.0 for k in kps], dtype=np.float32)
    th = np.deg2rad(np.array([k.angle if k.angle >= 0 else 0 for k in kps], dtype=np.float32))
    c, s = np.cos(th), np.sin(th)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (N,2,2)
    return centers, R * half[:, None, None]


def extract_patches(img: np.ndarray, centers: np.ndarray, frames: np.ndarray,
                    size: int = PATCH_SIZE) -> np.ndarray:
    """Sample N oriented, scale-normalised patches (N,size,size) uint8 with bilinear
    interpolation from a Gaussian pyramid (avoids aliasing for large keypoints)."""
    N = len(centers)
    out = np.zeros((N, size, size), np.uint8)
    if N == 0:
        return out
    scale = np.sqrt(np.abs(np.linalg.det(frames)))                 # half-side in pixels
    step = 2 * scale / (size - 1)                                   # pixels per patch pixel
    levels = np.clip(np.floor(np.log2(np.maximum(step, 1.0))), 0, 6).astype(int)
    pyr = [img.astype(np.float32)]
    for _ in range(int(levels.max())):
        pyr.append(cv2.pyrDown(pyr[-1]))
    lin = torch.linspace(-1, 1, size)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    base = torch.stack([gx, gy], -1).view(-1, 2)                    # (S*S,2) x,y
    for lv in np.unique(levels):
        idx = np.where(levels == lv)[0]
        im = pyr[lv]
        Hh, Ww = im.shape
        f = 2.0 ** lv
        c = torch.from_numpy(np.asarray(centers[idx] / f, np.float32))          # (n,2)
        A = torch.from_numpy(np.asarray(frames[idx] / f, np.float32))           # (n,2,2)
        pts = c[:, None, :] + torch.einsum("pk,njk->npj", base, A)   # (n,S*S,2)
        gxn = 2 * pts[..., 0] / (Ww - 1) - 1
        gyn = 2 * pts[..., 1] / (Hh - 1) - 1
        grid = torch.stack([gxn, gyn], -1).view(1, len(idx) * size, size, 2).float()
        t = torch.from_numpy(im)[None, None]
        p = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)
        out[idx] = p.view(len(idx), size, size).clamp(0, 255).round().byte().numpy()
    return out


def homography_jacobian(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Local 2x2 affine approximation (Jacobian) of homography H at points (N,2)."""
    x, y = pts[:, 0], pts[:, 1]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    J = np.empty((len(pts), 2, 2))
    J[:, 0, 0] = (H[0, 0] - u * H[2, 0]) / w
    J[:, 0, 1] = (H[0, 1] - u * H[2, 1]) / w
    J[:, 1, 0] = (H[1, 0] - v * H[2, 0]) / w
    J[:, 1, 1] = (H[1, 1] - v * H[2, 1]) / w
    return J


# =====================================================================================
# 5. COMPACT CNN DESCRIPTOR  (Student B)
# =====================================================================================
class CompactDescriptor(nn.Module):
    """L2-Net / HardNet-style fully-convolutional descriptor, shrunk by `width`.

    width=32, dim=128 reproduces the HardNet architecture (~1.3M params).
    width=16, dim=64  is our default compact model (~0.3M params, 256 B/descriptor).
    Input: (N,1,32,32) patches in [0,1]. Output: (N,dim) L2-normalised descriptors.
    """

    def __init__(self, width: int = 16, dim: int = 64, dropout: float = 0.1):
        super().__init__()
        w = width
        self.width, self.dim, self.dropout = width, dim, dropout

        def block(i, o, s=1):
            return [nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False),
                    nn.BatchNorm2d(o, affine=False), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *block(1, w), *block(w, w),
            *block(w, 2 * w, 2), *block(2 * w, 2 * w),
            *block(2 * w, 4 * w, 2), *block(4 * w, 4 * w),
            nn.Dropout(dropout),
            nn.Conv2d(4 * w, dim, 8, bias=False),
            nn.BatchNorm2d(dim, affine=False),
        )

    def config(self):
        return {"width": self.width, "dim": self.dim, "dropout": self.dropout}

    @staticmethod
    def _input_norm(x):
        flat = x.reshape(x.size(0), -1)
        mp = flat.mean(1).view(-1, 1, 1, 1)
        sp = flat.std(1).view(-1, 1, 1, 1) + 1e-7
        return (x - mp) / sp

    def forward(self, x):
        x = self.features(self._input_norm(x))
        return F.normalize(x.reshape(x.size(0), -1), dim=1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, size: int = PATCH_SIZE) -> int:
    """Multiply-accumulate operations per patch (conv layers only)."""
    macs = []

    def hook(m, i, o):
        macs.append(o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, size, size, device=next(model.parameters()).device))
    for h in hs:
        h.remove()
    model.train(was_training)
    return int(sum(macs))


@torch.no_grad()
def describe_patches(model, patches: np.ndarray, device=None, batch: int = 2048) -> np.ndarray:
    """uint8 patches (N,S,S) -> float32 descriptors (N,dim)."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(patches), batch):
        x = torch.from_numpy(patches[i:i + batch]).float().div_(255).unsqueeze(1).to(device)
        out.append(model(x).float().cpu())
    if not out:
        return np.zeros((0, model.dim), np.float32)
    return torch.cat(out).numpy()


def sift_describe_patches(patches: np.ndarray) -> np.ndarray:
    """Reference: SIFT descriptor computed on the SAME 32x32 patches (upsampled to 64x64),
    so patch-level FPR95 of SIFT and of the CNN can be compared directly."""
    sift = cv2.SIFT_create()
    kp = [cv2.KeyPoint(31.5, 31.5, 64 / PATCH_MAG, 0)]
    out = np.zeros((len(patches), 128), np.float32)
    for i, p in enumerate(patches):
        big = cv2.resize(p, (64, 64), interpolation=cv2.INTER_LINEAR)
        _, d = sift.compute(big, kp)
        if d is not None and len(d):
            out[i] = d[0]
    return out


def save_checkpoint(model, path, **extra):
    torch.save({"config": model.config(), "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    m = CompactDescriptor(**ck["config"]).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck


# =====================================================================================
# 6. TRAINING DATA + TRAINING LOOP  (Student B)
# =====================================================================================
class PairSampler:
    """Holds matching patch pairs and samples batches of DISTINCT 3D points.

    Two sources:
      * from_phototour(): Brown/UBC PhotoTour patches (the standard HardNet training set)
      * from_images():    self-supervised pairs made by warping any images with random
                          homographies (SuperPoint-style "homographic adaptation")
    """

    def __init__(self, sample_fn, n_points, name):
        self._sample, self.n_points, self.name = sample_fn, n_points, name

    def sample(self, batch_size, rng: np.random.Generator):
        return self._sample(batch_size, rng)

    # ---- PhotoTour ----
    @classmethod
    def from_phototour(cls, root, name="liberty", max_patches=None, seed=SEED):
        from torchvision.datasets import PhotoTour
        ds = PhotoTour(root=root, name=name, train=True, download=True)
        data, labels = ds.data, ds.labels.numpy()
        if max_patches and max_patches < len(labels):
            # keep whole 3D points so that every kept label still has >=2 patches
            cut = np.searchsorted(labels, labels[max_patches])
            data, labels = data[:cut], labels[:cut]
        small = data.view(-1, 32, 2, 32, 2).float().mean((2, 4)).round().byte()  # 64->32 area resize
        order = np.argsort(labels, kind="stable")
        labels_s = labels[order]
        uniq, start, count = np.unique(labels_s, return_index=True, return_counts=True)
        keep = count >= 2
        uniq, start, count = uniq[keep], start[keep], count[keep]

        def sample(b, rng):
            sel = rng.choice(len(uniq), size=min(b, len(uniq)), replace=False)
            i1 = rng.integers(0, count[sel])
            i2 = (i1 + rng.integers(1, count[sel])) % count[sel]
            a = small[order[start[sel] + i1]]
            p = small[order[start[sel] + i2]]
            return a, p

        print(f"PhotoTour/{name}: {len(labels)} patches, {len(uniq)} 3D points")
        return cls(sample, len(uniq), f"phototour-{name}")

    # ---- Synthetic homography pairs from any image folder ----
    @classmethod
    def from_images(cls, images, pairs_per_image=300, seed=SEED, mag=PATCH_MAG):
        """images: list of uint8 grayscale arrays (NOT from HPatches -> no test leakage)."""
        rng = np.random.default_rng(seed)
        sift = cv2.SIFT_create(nfeatures=pairs_per_image * 2)
        A, P = [], []
        for img in images:
            h, w = img.shape
            H = random_homography(h, w, rng)
            img2 = photometric_change(cv2.warpPerspective(img, H, (w, h)), rng, strength=0.5)
            kps = sift.detect(img, None)
            if len(kps) == 0:
                continue
            c1, f1 = keypoint_frames(kps, mag)
            ones = np.ones((len(c1), 1))
            c2 = (H @ np.hstack([c1, ones]).T).T
            c2 = (c2[:, :2] / c2[:, 2:]).astype(np.float32)
            f2 = np.einsum("nij,njk->nik", homography_jacobian(H, c1), f1).astype(np.float32)
            r1 = np.sqrt(np.abs(np.linalg.det(f1))) * 1.5
            r2 = np.sqrt(np.abs(np.linalg.det(f2))) * 1.5
            ok = ((c1[:, 0] > r1) & (c1[:, 1] > r1) & (c1[:, 0] < w - r1) & (c1[:, 1] < h - r1) &
                  (c2[:, 0] > r2) & (c2[:, 1] > r2) & (c2[:, 0] < w - r2) & (c2[:, 1] < h - r2))
            idx = np.where(ok)[0]
            if len(idx) > pairs_per_image:
                idx = rng.choice(idx, pairs_per_image, replace=False)
            if len(idx) == 0:
                continue
            A.append(extract_patches(img, c1[idx], f1[idx]))
            P.append(extract_patches(img2, c2[idx], f2[idx]))
        A = torch.from_numpy(np.concatenate(A))
        P = torch.from_numpy(np.concatenate(P))

        def sample(b, rng):
            sel = rng.choice(len(A), size=min(b, len(A)), replace=False)
            return A[sel], P[sel]

        print(f"Homography pairs: {len(A)} pairs from {len(images)} images")
        sampler = cls(sample, len(A), "homography")
        sampler.A, sampler.P = A, P
        return sampler


def load_image_folder(folder, max_images=None, max_side=640, seed=SEED):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.*"), recursive=True))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".ppm", ".bmp"))]
    if max_images and len(files) > max_images:
        files = sorted(random.Random(seed).sample(files, max_images))
    out = []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if max(im.shape) > max_side:
            s = max_side / max(im.shape)
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out.append(im)
    return out


def download_coco_val(data_dir="/content/data") -> str:
    """COCO val2017 (5k images, ~1 GB) -- fallback training images if PhotoTour is unavailable."""
    folder = os.path.join(data_dir, "val2017")
    if os.path.isdir(folder) and len(os.listdir(folder)) > 100:
        return folder
    z = os.path.join(data_dir, "val2017.zip")
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(z):
        _download(COCO_VAL_URL, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(data_dir)
    return folder


def hardnet_loss(a, p, margin=1.0, anchor_swap=True):
    """Hardest-in-batch triplet margin loss (Mishchuk et al., NeurIPS 2017)."""
    d = torch.cdist(a, p) + 1e-8
    pos = torch.diag(d)
    eye = torch.eye(d.size(0), device=d.device)
    d_neg = d + 10.0 * eye
    d_neg = d_neg + 10.0 * (d_neg < 0.008).float()        # ignore near-duplicates
    neg = d_neg.min(1).values
    if anchor_swap:
        neg = torch.minimum(neg, d_neg.min(0).values)
    return F.relu(margin + pos - neg).mean()


def _augment_pair(a, p, rng):
    """Same random flip / 90-degree rotation applied to both patches of a pair."""
    k = int(rng.integers(0, 4))
    if k:
        a, p = torch.rot90(a, k, (2, 3)), torch.rot90(p, k, (2, 3))
    if rng.random() < 0.5:
        a, p = a.flip(3), p.flip(3)
    return a, p


def train_descriptor(model, sampler: PairSampler, epochs=10, steps_per_epoch=500, batch_size=512,
                     lr=1e-3, weight_decay=1e-4, device=None, val_fn=None, ckpt_path=None,
                     seed=SEED, augment=True, log=print):
    """Train with Adam + cosine LR. `val_fn(model) -> float` (e.g. FPR95, lower is better).
    Saves the best checkpoint (by val) to `ckpt_path`. Returns a list of per-epoch dicts."""
    device = device or get_device()
    model.to(device)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    hist, best = [], float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for _ in range(steps_per_epoch):
            a, p = sampler.sample(batch_size, rng)
            a = a.float().div(255).unsqueeze(1).to(device)
            p = p.float().div(255).unsqueeze(1).to(device)
            if augment:
                a, p = _augment_pair(a, p, rng)
            loss = hardnet_loss(model(a), model(p))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        row = {"epoch": ep, "loss": float(np.mean(losses)), "epoch_sec": time.time() - t0}
        if val_fn is not None:
            model.eval()
            row["val_fpr95"] = float(val_fn(model))
        hist.append(row)
        log(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        score = row.get("val_fpr95", row["loss"])
        if ckpt_path and score < best:
            best = score
            save_checkpoint(model, ckpt_path, history=hist, sampler=sampler.name, seed=seed)
    return hist


# ---- patch-level validation (FPR95) ----
def fpr95(distances: np.ndarray, is_match: np.ndarray) -> float:
    """False-positive rate when 95% of the true matches are accepted (lower is better)."""
    distances, is_match = np.asarray(distances), np.asarray(is_match).astype(bool)
    thr = np.quantile(distances[is_match], 0.95)
    neg = ~is_match
    return float((distances[neg] <= thr).sum() / max(neg.sum(), 1))


def make_val_fn_pairs(A: torch.Tensor, P: torch.Tensor, seed=SEED, n=10000):
    """FPR95 on held-out (anchor, positive) patch pairs; negatives = shuffled positives."""
    rng = np.random.default_rng(seed)
    n = min(n, len(A))
    sel = rng.choice(len(A), n, replace=False)
    a, p = A[sel].numpy(), P[sel].numpy()
    perm = (np.arange(n) + rng.integers(1, n, n)) % n

    def val(model):
        da = describe_patches(model, a)
        dp = describe_patches(model, p)
        pos = np.linalg.norm(da - dp, axis=1)
        neg = np.linalg.norm(da - dp[perm], axis=1)
        return fpr95(np.concatenate([pos, neg]), np.r_[np.ones(n), np.zeros(n)])

    return val


def make_val_fn_phototour(root, name="notredame", n=20000, seed=SEED):
    """Standard PhotoTour FPR95 on the official 100k match/non-match pairs (subsampled to n)."""
    from torchvision.datasets import PhotoTour
    ds = PhotoTour(root=root, name=name, train=False, download=True)
    m = ds.matches.numpy()
    rng = np.random.default_rng(seed)
    m = m[rng.choice(len(m), min(n, len(m)), replace=False)]
    small = ds.data.view(-1, 32, 2, 32, 2).float().mean((2, 4)).round().byte().numpy()
    a, p, lab = small[m[:, 0]], small[m[:, 1]], m[:, 2]

    def val(model):
        da, dp = describe_patches(model, a), describe_patches(model, p)
        return fpr95(np.linalg.norm(da - dp, axis=1), lab)

    return val


# =====================================================================================
# 7. MATCHING + METRICS + TIMING  (Student C)
# =====================================================================================
def match(f1: Features, f2: Features, mode="mnn", ratio=0.8) -> np.ndarray:
    """Return (M,2) index pairs. mode='mnn' (mutual nearest neighbour, the HPatches/D2-Net
    protocol) or 'ratio' (Lowe's ratio test)."""
    if len(f1) == 0 or len(f2) == 0:
        return np.zeros((0, 2), int)
    if mode == "mnn":
        ms = cv2.BFMatcher(f1.norm, crossCheck=True).match(f1.desc, f2.desc)
        return np.array([[m.queryIdx, m.trainIdx] for m in ms], int).reshape(-1, 2)
    if mode == "ratio":
        if len(f2) < 2:
            return np.zeros((0, 2), int)
        knn = cv2.BFMatcher(f1.norm).knnMatch(f1.desc, f2.desc, k=2)
        good = [[m[0].queryIdx, m[0].trainIdx] for m in knn
                if len(m) == 2 and m[0].distance < ratio * m[1].distance]
        return np.array(good, int).reshape(-1, 2)
    raise ValueError(mode)


def warp_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:]


def corner_error(H_est, H_gt, w, h) -> float:
    """Mean distance between the 4 image corners warped by H_est and by H_gt."""
    if H_est is None:
        return float("inf")
    c = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    return float(np.linalg.norm(warp_points(c, H_est) - warp_points(c, H_gt), axis=1).mean())


def evaluate_pair(f1: Features, f2: Features, matches: np.ndarray, H_gt: np.ndarray,
                  shape1, ransac_thresh=RANSAC_THRESHOLD, seed=SEED,
                  ransac_iters=5000, ransac_conf=0.999) -> dict:
    """All per-pair metrics: MMA@1..10 px, homography corner error, inliers, counts."""
    h, w = shape1[:2]
    r = {"n_kp1": len(f1), "n_kp2": len(f2), "n_matches": len(matches)}
    if len(matches):
        p1, p2 = f1.pts[matches[:, 0]], f2.pts[matches[:, 1]]
        err = np.linalg.norm(warp_points(p1, H_gt) - p2, axis=1)
    else:
        err = np.zeros(0)
    for t in MMA_THRESHOLDS:
        r[f"mma@{t}"] = float((err <= t).mean()) if len(err) else 0.0
    r["n_correct@3"] = int((err <= 3).sum())
    H_est, n_in = None, 0
    if len(matches) >= 4:
        cv2.setRNGSeed(seed)
        H_est, mask = cv2.findHomography(p1, p2, cv2.RANSAC, ransac_thresh,
                                     maxIters=ransac_iters, confidence=ransac_conf)
        n_in = int(mask.sum()) if mask is not None else 0
    r["n_inliers"] = n_in
    r["inlier_ratio"] = n_in / len(matches) if len(matches) else 0.0
    r["corner_err"] = corner_error(H_est, H_gt, w, h)
    for t in HOMOG_THRESHOLDS:
        r[f"hacc@{t}"] = float(r["corner_err"] <= t)
    return r


def time_extractor(ext: Extractor, img: np.ndarray, warmup=2, repeats=10) -> dict:
    """Median wall-clock time of a full extraction (detect+describe) on one image."""
    for _ in range(warmup):
        ext(img)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ext(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t0))
    return {"method": ext.name, "extract_ms_median": float(np.median(ts)),
            "extract_ms_iqr": float(np.subtract(*np.percentile(ts, [75, 25])))}


def time_describe_only(ext: Extractor, img: np.ndarray, kps, warmup=2, repeats=10) -> float:
    """Median time to DESCRIBE a fixed set of keypoints (isolates the descriptor cost)."""
    for _ in range(warmup):
        ext.describe(img, kps)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ext.describe(img, kps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(ts))


# =====================================================================================
# 8. BENCHMARK RUNNER + REPORTING  (Student C)
# =====================================================================================
def run_benchmark(extractor: Extractor, seq_dirs, match_mode="mnn", ratio=0.8, max_side=None,
                  method_name=None, verbose=True, ransac_seeds=(SEED,),
                  ransac_thresh=RANSAC_THRESHOLD, resize=None, ransac_iters=5000,
                  ransac_conf=0.999) -> pd.DataFrame:
    """Image 1 vs images 2..6 of every sequence. One row per image pair."""
    name = method_name or extractor.name
    rows = []
    t_start = time.time()
    for si, sd in enumerate(seq_dirs):
        imgs, Hs = load_sequence(sd, max_side, resize)
        feats, tex = [], []
        for im in imgs:
            t0 = time.perf_counter()
            feats.append(extractor(im))
            tex.append(1e3 * (time.perf_counter() - t0))
        for i in range(2, 7):
            t0 = time.perf_counter()
            m = match(feats[0], feats[i - 1], match_mode, ratio)
            tm = 1e3 * (time.perf_counter() - t0)
            r = evaluate_pair(feats[0], feats[i - 1], m, Hs[i], imgs[0].shape, ransac_thresh, ransac_seeds[0],
                              ransac_iters, ransac_conf)
            if len(ransac_seeds) > 1:   # average homography accuracy over several RANSAC seeds
                rs = [r] + [evaluate_pair(feats[0], feats[i - 1], m, Hs[i], imgs[0].shape, ransac_thresh, sd,
                                          ransac_iters, ransac_conf)
                            for sd in ransac_seeds[1:]]
                for k in ["corner_err", "n_inliers", "inlier_ratio"] + [f"hacc@{t}" for t in HOMOG_THRESHOLDS]:
                    r[k] = float(np.mean([x[k] for x in rs]))
            r.update({"method": name, "seq": os.path.basename(sd), "kind": seq_kind(sd), "pair": f"1-{i}",
                      "match_mode": match_mode, "extract_ms": float(np.mean(tex)), "match_ms": tm,
                      "dim": extractor.dim, "bytes_per_desc": extractor.bytes_per_desc})
            rows.append(r)
        if verbose and (si + 1) % 10 == 0:
            print(f"  [{name}] {si + 1}/{len(seq_dirs)} sequences, {time.time() - t_start:.0f}s")
    if verbose:
        print(f"  [{name}] done: {len(rows)} pairs in {time.time() - t_start:.0f}s")
    cols = ["method", "seq", "kind", "pair"]
    df = pd.DataFrame(rows)
    return df[cols + [c for c in df.columns if c not in cols]]


SUMMARY_COLS = ["mma@1", "mma@3", "mma@5", "mma@10", "hacc@1", "hacc@3", "hacc@5",
                "n_kp1", "n_matches", "inlier_ratio", "extract_ms", "match_ms", "bytes_per_desc"]


def summarize(df: pd.DataFrame, by=("method",), split_kind=True) -> pd.DataFrame:
    """Mean of each metric per method (and per illumination / viewpoint / overall)."""
    by = list(by)
    parts = [df.assign(subset="overall")]
    if split_kind:
        parts += [df.assign(subset=df["kind"])]
    d = pd.concat(parts)
    g = d.groupby(by + ["subset"], sort=False)[[c for c in SUMMARY_COLS if c in d]].mean()
    return g


def mma_curve(df: pd.DataFrame, method: str, kind: str | None = None) -> np.ndarray:
    d = df[df["method"] == method]
    if kind:
        d = d[d["kind"] == kind]
    return np.array([d[f"mma@{t}"].mean() for t in MMA_THRESHOLDS])


def plot_mma(df: pd.DataFrame, path=None, methods=None, title_prefix=""):
    import matplotlib.pyplot as plt
    methods = methods or list(dict.fromkeys(df["method"]))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, kind in zip(axs, [None, "illumination", "viewpoint"]):
        for m in methods:
            ax.plot(MMA_THRESHOLDS, mma_curve(df, m, kind), marker="o", ms=3, label=m)
        ax.set_title(f"{title_prefix}{kind or 'overall'}")
        ax.set_xlabel("threshold [px]")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("MMA")
    axs[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def _tex_escape(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def df_to_latex(df: pd.DataFrame, path=None, caption="", label="", fmt="{:.3f}",
                bold_max=(), bold_min=()) -> str:
    """Write a booktabs LaTeX table (drop it into the report with \\input{tables/xxx.tex}).
    bold_max / bold_min: columns where the best value is bolded."""
    d = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df.copy()
    best = {}
    for c in bold_max:
        if c in d:
            best[c] = d[c].max()
    for c in bold_min:
        if c in d:
            best[c] = d[c].min()

    def cell(c, v):
        if isinstance(v, (float, np.floating)):
            s = fmt.format(v) if np.isfinite(v) else "--"
        else:
            s = _tex_escape(v)
        if c in best and v == best[c]:
            s = r"\textbf{" + s + "}"
        return s

    align = "".join("r" if pd.api.types.is_numeric_dtype(d[c]) else "l" for c in d.columns)
    lines = [r"\begin{table}[h]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}", r"\toprule",
             " & ".join(r"\textbf{" + _tex_escape(c) + "}" for c in d.columns) + r" \\", r"\midrule"]
    for _, row in d.iterrows():
        lines.append(" & ".join(cell(c, row[c]) for c in d.columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    tex = "\n".join(lines)
    if path:
        with open(path, "w") as f:
            f.write(tex + "\n")
    return tex


# =====================================================================================
# 9. REGISTRATION DEMO  (Student C)
# =====================================================================================
def register(img1, img2, ext: Extractor, match_mode="mnn", ransac_thresh=RANSAC_THRESHOLD):
    """Estimate the homography img1 -> img2. Returns (H, f1, f2, matches, inlier_mask)."""
    f1, f2 = ext(img1), ext(img2)
    m = match(f1, f2, match_mode)
    if len(m) < 4:
        return None, f1, f2, m, np.zeros(len(m), bool)
    cv2.setRNGSeed(SEED)
    H, mask = cv2.findHomography(f1.pts[m[:, 0]], f2.pts[m[:, 1]], cv2.RANSAC, ransac_thresh)
    return H, f1, f2, m, (mask.ravel().astype(bool) if mask is not None else np.zeros(len(m), bool))


def checkerboard_overlay(img1, img2, H, tiles=8):
    """Warp img1 into img2's frame and interleave them in a checkerboard: misalignment
    shows up as broken edges at tile borders."""
    h, w = img2.shape[:2]
    warped = cv2.warpPerspective(img1, H, (w, h)) if H is not None else np.zeros_like(img2)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy * tiles // h + xx * tiles // w) % 2).astype(bool)
    out = img2.copy()
    out[mask] = warped[mask]
    return out


def draw_inlier_matches(img1, img2, f1, f2, m, inliers, max_draw=150, seed=SEED):
    idx = np.where(inliers)[0]
    if len(idx) > max_draw:
        idx = np.random.default_rng(seed).choice(idx, max_draw, replace=False)
    dm = [cv2.DMatch(int(m[i, 0]), int(m[i, 1]), 0) for i in idx]
    return cv2.drawMatches(img1, f1.kps, img2, f2.kps, dm, None,
                           matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
