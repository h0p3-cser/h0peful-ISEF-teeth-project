"""Deterministic same-modality registration; transforms locate ROIs, not intensities.

All matrices map source ORIGINAL pixels to target ORIGINAL pixels.  Canonical
tooth crops only assist fitting; measurements must use the original JPEG.
"""
from dataclasses import dataclass
import cv2
import numpy as np


CONFIG = dict(canvas=480, pyramid=[120, 240, 480], iterations=80,
              min_dice=.90, min_gray_ncc=.25, max_cycle_fraction=.035,
              min_singular=.55, max_singular=1.8, max_condition=1.8,
              min_containment=.95, seed=1913)


def homogeneous(matrix):
    m = np.asarray(matrix, dtype=float)
    return np.vstack([m, [0., 0., 1.]]) if m.shape == (2, 3) else m.copy()


def points_transform(points, matrix):
    p = np.c_[np.asarray(points, float), np.ones(len(points))]
    return (p @ homogeneous(matrix).T)[:, :2]


@dataclass
class Frame:
    image_id: str
    rgb: np.ndarray
    mask: np.ndarray
    gray: np.ndarray
    gradient: np.ndarray
    softmask: np.ndarray
    canonical: np.ndarray


def canonical_frame(image_id, rgb, mask, canvas=None):
    size = canvas or CONFIG['canvas']
    if rgb.shape[:2] != mask.shape or not np.any(mask):
        raise ValueError('Nonempty original-size tooth mask required')
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    scale = .82 * size / max(w, h)
    transform = np.array([[scale, 0., size / 2 - scale * (x + w / 2)],
                          [0., scale, size / 2 - scale * (y + h / 2)],
                          [0., 0., 1.]])
    m = cv2.warpAffine(mask, transform[:2], (size, size), flags=cv2.INTER_NEAREST)
    # G avoids using background colour or the unstable blue channel for fitting.
    g = cv2.warpAffine(rgb[:, :, 1], transform[:2], (size, size))
    g = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(g)
    g = cv2.GaussianBlur(g.astype(np.float32) / 255., (5, 5), 0)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0),
                         cv2.Sobel(g, cv2.CV_32F, 0, 1))
    inner = cv2.erode(m, np.ones((7, 7), np.uint8)) > 0
    grad /= max(float(np.percentile(grad[inner], 95)), .01)
    grad = np.minimum(grad, 2.)
    grad[~inner] = 0
    # Smooth silhouette supplies a broad basin before interior fitting.
    soft = cv2.GaussianBlur((m > 0).astype(np.float32), (0, 0), 3.)
    return Frame(image_id, rgb, m, g, grad, soft, transform)


def mask_moments(mask):
    y, x = np.nonzero(mask)
    p = np.c_[x, y].astype(float)
    return p.mean(axis=0), np.cov(p.T)


def initial_transforms(a, b):
    ca, va = mask_moments(a.mask)
    cb, vb = mask_moments(b.mask)
    ea, ua = np.linalg.eigh(va)
    eb, ub = np.linalg.eigh(vb)
    # Principal-axis sign is ambiguous. Try both proper rotations explicitly.
    if np.linalg.det(ua) < 0: ua[:, 0] *= -1
    if np.linalg.det(ub) < 0: ub[:, 0] *= -1
    mats = [('centres', np.eye(2))]
    for sign in [1., -1.]:
        linear = ub @ np.diag(sign * np.sqrt(eb / ea)) @ ua.T
        mats.append(('moments_' + str(int(sign)), linear))
    return [(name, np.c_[linear, cb - linear @ ca]) for name, linear in mats]


def pyramid_ecc(a, b, init, kind):
    source, target = getattr(a, kind), getattr(b, kind)
    size = len(a.mask)
    warp = np.asarray(init, np.float32).copy()
    errors = []
    for resolution in CONFIG['pyramid']:
        f = resolution / size
        mat = warp.copy()
        mat[:, 2] *= f
        aa = cv2.resize(source, (resolution, resolution), interpolation=cv2.INTER_AREA)
        bb = cv2.resize(target, (resolution, resolution), interpolation=cv2.INTER_AREA)
        fitmask = None if kind == 'softmask' else cv2.resize(b.mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        try:
            _, candidate = cv2.findTransformECC(aa, bb, mat, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, CONFIG['iterations'], 1e-5), fitmask, 5)
            candidate[:, 2] /= f
            if not np.isfinite(candidate).all(): raise ValueError('Nonfinite transform')
            warp = candidate.copy()
        except (cv2.error, ValueError):
            errors.append(resolution)
    return warp.astype(float), errors


def correlation(x, y):
    if len(x) < 20: return -1.
    xx, yy = x.astype(float) - x.mean(), y.astype(float) - y.mean()
    den = np.linalg.norm(xx) * np.linalg.norm(yy)
    return float(xx @ yy / den) if den > 1e-10 else -1.


def assess(a, b, matrix):
    size = len(a.mask)
    m = np.asarray(matrix, float)
    if not np.isfinite(m).all():
        return dict(dice=0., gray_ncc=-1., gradient_ncc=-1., plausible=False, score=-10.)
    sv = np.linalg.svd(m[:, :2], compute_uv=False)
    det = float(np.linalg.det(m[:, :2]))
    plausible = bool(det > 0 and sv.min() >= CONFIG['min_singular'] and
                     sv.max() <= CONFIG['max_singular'] and sv.max() / sv.min() <= CONFIG['max_condition'])
    moved = cv2.warpAffine(a.mask, m, (size, size), flags=cv2.INTER_NEAREST) > 0
    actual = b.mask > 0
    dice = float(2 * np.logical_and(moved, actual).sum() / max(1, moved.sum() + actual.sum()))
    overlap = cv2.erode((moved & actual).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    gray = cv2.warpAffine(a.gray, m, (size, size))
    grad = cv2.warpAffine(a.gradient, m, (size, size))
    ncc = correlation(gray[overlap], b.gray[overlap])
    gncc = correlation(grad[overlap], b.gradient[overlap])
    # Geometry first. Interior agreement disambiguates similar outlines.
    score = dice + .15 * ncc + .05 * gncc if plausible else -10.
    return dict(dice=dice, gray_ncc=ncc, gradient_ncc=gncc, plausible=plausible,
                singular_min=float(sv.min()), singular_max=float(sv.max()),
                determinant=det, score=score)


def fit_one_direction(a, b):
    attempts = []
    for name, init in initial_transforms(a, b):
        shape, errors = pyramid_ecc(a, b, init, 'softmask')
        metrics = assess(a, b, shape)
        attempts.append(dict(method=name + '_outline', matrix=shape, errors=errors, **metrics))
    base = max(attempts, key=lambda d: d['score'])
    for kind in ['gray', 'gradient']:
        matrix, errors = pyramid_ecc(a, b, base['matrix'], kind)
        attempts.append(dict(method=base['method'] + '_' + kind, matrix=matrix,
                             errors=errors, **assess(a, b, matrix)))
    eligible = [r for r in attempts if r['plausible'] and r['dice'] >= CONFIG['min_dice']]
    best = max(eligible or attempts, key=lambda d: d['score'])
    return best, attempts


def register_frames(a, b, original_points):
    if a.image_id == b.image_id:
        return np.eye(3), dict(method='identity', candidate_ok=True, dice=1., gray_ncc=1., cycle_fraction=0.), []
    best, forward = fit_one_direction(a, b)
    back, reverse = fit_one_direction(b, a)
    m = homogeneous(best['matrix'])
    q = points_transform(original_points, a.canonical)
    returned = points_transform(points_transform(q, m), back['matrix'])
    cycle = float(np.max(np.linalg.norm(returned - q, axis=1)) / (.82 * len(a.mask)))
    ok = bool(best['plausible'] and back['plausible'] and
              min(best['dice'], back['dice']) >= CONFIG['min_dice'] and
              min(best['gray_ncc'], back['gray_ncc']) >= CONFIG['min_gray_ncc'] and
              cycle <= CONFIG['max_cycle_fraction'])
    original = np.linalg.inv(b.canonical) @ m @ a.canonical
    info = {k: v for k, v in best.items() if k not in ['matrix', 'errors']}
    info.update(candidate_ok=ok, cycle_fraction=cycle, reverse_dice=back['dice'],
                reverse_gray_ncc=back['gray_ncc'], reverse_method=back['method'])
    trials = [dict(direction=direction, **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()})
              for direction, rows in [('forward', forward), ('reverse', reverse)] for row in rows]
    return original, info, trials


def assert_same_view(source, target):
    if (source.tooth_id, source.modality, source.angle_id) != (target.tooth_id, target.modality, target.angle_id):
        raise ValueError('Registration across teeth, modalities or angles is prohibited')
