"""Common oriented sRGB decoding, three independent affine fits, and saved-PNG colour QC."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageOps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from skimage.color import deltaE_ciede2000, rgb2lab

from ..exposure import linear_to_srgb, srgb_to_linear

CHANNELS = "RGB"
LINEAR = srgb_to_linear(np.arange(256))
RAW_MODE = "raw_rgb_srgb_exif_v1"
CORRECTED_MODE = "root_huber_linear_srgb_png_v1"


class FrameIssue(ValueError):
    """Expected frame-level technical failure with a stable reporting code."""

    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(detail or code)


def decode(path, corrected=False):
    """Return the original field in display orientation and its colour provenance."""
    with Image.open(path) as source:
        exif = source.getexif()
        orientation = int(exif.get(274, 1))
        exif_space = exif.get_ifd(34665).get(40961)
        info = dict(native_width=source.width, native_height=source.height, source_mode=source.mode,
                    orientation=orientation, exif_color_space=exif_space, icc_sha256="")
        icc = source.info.get("icc_profile")
        if corrected and (source.format != "PNG" or source.mode != "RGB" or orientation != 1):
            raise FrameIssue("invalid_corrected_png")
        oriented = ImageOps.exif_transpose(source)
        if icc:
            info["icc_sha256"] = sha256(icc).hexdigest()
            try:
                profile = ImageCms.ImageCmsProfile(BytesIO(icc))
                image = ImageCms.profileToProfile(oriented, profile, ImageCms.createProfile("sRGB"), outputMode="RGB")
            except (OSError, ValueError, ImageCms.PyCMSError) as exc:
                raise FrameIssue("unsupported_color_profile", str(exc)) from exc
            interpretation = "icc_to_srgb"
        elif corrected or source.info.get("srgb") is not None:
            image, interpretation = oriented.convert("RGB"), "encoded_srgb"
        elif exif_space not in (None, 1, 65535) or source.mode not in ("RGB", "L", "RGBA", "P", "YCbCr"):
            raise FrameIssue("unsupported_color_profile", f"{source.mode}, EXIF {exif_space}, no ICC")
        elif exif_space == 65535 and exif.get_ifd(40965).get(1) in ("R03", b"R03"):
            raise FrameIssue("unsupported_color_profile", "Adobe RGB without a usable profile")
        else:
            image = oriented.convert("RGB")
            interpretation = "exif_srgb" if exif_space == 1 else "assumed_srgb"
        pixels = np.asarray(image).copy()
    info.update(color_interpretation=interpretation, width=image.width, height=image.height,
                grayscale_content=bool(np.array_equal(pixels[...,0],pixels[...,1]) and
                                       np.array_equal(pixels[...,1],pixels[...,2])))
    return pixels, info


def pixel_hash(rgb):
    digest = sha256(str((rgb.shape[1], rgb.shape[0])).encode("ascii"))
    digest.update(np.ascontiguousarray(rgb).data)
    return digest.hexdigest()


def fit_channels(x, y, config):
    if len(x) < config["min_fit_tiles"]:
        raise FrameIssue("insufficient_fit_tiles")
    spans = np.stack([np.diff(np.percentile(z, [10,90], axis=0), axis=0)[0] for z in (x, y)])
    if not np.isfinite(spans).all() or np.any(spans < config["min_linear_span"]):
        raise FrameIssue("insufficient_channel_span")
    parameters = []
    for channel in range(3):
        reg = HuberRegressor(**{k:config[k] for k in ("epsilon","alpha","max_iter","tol")})
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                reg.fit(x[:, channel, None], y[:, channel])
            except (ValueError, ConvergenceWarning, FloatingPointError) as exc:
                raise FrameIssue("huber_not_converged", f"{CHANNELS[channel]}: {exc}") from exc
        a, b = float(reg.coef_[0]), float(reg.intercept_)
        if not np.isfinite([a, b]).all() or a <= 0:
            raise FrameIssue("invalid_huber_parameters", CHANNELS[channel])
        residual = y[:,channel] - (a*x[:,channel]+b)
        parameters.append((a, b, int(reg.n_iter_), float(np.median(np.abs(residual)))))
    return parameters


def apply_rgb(rgb, slopes, intercepts, crown):
    """Use one lookup per native 8-bit sample; count excursions before clipping/rounding."""
    mapped = LINEAR[:,None] * np.asarray(slopes) + np.asarray(intercepts)
    encoded = np.rint(linear_to_srgb(np.clip(mapped, 0, 1))).astype(np.uint8)
    output = np.empty_like(rgb)
    outside = np.zeros(rgb.shape[:2],bool)
    interior = np.ones_like(outside)
    stats = {}
    for i, c in enumerate(CHANNELS):
        codes = rgb[...,i]
        output[...,i] = encoded[codes,i]
        lower, upper = mapped[:,i] < 0, mapped[:,i] > 1
        histogram = np.bincount(codes.ravel(), minlength=256)
        present = histogram > 0
        stats.update({f"preclip_min_{c}":float(mapped[present,i].min()),
                      f"preclip_max_{c}":float(mapped[present,i].max()),
                      f"below_zero_{c}":int(histogram[lower].sum()), f"above_one_{c}":int(histogram[upper].sum())})
        outside |= (lower | upper)[codes]
        interior &= (codes > 0) & (codes < 255)
    stats.update(whole_pixels=outside.size, whole_outside_px=int(outside.sum()),
                 whole_outside_pct=float(100*outside.mean()))
    if crown is not None and crown.any():
        n = int(crown.sum())
        new = int((crown & interior & outside).sum())
        stats.update(crown_pixels=n,crown_new_clip_px=new,crown_new_clip_pct=100.0*new/n)
    return output, stats, outside & interior


def save_png(path, rgb):
    from PIL.PngImagePlugin import PngInfo
    metadata = PngInfo()
    metadata.add(b"sRGB", b"\x00")
    Image.fromarray(rgb).save(Path(path), pnginfo=metadata, compress_level=3)


def differences(encoded, target):
    """Encoded sRGB channel medians in 8-bit units to D65/2° Lab, weights 1/1/1."""
    a, b = [rgb2lab(np.asarray(z,float)/255.0, illuminant="D65", observer="2") for z in (encoded,target)]
    return np.column_stack([deltaE_ciede2000(a,b,kL=1,kC=1,kH=1),np.abs(a[:,0]-b[:,0]),
                            np.linalg.norm(a[:,1:]-b[:,1:],axis=1)])
