"""Shared frozen ImageNet encoder for recorded white-light input modes."""
from pathlib import Path
import logging

import numpy as np
from PIL import Image, ImageOps

from ..files import sha


def pooled_vectors(net, tensors):
    """Both encoders expose the global-average-pool output via an identity head."""
    import torch
    with torch.inference_mode():
        return net(torch.stack(tensors)).cpu().numpy()

GRAY_MODE = 'whole_frame_exif_gray8_bilinear_long224_center_black_v1'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def native_gray(path):
    """Original project grayscale conversion in native, EXIF-oriented coordinates."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert('L')


def white_input(path, mode=GRAY_MODE):
    """Deterministic complete white-light field, in the orientation of the photograph."""
    if mode == GRAY_MODE:
        photo = native_gray(path)
    else:
        from ..rgb.color import CORRECTED_MODE, RAW_MODE, decode
        if mode not in (RAW_MODE,CORRECTED_MODE):
            raise ValueError('Unknown white-light input mode: '+mode)
        photo = Image.fromarray(decode(path,corrected=mode==CORRECTED_MODE)[0])
    thumb = ImageOps.contain(photo, (224, 224), Image.Resampling.BILINEAR)
    canvas = Image.new(photo.mode, (224, 224), 0)
    canvas.paste(thumb, ((224 - thumb.width) // 2, (224 - thumb.height) // 2))
    return canvas.convert('RGB')


def gray_input(path):
    return white_input(path, GRAY_MODE)


def encoder_weights(c):
    import torch
    from torchvision.models import ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    if c['weights'] != str(weights):
        raise ValueError('Unsupported encoder weights')
    override = c.get('checkpoint')
    checkpoint = Path(override).expanduser().resolve() if override else Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.rsplit('/', 1)[-1]
    if not override and not checkpoint.is_file():
        weights.get_state_dict(progress=True, check_hash=True)
    if sha(checkpoint) != c['weights_sha256']:
        raise ValueError('ImageNet checkpoint SHA-256 differs from the locked protocol')
    return checkpoint, weights.url


def imagenet18(c):
    import torch
    from torchvision.models import resnet18
    from ..rgb.color import CORRECTED_MODE, RAW_MODE
    if c['white_light_mode'] not in (GRAY_MODE, RAW_MODE, CORRECTED_MODE):
        raise ValueError('Unsupported white-light preprocessing')
    checkpoint, url = encoder_weights(c)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    net = resnet18(weights=None)
    net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    net.fc = torch.nn.Identity()
    return net.requires_grad_(False).eval(), checkpoint, url


def white_vectors(paths, net, mode=GRAY_MODE):
    from torchvision.transforms import Compose, Normalize, ToTensor
    normalize = Compose([ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    paths = list(paths)
    chunks = []
    for start in range(0, len(paths), 8):
        tensors = [normalize(white_input(p, mode)) for p in paths[start:start + 8]]
        chunks.append(pooled_vectors(net, tensors))
        completed = start + len(tensors)
        if completed % 128 == 0 or completed == len(paths):
            logging.getLogger(__name__).info("Encoded %d/%d images", completed, len(paths))
    vectors = np.concatenate(chunks)
    if vectors.shape != (len(paths), 512) or not np.isfinite(vectors).all():
        raise ValueError('Invalid frozen ResNet-18 features')
    return vectors


def gray_vectors(paths, net):
    return white_vectors(paths, net, GRAY_MODE)
