from torch import Tensor
from PIL import Image
from . import _functional_pil as F_pil


def cutout(
    img: Tensor | Image.Image,
    p: float | tuple[float, float],
    fillcolor: int | tuple[int, int, int],
) -> Tensor | Image.Image:
    return F_pil.cutout(img, p, fillcolor)


def d_cutout(
    img: Tensor | Image.Image,
    target: Tensor,
    p: float | tuple[float, float],
    fillcolor: int | tuple[int, int, int],
    mode: int,
) -> Tensor | Image.Image:
    return F_pil.d_cutout(img, target, p, fillcolor, mode)


def gridmask(
    img: Tensor | Image.Image,
    d: int, 
    r:float,
    fillcolor: tuple[int, int, int],
    rotate: bool
) -> Tensor | Image.Image:
    return F_pil.gridmask(img, d, r, fillcolor, rotate)


def d_gridmask(
    img: Tensor | Image.Image,
    target: Tensor,
    mode: int,
    d: int, 
    r:float,
    fillcolor: tuple[int, int, int],
) -> Tensor | Image.Image:
    return F_pil.d_gridmask(img, target, mode, d, r, fillcolor)


def patch_gridmask(
    img: Tensor | Image.Image,
    p: int,
    n: int,
    r: float,
    fillcolor: tuple[int, int, int],
) -> Tensor | Image.Image:
    return F_pil.patch_gridmask(img, p, n, r, fillcolor)
