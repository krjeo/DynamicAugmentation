import numpy as np
from PIL import Image
from torch import Tensor
import math


def cutout(
    img: Image.Image,
    p: float | tuple[float, float],
    fillcolor: int | tuple[int, int]
) -> Image.Image:
    
    if not isinstance(img, Image.Image):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    
    if not isinstance(p, float):
        p = np.random.uniform(p[0], p[1])
    
    width, height = img.size
    img_area = width * height
    
    min_width = math.ceil((0.35 + 0.65 * p) * width)
    max_width = math.ceil((0.65 + 0.35 * p) * width)
    w = np.random.randint(min_width, max_width)
    h = math.ceil(p * img_area / w)
    
    img_array = np.array(img)
    
    if isinstance(fillcolor, int):
        fillcolor = [fillcolor, fillcolor, fillcolor]
    
    i = np.random.randint(0, width - w + 1)
    j = np.random.randint(0, height - h + 1)
    for c in range(3):
        img_array[i:i+w, j:j+h, c] = fillcolor[c]
    
    return Image.fromarray(np.uint8(img_array))


def d_cutout(
    img: Image.Image,
    target: Tensor,
    p: float | tuple[float, float],
    fillcolor: int | tuple[int, int],
    mode: int,
) -> Image.Image:
    
    if not isinstance(img, Image.Image):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    
    if not isinstance(p, float):
        p = np.random.uniform(p[0], p[1])
    
    width, height = img.size
    img_area = width * height
    
    min_width = math.ceil((0.35 + 0.65 * p) * width)
    max_width = math.ceil((0.65 + 0.35 * p) * width)
    w = np.random.randint(min_width, max_width)
    h = math.ceil(p * img_area / w)
    
    img_array = np.array(img)
    
    if isinstance(fillcolor, int):
        fillcolor = [fillcolor, fillcolor, fillcolor]
    
    i = np.random.randint(0, width - w + 1)
    j = np.random.randint(0, height - h + 1)
    for c in range(3):
        img_array[i:i+w, j:j+h, c] = fillcolor[c]
    
    if mode == 0:
        pass
    elif mode == 1:
        target *= (1 - p)
    else:
        # More functions will be added
        pass
    
    return Image.fromarray(np.uint8(img_array)), target


def gridmask(
    img: Image.Image,
    d: int | tuple[int, int], 
    r: float | tuple[float, float],
    fillcolor: int | tuple[int, int, int],
    rotate: bool,
) -> Image.Image:

    if not isinstance(img, Image.Image):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    
    if not isinstance(d, int):
        d = np.random.randint(d[0], d[1])
    if not isinstance(r, float):
        r = np.random.uniform(r[0], r[1])
    
    width, height = img.size
    m1 = (max(width, height) // d + 1) * d

    l = math.ceil(d * r)
    m2 = math.ceil(1.5 * m1)
    
    theta_w = np.random.randint(d)
    theta_h = np.random.randint(d)
    
    mask = np.zeros((m2, m2), np.float32)
    
    if theta_w < (d - l):
        for i in range(0, ((m2 - theta_w) // d) + 1):
            start = theta_w + d * i
            end = min(start + l, m2)
            mask[start:end, :] = 1
    else:
        for i in range(0, ((m2 + theta_w) // d) + 1):
            start = d * (i - 1) + theta_w
            end = min(max(0, start + l), m2)
            mask[start:end, :] = 1
    if theta_h < (d - l):
        for i in range(0, ((m2 - theta_h) // d) + 1):
            start = theta_h + d * i
            end = min(start + l, m2)
            mask[:, start:end] = 1
    else:
        for i in range(0, ((m2 + theta_h) // d) + 1):
            start = d * (i - 1) + theta_h
            end = min(max(0, start + l), m2)
            mask[:, start:end] = 1
    
    def center_crop(image, crop_width, crop_height):
        width, height = image.size
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        return image.crop((left, top, right, bottom))
    
    mask = Image.fromarray(np.uint8(mask))
    if rotate:
        mask = mask.rotate(np.random.randint(-90, 90))
    mask = center_crop(mask, width, height)
    
    img_array = np.array(img)
    mask_array = np.array(mask)
    
    if isinstance(fillcolor, int):
        fillcolor = [fillcolor, fillcolor, fillcolor]
    for c in range(3):
        img_array[:, :, c][mask_array == 0] = fillcolor[c]
        
    return Image.fromarray(np.uint8(img_array))


def d_gridmask(
    img: Image.Image,
    target: Tensor,
    mode: int,
    d: int | tuple[int, int], 
    r: float | tuple[float, float],
    fillcolor: tuple[int, int, int],
) -> tuple[Image.Image, Tensor]:

    if not isinstance(img, Image.Image):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    
    if not isinstance(d, int):
        d = np.random.randint(d[0], d[1])
    if not isinstance(r, float):
        r = np.random.uniform(r[0], r[1])
    
    width, height = img.size
    m1 = (max(width, height) // d + 1) * d

    l = math.ceil(d * r)
    m2 = math.ceil(1.5 * m1)
    
    theta_w = np.random.randint(d)
    theta_h = np.random.randint(d)
    
    mask = np.zeros((m2, m2), np.float32)
    
    if theta_w < (d - l):
        for i in range(0, ((m2 - theta_w) // d) + 1):
            start = theta_w + d * i
            end = min(start + l, m2)
            mask[start:end, :] = 1
    else:
        for i in range(0, ((m2 + theta_w) // d) + 1):
            start = d * (i - 1) + theta_w
            end = min(max(0, start + l), m2)
            mask[start:end, :] = 1
    if theta_h < (d - l):
        for i in range(0, ((m2 - theta_h) // d) + 1):
            start = theta_h + d * i
            end = min(start + l, m2)
            mask[:, start:end] = 1
    else:
        for i in range(0, ((m2 + theta_h) // d) + 1):
            start = d * (i - 1) + theta_h
            end = min(max(0, start + l), m2)
            mask[:, start:end] = 1
    
    def center_crop(image, crop_width, crop_height):
        width, height = image.size
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        return image.crop((left, top, right, bottom))
    
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(np.random.randint(180))
    mask = center_crop(mask, width, height)
    
    img_array = np.array(img)
    mask_array = np.array(mask)
    
    if isinstance(fillcolor, int):
        fillcolor = [fillcolor, fillcolor, fillcolor]
    for c in range(3):
        img_array[:, :, c][mask_array == 0] = fillcolor[c]
    
    if mode == 0:
        pass
    elif mode == 1:
        ratio = 2 * r - r ** 2
        target *= ratio
    elif mode == 2:
        # More functions will be added
        m = (d * (1 - r)) ** 2 / (width * height)
        ratio = 2 * r - r ** 2
        temp = 1.5 / (1 + math.exp(-26*(m-0.2)))
        target *= (ratio) / (temp + ratio)

    return Image.fromarray(np.uint8(img_array)), target


def patch_gridmask(
    img: Image.Image,
    p: int,
    n: int | tuple[int, int],
    r: float | tuple[float, float],
    fillcolor: tuple[int, int, int],
) -> Image.Image:

    if not isinstance(img, Image.Image):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")
    
    if not isinstance(n, int):
        n = np.random.randint(n[0], n[1])
    if not isinstance(r, float):
        r = np.random.uniform(r[0], r[1])
    
    width, height = img.size
    
    masking_size = math.ceil(p * (1 - r))
    d = math.ceil(masking_size / (n + 1))
    l = math.ceil((r * p) / (2 * n))
    
    mask = np.zeros((p, p), np.float32)
    for i in range(n):
        start = 2 * l * i + d * (i + 1)
        end = start + 2 * l
        mask[start:end, :] = 1
    for i in range(n):
        start = 2 * l * i + d * (i + 1)
        end = start + 2 * l
        mask[:, start:end] = 1
    
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(np.random.randint(90), expand=True, fillcolor=1)
    mask_size, _ = mask.size
    
    img_array = np.array(img)
    mask_array = np.array(mask)
    
    if isinstance(fillcolor, int):
        fillcolor = [fillcolor, fillcolor, fillcolor]
        
    i = np.random.randint(0, width - mask_size + 1)
    j = np.random.randint(0, height - mask_size + 1)
    for c in range(3):
        img_array[i:i+mask_size, j:j+mask_size, c][mask_array == 0] = fillcolor[c]
    
    return Image.fromarray(np.uint8(img_array))

