from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """
    Carrega imagem de um arquivo

    Args:
        url: URL completa da imagem
    Returns:
        Imagem carregada
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img


def load_image_from_url(url: str) -> np.ndarray:
    """
    Carrega imagem de uma URL

    Args:
        url: URL completa da imagem
    Returns:
        Imagem carregada
    """
    req = urlopen('https://i.imgur.com/DrjBucJ.png')
    image = np.asarray(bytearray(req.read()), dtype=np.uint8)
    return image

    # image = cv2.imdecode(image, -1)


def save_image(image: np.ndarray, output_path: Path):
    """
    Grava imagem em um arquivo

    Args:
        image: Imagem carregada
        output_path: Caminho do arquivo
    """
    cv2.imwrite(output_path, image)


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    Converte imagem colorida para tons de cinza

    Args:
        image: Imagem carregada
    """
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
