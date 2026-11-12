from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """ Carrega imagem de um arquivo """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img


def save_image(image: np.ndarray, output_path: Path):
    """ Salva imagem em um arquivo """
    cv2.imwrite(output_path, image)


def to_gray(img: np.ndarray) -> np.ndarray:
    """ Converte imagem colorida para tons de cinza """
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
