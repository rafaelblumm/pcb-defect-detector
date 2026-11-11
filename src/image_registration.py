import cv2
import numpy as np

from config import ImageRegistrationConfig


class ImageRegistrator:
    """
    Realiza registro translacional por busca local utiliando algoritmo de
    soma da diferença dos quadrados (SSD - Sum of Squared Differences)
    """

    def __init__(self, cfg: ImageRegistrationConfig):
        """
        Cria novo registrador de imagens

        Args:
            cfg: Configurações do registrador
        """
        self.cfg = cfg


    def register(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray
    ):
        """
        Realiza registro translacional da imagem resultado com a imagem matriz.

        Args:
            reference_img: Imagem referência
            target_img: Imagem alvo

        Returns:
            Tupla de (dx, dy, metric): Deslocamento horizontal, deslocamento
            vertical e métrica de diferença entre imagens
        """
        if reference_img.shape != target_img.shape:
            raise "Imagem de referência e imagem alvo possuem tamanhos " \
                  "diferentes"

        radius = self.cfg.radius
        displacements = [
            (dx, dy)
            for dx in range(-radius, radius + 1, self.cfg.step)
            for dy in range(-radius, radius + 1, self.cfg.step)
        ]

        best_displacement = (0, 0)
        best_metric = float('inf')

        for dx, dy in displacements:
            metric = self._compute_ssd_metric(
                reference_img,
                target_img,
                dx,
                dy
            )
            if metric < best_metric:
                best_metric = metric
                best_displacement = (dx, dy)

        return best_displacement[0], best_displacement[1], best_metric


    def _compute_ssd_metric(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray,
        dx: int,
        dy: int,
    ):
        """
        Calcula métrica de diferença entre regiões da imagem

        Args:
            reference_img: Imagem referência
            target_img: Imagem alvo
            dx: Deslocamento horizontal
            dy: Deslocamento vertical

        Returns:
            Métrica de diferença entre regiões da imagem
        """
        height, width = reference_img.shape

        # Região de interesse baseado no deslocamento
        if dx >= 0:
            ref_x_start, ref_x_end = dx, width
            tgt_x_start, tgt_x_end = 0, width - dx
        else:
            ref_x_start, ref_x_end = 0, width + dx
            tgt_x_start, tgt_x_end = -dx, width

        if dy >= 0:
            ref_y_start, ref_y_end = dy, height
            tgt_y_start, tgt_y_end = 0, height - dy
        else:
            ref_y_start, ref_y_end = 0, height + dy
            tgt_y_start, tgt_y_end = -dy, height

        ref_roi = reference_img[ref_y_start:ref_y_end, ref_x_start:ref_x_end]
        tgt_roi = target_img[tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end]

        if ref_roi.size == 0 or tgt_roi.size == 0:
            return float('inf')

        return np.sum((ref_roi.astype(float) - tgt_roi.astype(float)) ** 2)


def apply_transformation(
        image: np.ndarray,
        dx: int,
        dy: int
    ) -> np.ndarray:
        """
        Aplica transformação translacional na imagem

        Args:
            image: Imagem de entrada
            dx: Deslocamento horizontal
            dy: Deslocamento vertical

        Returns:
            Imagem tranformada
        """
        return cv2.warpAffine(
            image,
            np.float32([[1, 0, dx], [0, 1, dy]]),
            (image.shape[0], image.shape[1])
        )
