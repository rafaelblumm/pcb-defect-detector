from typing import List

import cv2
import numpy as np

from pcb_defect_detector.config import DefectFinderConfigs
from pcb_defect_detector.defect import Defect, DefectBox


class PCBDefectFinder:
    """ Identificador de defeitos em PCBs """

    def __init__(self, cfg: DefectFinderConfigs):
        """
        Cria novo identificador de defeitos

        Args:
            cfg: Configurações da etapa de identificação de defeitos
        """
        self.cfg = cfg
        self.images = {}


    def locate_defects(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray
    ) -> List[Defect]:
        """
        Localiza defeitos comparando a imagem alvo com a imagem referência

        Args:
            reference_img: Imagem referência em escala de cinza
            target_img: Imagem alvo alinhada em escala de cinza

        Returns:
            Lista de defeitos identificados, ainda sem categoria
        """
        diff = self._diff(reference_img, target_img)
        self.images["diff"] = diff.copy()

        threshold = self._threshold(diff)
        self.images["threshold"] = threshold.copy()

        return self._analyze_components(threshold)


    def _diff(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray
    ) -> np.ndarray:
        """
        Calcula diferença entre imagens

        Args:
            reference_img: Imagem referência
            target_img: Imagem alvo

        Returns:
            Imagem de diferença em uint8
        """
        diff = cv2.absdiff(
            reference_img.astype(float),
            target_img.astype(float)
        )
        if diff.max() > 0:
            return (diff * 255.0 / diff.max()).astype(np.uint8)

        return diff.astype(np.uint8)


    def _threshold(self, diff: np.ndarray) -> np.ndarray:
        """
        Limiariza imagem de diferença com algoritmo de limiar te Otsu e aplica
        operações morfológicas em pixels adjacentes de abertura e fechamento
        para remover ruídos

        Args:
            diff: Imagem de diferença

        Returns:
            Imagem binarizada
        """
        _, binary = cv2.threshold(
            diff,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if self.cfg.morphology_kernel_size is None:
            return binary

        kernel_size = self.cfg.morphology_kernel_size
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        for op in {cv2.MORPH_OPEN, cv2.MORPH_CLOSE}:
            binary = cv2.morphologyEx(binary, op, kernel, iterations=1)

        return binary


    def _analyze_components(self, binary: np.ndarray) -> List[Defect]:
        """
        Analisa componentes conectados e identifica defeitos

        Args:
            binary: Imagem analisada

        Returns:
            Lista de defeitos
        """
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        defects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.cfg.min_defect_area <= int(area) <= self.cfg.max_defect_area:
                defects.append(self._build_defect(contour, area))

        return defects


    def _build_defect(self, contour, area: float) -> Defect:
        """
        Cria `Defect` a partir de um contorno de defeito

        Args:
            contour: Contorno do defeito
            area: Área do contorno (pixels quadrados)

        Returns:
            Dados do defeito
        """
        box = self._bounding_box(contour)
        centroid = self._centroid(contour, box)

        return Defect(
            box=box,
            area=area,
            centroid_x=centroid[0],
            centroid_y=centroid[1],
            circularity=self._circularity(contour, area),
            contour=contour
        )


    def _bounding_box(self, contour) -> DefectBox:
        """
        Calcula dimensões da caixa delimitadora do contorno

        Args:
            contour: Contorno do defeito

        Returns:
            Caixa delimitadora do defeito
        """
        x, y, w, h = cv2.boundingRect(contour)
        return DefectBox(x=x, y=y, width=w, height=h)


    def _centroid(self, contour, box: DefectBox) -> tuple[float, float]:
        """
        Calcula a centróide do contorno a partir dos momentos da imagem. Se a
        área for zero, usa o centro da caixa delimitadora

        Args:
            contour: Contorno do defeito
            box: Caixa delimitadora do defeito

        Returns:
            Tupla de (x, y) da centróide em pixels
        """
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return (
                box.position[0] + box.size[0] / 2,
                box.position[1] + box.size[1] / 2
            )

        return (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"]
        )


    def _circularity(self, contour, area: float) -> float:
        """
        Calcula a circularidade do defeito

        Args:
            contour: Contorno do defeito
            area: Área do contorno (pixels quadrados)

        Returns:
            Circularidade entre 0 e 1, ou 0.0 se o perímetro for zero
        """
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            return 4 * np.pi * area / (perimeter ** 2)

        return 0.0
