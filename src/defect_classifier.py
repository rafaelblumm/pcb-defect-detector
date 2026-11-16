from typing import List

import cv2
import numpy as np

from config import DefectClassifierConfigs
from defect import Defect, DefectType

MIN_RING_COMPONENT_AREA = 2
""" Área mínima (pixels) de um componente do anel para ser contado """


class DefectClassifier:
    """ Categorizador de defeitos em PCBs """

    def __init__(self, cfg: DefectClassifierConfigs):
        """
        Cria novo categorizador de defeitos

        Args:
            cfg: Configurações da etapa de categorização de defeitos
        """
        self.cfg = cfg


    def classify(
        self,
        defects: List[Defect],
        reference_img: np.ndarray,
        target_img: np.ndarray
    ) -> List[Defect]:
        """
        Marca como mordida de rato os defeitos que removem cobre na borda da
        trilha, comparando as imagens referência e alvo e o contexto ao redor

        Args:
            defects: Defeitos identificados (com contorno)
            reference_img: Imagem referência em escala de cinza
            target_img: Imagem alvo alinhada em escala de cinza

        Returns:
            Os mesmos defeitos, com categoria e atributos preenchidos
        """
        ref_copper, tgt_copper = self._copper_masks(reference_img, target_img)
        for defect in defects:
            self._extract_features(defect, ref_copper, tgt_copper)
            defect.type = (
                DefectType.MOUSEBITE
                if self._is_mousebite(defect)
                else DefectType.UNKNOWN
            )

        return defects


    def filter(self, defects: List[Defect]) -> List[Defect]:
        """
        Mantém apenas as mordidas de rato

        Args:
            defects: Defeitos categorizados

        Returns:
            Defeitos categorizados como mordida de rato
        """
        return [d for d in defects if d.type == DefectType.MOUSEBITE]


    def _copper_masks(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Segmenta o cobre nas duas imagens usando o limiar de Otsu da referência.
        O cobre é a classe de menor área da referência
        Args:
            reference_img: Imagem referência em escala de cinza
            target_img: Imagem alvo em escala de cinza

        Returns:
            Máscaras booleanas de cobre da referência e do alvo
        """
        threshold, _ = cv2.threshold(
            reference_img,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        ref_bright = reference_img > threshold
        tgt_bright = target_img > threshold
        copper_is_bright = ref_bright.mean() < 0.5
        if copper_is_bright:
            return ref_bright, tgt_bright

        return ~ref_bright, ~tgt_bright


    def _extract_features(
        self,
        defect: Defect,
        ref_copper: np.ndarray,
        tgt_copper: np.ndarray
    ):
        """
        Calcula `copper_delta`, `edge_contact` e `ring_copper_components`
        do defeito

        Args:
            defect: Defeito a analisar
            ref_copper: Máscara de cobre da referência
            tgt_copper: Máscara de cobre do alvo
        """
        band = self.cfg.edge_band_px
        pad = band + 2
        box = defect.box
        height, width = ref_copper.shape
        x0, y0 = max(box.x - pad, 0), max(box.y - pad, 0)
        x1 = min(box.x + box.width + pad, width)
        y1 = min(box.y + box.height + pad, height)

        blob = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.drawContours(
            blob,
            [defect.contour - np.array([x0, y0])],
            -1,
            255,
            cv2.FILLED
        )
        blob = blob > 0
        ref = ref_copper[y0:y1, x0:x1]
        tgt = tgt_copper[y0:y1, x0:x1]

        defect.copper_delta = float(tgt[blob].mean() - ref[blob].mean())

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * band + 1, 2 * band + 1)
        )
        dilated = cv2.dilate(blob.astype(np.uint8), kernel) > 0
        ring = dilated & ~blob
        if not ring.any():
            defect.edge_contact = 0.0
            defect.ring_copper_components = 0
            return

        defect.edge_contact = float((~ref[ring]).mean())
        defect.ring_copper_components = self._count_components(ring & ref)


    def _count_components(self, mask: np.ndarray) -> int:
        """
        Conta componentes conectados (8-vizinhança) relevantes de uma máscara

        Args:
            mask: Máscara booleana

        Returns:
            Número de componentes com área mínima
        """
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8
        )
        areas = stats[1:, cv2.CC_STAT_AREA]

        return int((areas >= MIN_RING_COMPONENT_AREA).sum())


    def _is_mousebite(self, defect: Defect) -> bool:
        """
        Verifica se um defeito de cobre removido na borda da trilha, que não
        a interrompe, tem dimensões e forma de mordida de rato

        Args:
            defect: Defeito com atributos calculados

        Returns:
            `True` se for mordida de rato
        """
        return (
            defect.copper_delta <= -self.cfg.min_copper_delta
            and defect.edge_contact >= self.cfg.min_edge_contact
            and defect.ring_copper_components < 2
            and self.cfg.mousebite_min_area <= defect.area <= self.cfg.mousebite_max_area
            and defect.edge_contact <= self.cfg.mousebite_max_edge_contact
            and defect.circularity <= self.cfg.mousebite_max_circularity
        )
