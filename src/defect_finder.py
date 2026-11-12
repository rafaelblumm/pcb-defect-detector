import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from config import DefectFinderConfigs
from defect import Defect, DefectBox
from utils import save_image

class PCBDefectFinder:
    """ Identificador de defeitos em PCBs """

    def __init__(self, cfg: DefectFinderConfigs):
        """
        Cria novo identificador de defeitos

        Args:
            cfg: Configurações da etapa de identificação de defeitos
        """
        self.cfg = cfg


    def locate_defects(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray
    ) -> List[str]:
        """
        Localiza defeitos comparando a imagem alvo com a imagem referência e
        grava as imagens de resultado no diretório de saída

        Args:
            reference_img: Imagem referência em escala de cinza
            target_img: Imagem alvo alinhada em escala de cinza

        Returns:
            Lista de defeitos identificados
        """
        diff = self._diff(reference_img, target_img)
        threshold = self._threshold(diff)
        defects = self._analyze_components(threshold)
        output_dir = Path(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "output"
        )
        self.save_images(
            output_dir,
            target_img,
            reference_img,
            defects
        )

        return defects


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
            circularity=self._circularity(contour, area)
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


    def save_images(
        self,
        output_dir: Path,
        target_registered: np.ndarray,
        reference: np.ndarray,
        defects: List[Defect]
    ):
        """
        Grava as imagens de resultado:
        - Registrada
        - Diferença
        - Defeitos marcados

        Args:
            output_dir: Diretório onde as imagens serão gravadas
            target_registered: Imagem alvo alinhada
            reference: Imagem referência
            defects: Defeitos a serem marcados na imagem
        """
        print("Gravando resultados")
        output_dir.mkdir(parents=True, exist_ok=True)

        reg_path = output_dir / "registered.png"
        save_image(target_registered, str(reg_path))
        print(f"  Imagem registrada: {reg_path}")

        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference
        tgt_gray = cv2.cvtColor(target_registered, cv2.COLOR_BGR2GRAY) if len(target_registered.shape) == 3 else target_registered
        diff = cv2.absdiff(ref_gray.astype(float), tgt_gray.astype(float))
        diff = (diff * 255.0 / (diff.max() + 1e-8)).astype(np.uint8)
        diff_path = output_dir / "difference.png"
        print(f"  Imagem com diferença: {diff_path}")
        save_image(diff, str(diff_path))

        if len(target_registered.shape) == 2:
            target_rgb = cv2.cvtColor(target_registered, cv2.COLOR_GRAY2BGR)
        else:
            target_rgb = target_registered.copy()

        defects_img = target_rgb.copy()
        for defect in defects:
            box = defect.box
            cv2.rectangle(
                defects_img,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                (0, 0, 255),
                2
            )
        defects_path = output_dir / "defects.png"
        save_image(defects_img, str(defects_path))
        print(f"  Imagem com defeitos: {defects_path}")
