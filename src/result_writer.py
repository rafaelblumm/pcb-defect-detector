from pathlib import Path
from typing import List

import cv2
import numpy as np

from defect import Defect
from utils import save_image


class ResultWriter:
    """ Gravador das imagens de resultado da detecção de defeitos """

    def __init__(self, output_dir: Path):
        """
        Cria novo gravador de resultados

        Args:
            output_dir: Diretório onde as imagens serão gravadas
        """
        self.output_dir = output_dir


    def save_images(
        self,
        target_registered: np.ndarray,
        reference: np.ndarray,
        diff: np.ndarray,
        threshold: np.ndarray,
        defects: List[Defect],
        classified_defects: List[Defect]
    ):
        """
        Grava as imagens de resultado:
        - Registrada
        - Diferença
        - Defeitos marcados, com a categoria de cada um

        Args:
            target_registered: Imagem alvo alinhada
            reference: Imagem referência
            defects: Defeitos a serem marcados na imagem
        """
        print("Gravando resultados")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        reg_path = self.output_dir / "registered.png"
        save_image(target_registered, str(reg_path))
        print(f"  Imagem registrada: {reg_path}")

        diff_path = self.output_dir / "difference.png"
        print(f"  Imagem com diferença: {diff_path}")
        save_image(diff, str(diff_path))

        threshold_path = self.output_dir / "threshold.png"
        print(f"  Imagem com limiar: {threshold_path}")
        save_image(threshold, str(threshold_path))

        target_rgb = cv2.cvtColor(target_registered, cv2.COLOR_GRAY2BGR)
        defects_img = target_rgb.copy()
        for defect in defects:
            box = defect.box
            self._draw_box(defects_img, defect)
            cv2.putText(
                defects_img,
                defect.type.value,
                (box.x, max(box.y - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )
        defects_path = self.output_dir / "defects.png"
        save_image(defects_img, str(defects_path))
        print(f"  Imagem com defeitos: {defects_path}")

        classified_img = target_rgb.copy()
        for defect in classified_defects:
            self._draw_box(classified_img, defect)
        classified_path = self.output_dir / "classified_defects.png"
        save_image(classified_img, str(classified_path))
        print(f"  Imagem com defeitos classificados: {classified_path}")


    @staticmethod
    def _draw_box(image: np.ndarray, defect: Defect):
        """ Desenha a caixa delimitadora do defeito na imagem """
        box = defect.box
        cv2.rectangle(
            image,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            (0, 0, 255),
            2
        )
