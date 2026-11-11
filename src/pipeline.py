from typing import List

import numpy as np

from pathlib import Path

from config import PCBDetectionPipelineConfigs
from defect import Defect
from defect_finder import PCBDefectFinder
from image_registration import ImageRegistrator, apply_transformation
from utils import load_image, to_gray


class PCBDetectionPipeline:
    """ Classe de pipeline para detecção de defeitos em PCB """

    def __init__(
        self,
        cfg: PCBDetectionPipelineConfigs = PCBDetectionPipelineConfigs()
    ):
        self.cfg = cfg


    def process(self, reference_path: Path, target_path: Path):
        ref, target = self._load_images(
            reference_path,
            target_path
        )
        registered_image = self._register_images(ref, target)
        defects = self._identify_defects(ref, registered_image)
        print("Processo finalizado")


    def _load_images(
        self,
        reference_path: Path,
        target_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        print("Carregando imagens")
        print(f"  Matriz: {reference_path}")
        reference_img = load_image(reference_path)

        print(f"  Resultado: {target_path}")
        target_img = load_image(target_path)

        print("Convertendo imagens para escala de cinza")
        reference_img = to_gray(reference_img)
        target_img = to_gray(target_img)

        return reference_img, target_img


    def _register_images(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray
    ) -> np.ndarray:
        print(f"Realizando registro translacional por busca local com SSD")
        dx, dy, metric = ImageRegistrator(self.cfg.registration).register(
            reference_img,
            target_img
        )
        print(f"  Resultado:")
        print(f"    dx={dx}")
        print(f"    dy={dy}")
        print(f"    métrica={metric}")
        print(f"Aplicando transformações para alinhamento das imagens")

        return apply_transformation(target_img, dx, dy)


    def _identify_defects(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray
    ) -> List[Defect]:
        print("Buscando defeitos")
        defects = PCBDefectFinder(self.cfg.defect).locate_defects(
            reference_img,
            target_img
        )
        print(f"Identificados {len(defects)} defeitos")

        return defects
