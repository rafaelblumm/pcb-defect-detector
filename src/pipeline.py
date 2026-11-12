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
        """
        Cria novo pipeline de detecção de defeitos

        Args:
            cfg: Configurações das etapas do pipeline
        """
        self.cfg = cfg


    def process(self, reference_path: Path, target_path: Path):
        """
        Executa o pipeline completo:
        1. Carrega imagens
        2. Transforma imagens em escala de cinza
        3. Alinha imagens
        4. Calcula mapa de diferenças
        5. Indica ocorrências de defeitos na imagem de teste

        Args:
            reference_path: Caminho da imagem matriz (referência)
            target_path: Caminho da imagem resultado (alvo)
        """
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
        """
        Carrega imagens e converte para tons de cinza

        Args:
            reference_path: Caminho da imagem matriz (referência)
            target_path: Caminho da imagem resultado (alvo)

        Returns:
            Tupla de imagem referência e imagem alvo em escala de cinza
        """
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
        """
        Alinha a imagem alvo à imagem referência por registro translacional

        Args:
            reference_img: Imagem referência em escala de cinza
            target_img: Imagem alvo em escala de cinza

        Returns:
            Imagem alvo alinhada com a referência
        """
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
        """
        Identifica defeitos na imagem alvo comparando com a imagem referência

        Args:
            reference_img: Imagem referência em escala de cinza
            target_img: Imagem alvo alinhada em escala de cinza

        Returns:
            Lista de defeitos identificados
        """
        print("Buscando defeitos")
        defects = PCBDefectFinder(self.cfg.defect).locate_defects(
            reference_img,
            target_img
        )
        print(f"Identificados {len(defects)} defeitos")

        return defects
