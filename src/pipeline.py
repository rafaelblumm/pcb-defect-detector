from contextlib import contextmanager
from time import perf_counter
from typing import Dict, Iterator, List

import numpy as np

from pathlib import Path

from config import PCBDetectionPipelineConfigs
from defect import Defect
from defect_classifier import DefectClassifier
from defect_finder import PCBDefectFinder
from image_registration import ImageRegistrator, apply_transformation
from result_writer import ResultSet, ResultWriter
from utils import load_image, to_gray

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
""" Diretório onde as imagens de resultado são gravadas """


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


    def load_and_process(
        self,
        reference_path: Path,
        target_path: Path
    ) -> ResultSet:
        """
        Carrega imagens de arquivos e executa o pipeline completo

        Args:
            reference_path: Caminho da imagem matriz
            target_path: Caminho da imagem resultado

        Returns:
            Imagens resultantes do pipeline
        """
        timings: Dict[str, float] = {}
        with self._timed(timings, "Carregamento e escala de cinza"):
            ref, target = self._load_images(
                reference_path,
                target_path
            )
        return self._run(ref, target, timings)


    def process(self, ref: np.ndarray, target: np.ndarray) -> ResultSet:
        """
        Executa o pipeline completo:
        1. Transforma imagens em escala de cinza
        2. Alinha imagens
        3. Calcula mapa de diferenças e identifica defeitos
        4. Categoriza os defeitos e mantém apenas as categorias de interesse
        5. Indica ocorrências de defeitos na imagem de teste

        Args:
            ref: Imagem matriz
            target: Imagem resultado

        Returns:
            Imagens resultantes do pipeline, com o tempo de cada etapa
        """
        return self._run(ref, target, {})


    def _run(
        self,
        ref: np.ndarray,
        target: np.ndarray,
        timings: Dict[str, float]
    ) -> ResultSet:
        """
        Executa as etapas do pipeline, registrando o tempo de cada uma

        Args:
            ref: Imagem matriz
            target: Imagem resultado
            timings: Tempos (em segundos) de etapas já executadas. É
                complementado com as etapas executadas aqui

        Returns:
            Imagens resultantes do pipeline, com o tempo de cada etapa
        """
        with self._timed(timings, "Registro de imagens"):
            registered_image = self._register_images(ref, target)
        with self._timed(timings, "Identificação de defeitos"):
            defects, diff, threshold = self._identify_defects(
                ref,
                registered_image
            )
        with self._timed(timings, "Classificação de defeitos"):
            classified_defects = self._classify_defects(
                ref,
                registered_image,
                defects
            )
        with self._timed(timings, "Gravação de resultados"):
            result_set = ResultWriter(OUTPUT_DIR).save_images(
                ref,
                registered_image,
                diff,
                threshold,
                defects,
                classified_defects
            )
        result_set.timings = timings
        print("Processo finalizado")

        return result_set


    @staticmethod
    @contextmanager
    def _timed(timings: Dict[str, float], stage: str) -> Iterator[None]:
        """ Registra em `timings` o tempo de execução, em segundos, da etapa """
        start = perf_counter()
        try:
            yield
        finally:
            timings[stage] = perf_counter() - start


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
        defect_finder = PCBDefectFinder(self.cfg.defect)
        defects = defect_finder.locate_defects(
            reference_img,
            target_img
        )
        print(f"  Identificados {len(defects)} defeitos")

        return (
            defects,
            defect_finder.images["diff"],
            defect_finder.images["threshold"]
        )


    def _classify_defects(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray,
        defects: List[Defect]
    ) -> List[Defect]:
        """
        Categoriza os defeitos e mantém apenas as categorias de interesse

        Args:
            reference_img: Imagem referência em escala de cinza
            target_img: Imagem alvo alinhada em escala de cinza
            defects: Defeitos identificados

        Returns:
            Lista de defeitos nas categorias de interesse
        """
        print("Categorizando defeitos")
        classifier = DefectClassifier(self.cfg.classifier)
        classifier.classify(defects, reference_img, target_img)
        kept = classifier.filter(defects)
        print(f"  Mantidos {len(kept)} de {len(defects)} defeitos")

        return kept
