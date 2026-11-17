from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RegistrationAlgorithm(str, Enum):
    """ Algoritmos disponíveis para registro translacional das imagens """
    SSD = "ssd"
    """ Busca local pela soma da diferença dos quadrados """
    FFT_CROSS_CORRELATION = "fft_cross_correlation"
    """ Correlação cruzada calculada via algoritmo Fast Fourier Transform """


@dataclass
class ImageRegistrationConfig:
    """ Configurações da etapa de registro translacional das imagens """
    algorithm: RegistrationAlgorithm = RegistrationAlgorithm.SSD
    """ Algoritmo utilizado no registro """
    radius: int = 50
    """ Deslocamento máximo da busca """
    step: int = 1
    """ Pulos na sequência de pixels. Quanto maior, mais performance e menos
        precisão. Utilizado apenas pelo algoritmo SSD """


@dataclass
class DefectFinderConfigs:
    """ Configurações da etapa de identificação de defeitos """
    morphology_kernel_size: Optional[int] = 5
    """ Tamanho do kernel para limpeza do ruído da limiarização """
    min_defect_area: int = 10
    """ Área mínima do defeito """
    max_defect_area: int = 10_000
    """ Área máxima do defeito """


@dataclass
class DefectClassifierConfigs:
    """ Configurações da etapa de categorização de defeitos """
    edge_band_px: int = 7
    """ Largura do anel ao redor do defeito usado para analisar o contexto """
    min_copper_delta: float = 0.3
    """ Variação mínima de cobre (em módulo) para considerar cobre removido """
    min_edge_contact: float = 0.01
    """ Fração mínima de substrato no anel para o defeito tocar a borda da
        trilha """
    mousebite_min_area: int = 10
    """ Área mínima de uma mordida de rato """
    mousebite_max_area: int = 1_500
    """ Área máxima de uma mordida de rato """
    mousebite_max_edge_contact: float = 0.45
    """ Fração máxima de substrato no anel de uma mordida de rato"""
    mousebite_max_circularity: float = 0.85
    """ Circularidade máxima de uma mordida de rato (irregular) """


@dataclass
class PCBDetectionPipelineConfigs:
    """ Configurações do pipeline de detecção de defeitos em PCB """
    registration: ImageRegistrationConfig = field(
        default_factory=ImageRegistrationConfig
    )
    """ Configurações da etapa de registro translacional das imagens """
    defect: DefectFinderConfigs = field(default_factory=DefectFinderConfigs)
    """ Configurações da etapa de identificação de defeitos """
    classifier: DefectClassifierConfigs = field(
        default_factory=DefectClassifierConfigs
    )
    """ Configurações da etapa de categorização de defeitos """
