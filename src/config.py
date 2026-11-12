from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImageRegistrationConfig:
    """ Configurações da etapa de registro translacional das imagens """
    radius: int = 50
    """ Deslocamento máximo da busca """
    step: int = 1
    """ Pulos na sequência de pixels. Quanto maior, mais performance e menos
        precisão """


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
class PCBDetectionPipelineConfigs:
    """ Configurações do pipeline de detecção de defeitos em PCB """
    registration: ImageRegistrationConfig = field(default_factory=ImageRegistrationConfig)
    """ Configurações da etapa de registro translacional das imagens """
    defect: DefectFinderConfigs = field(default_factory=DefectFinderConfigs)
    """ Configurações da etapa de identificação de defeitos """
