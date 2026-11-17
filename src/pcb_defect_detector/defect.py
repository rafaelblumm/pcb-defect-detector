from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class DefectType(Enum):
    """ Categorias de defeitos em PCBs """
    MOUSEBITE = "mousebite"
    UNKNOWN = "desconhecido"


@dataclass
class DefectBox:
    """ Caixa delimitadora (bounding box) de um defeito """
    x: int
    """ Coordenada X do topo (pixels) """
    y: int
    """ Coordenada Y do topo (pixels) """
    width: int
    """ Largura da caixa (pixels) """
    height: int
    """ Altura da caixa (pixels) """

    @property
    def position(self) -> tuple[int, int]:
        """ Posição (x, y) do topo da caixa """
        return (self.x, self.y)

    @property
    def size(self) -> tuple[int, int]:
        """ Dimensões (largura, altura) da caixa """
        return (self.width, self.height)


@dataclass
class Defect:
    """ Defeito identificado no circuito """
    box: DefectBox
    """ Caixa indicadora do defeito """
    area: float
    """ Área afetada (pixels quadrados) """
    centroid_x: float
    """ Coordenada X da centróide (pixels) """
    centroid_y: float
    """ Coordenada Y da centróide (pixels) """
    circularity: float
    """ Grau de circularidade do defeito """
    contour: Optional[np.ndarray] = field(default=None, repr=False)
    """ Contorno do defeito """
    type: DefectType = DefectType.UNKNOWN
    """ Categoria do defeito """
    copper_delta: float = 0.0
    """ Variação da fração de cobre na região do defeito (alvo - referência),
        entre -1 (cobre removido) e 1 (cobre adicionado) """
    edge_contact: float = 0.0
    """ Fração do anel ao redor do defeito que é substrato na referência """
    ring_copper_components: int = 0
    """ Número de trechos de cobre (na referência) ao redor do defeito """
