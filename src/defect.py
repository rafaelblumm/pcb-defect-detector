from dataclasses import dataclass


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


@dataclass
class DefectBox:
    x: int
    """ Coordenada X do topo (pixels) """
    y: int
    """ Coordenada Y do topo (pixels) """
    width: int
    """ Largura da caixa (pixels) """
    height: int
    """ Altura da caixa (pixels) """
