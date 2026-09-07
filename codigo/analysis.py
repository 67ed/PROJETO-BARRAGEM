"""Características geométricas de regiões candidatas."""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from codigo import config

logger = logging.getLogger(__name__)


@dataclass
class Regiao:
    """Representa uma região candidata a alteração visual na imagem.
    
    Attributes:
        x: Coordenada x do retângulo delimitador.
        y: Coordenada y do retângulo delimitador.
        largura: Largura do retângulo delimitador.
        altura: Altura do retângulo delimitador.
        area: Área do contorno detectado.
        perimetro: Perímetro do contorno detectado.
        orientacao: Ângulo de orientação da elipse ajustada, se aplicável.
    """
    x: int
    y: int
    largura: int
    altura: int
    area: float
    perimetro: float
    orientacao: float | None

    def para_dict(self) -> dict:
        """Converte a instância para dicionário."""
        return asdict(self)
        
    def __repr__(self) -> str:
        """Representação amigável da região."""
        return (f"Regiao(x={self.x}, y={self.y}, larg={self.largura}, "
                f"alt={self.altura}, area={self.area:.1f})")


def classificar_tamanho_visual(
    regiao: Regiao | None,
    forma_imagem: tuple[int, ...],
) -> dict:
    """Classifica o tamanho na foto; não representa risco estrutural real.
    
    Args:
        regiao: Objeto Regiao detectado, ou None.
        forma_imagem: Tupla contendo (altura, largura, ...) da imagem original.
        
    Returns:
        Dicionário com o código, rótulo e medidas relativas da região.
    """
    if regiao is None:
        return {
            "codigo": "nenhuma",
            "rotulo": "nenhuma região",
            "area_percentual": 0.0,
            "comprimento_px": 0,
            "comprimento_percentual": 0.0,
        }

    altura, largura = forma_imagem[:2]
    area_relativa = regiao.area / max(1, altura * largura)
    comprimento_px = max(regiao.largura, regiao.altura)
    comprimento_relativo = comprimento_px / max(1, altura, largura)

    if (
        area_relativa < config.AREA_RELATIVA_MEDIA
        and comprimento_relativo < config.COMPRIMENTO_RELATIVO_MEDIO
    ):
        codigo, rotulo = "pequena", "pequena alteração visual"
    elif (
        area_relativa < config.AREA_RELATIVA_GRANDE
        and comprimento_relativo < config.COMPRIMENTO_RELATIVO_GRANDE
    ):
        codigo, rotulo = "media", "alteração visual média"
    else:
        codigo, rotulo = "grande", "alteração visual extensa"

    return {
        "codigo": codigo,
        "rotulo": rotulo,
        "area_percentual": round(area_relativa * 100, 3),
        "comprimento_px": int(comprimento_px),
        "comprimento_percentual": round(comprimento_relativo * 100, 2),
    }


def regioes_da_mascara(
    mascara_binaria: np.ndarray,
    area_minima: float = config.AREA_MINIMA_REGIAO,
) -> list[Regiao]:
    """Extrai regiões (contornos) de uma máscara binária e computa características.
    
    Args:
        mascara_binaria: Máscara binária onde pixels > 0 indicam objetos.
        area_minima: Área mínima para considerar um contorno válido.
        
    Returns:
        Lista de objetos Regiao detectados, ordenada decrescente por área.
    """
    contornos, _ = cv2.findContours(
        mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    saida: list[Regiao] = []
    for contorno in contornos:
        area = float(cv2.contourArea(contorno))
        if area < area_minima:
            continue
        x, y, w, h = cv2.boundingRect(contorno)
        perimetro = float(cv2.arcLength(contorno, True))
        orientacao = None
        if len(contorno) >= 5:
            try:
                _, _, angulo = cv2.fitEllipse(contorno)
                orientacao = float(angulo)
            except cv2.error as err:
                logger.debug("Falha ao ajustar elipse: %s", err)
                orientacao = None
                
        saida.append(
            Regiao(
                x=int(x),
                y=int(y),
                largura=int(w),
                altura=int(h),
                area=area,
                perimetro=perimetro,
                orientacao=orientacao,
            )
        )
    saida.sort(key=lambda r: r.area, reverse=True)
    return saida


def iou_mascaras(predita: np.ndarray, referencia: np.ndarray) -> float:
    """Calcula o IoU (Intersection over Union) entre duas máscaras binárias.
    
    Args:
        predita: Máscara de predição (pixels > 0).
        referencia: Máscara de referência (pixels > 0).
        
    Returns:
        Valor do IoU (entre 0.0 e 1.0).
    """
    if predita.shape[:2] != referencia.shape[:2]:
        referencia = cv2.resize(
            referencia, (predita.shape[1], predita.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    p = predita > 0
    r = referencia > 0
    inter = np.logical_and(p, r).sum()
    uniao = np.logical_or(p, r).sum()
    if uniao == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / uniao)


def sobrepor(
    imagem_bgr: np.ndarray, 
    mascara: np.ndarray, 
    cor: tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """Cria uma visualização sobrepondo a máscara à imagem original.
    
    Args:
        imagem_bgr: Imagem original colorida.
        mascara: Máscara binária a ser sobreposta.
        cor: Cor em BGR para a sobreposição.
        
    Returns:
        Imagem com o overlay aplicado.
    """
    vis = imagem_bgr.copy()
    if vis.shape[:2] != mascara.shape[:2]:
        mascara = cv2.resize(mascara, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    camada = vis.copy()
    camada[mascara > 0] = cor
    misturada = cv2.addWeighted(vis, 0.65, camada, 0.35, 0)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(misturada, contornos, -1, cor, 1)
    return misturada
