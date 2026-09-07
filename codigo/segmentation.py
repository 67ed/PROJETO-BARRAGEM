"""Segmentação clássica (Canny / Otsu + morfologia). Sem treinar rede."""

from __future__ import annotations

import logging
import cv2
import numpy as np
from skimage.filters import frangi

from codigo import config

logger = logging.getLogger(__name__)


def _kernel(tamanho: int = config.KERNEL_MORFOLOGIA) -> np.ndarray:
    """
    Cria um kernel morfológico elíptico.
    
    Args:
        tamanho (int): O tamanho do kernel (diâmetro).
        
    Returns:
        np.ndarray: O kernel morfológico criado.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho, tamanho))


def limpar_componentes(mascara: np.ndarray) -> np.ndarray:
    """
    Descarta grãos isolados e blocos anormalmente grandes de textura/sombra.
    
    Args:
        mascara (np.ndarray): A máscara binária de entrada.
        
    Returns:
        np.ndarray: A máscara limpa de ruídos, contendo apenas componentes válidos.
    """
    binaria = (mascara > 0).astype(np.uint8)
    quantidade, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(binaria, 8)
    area_maxima = max(
        config.AREA_MINIMA_COMPONENTE,
        int(mascara.size * config.FRACAO_MAXIMA_COMPONENTE),
    )
    manter = np.zeros(quantidade, dtype=bool)
    if quantidade > 1:
        areas = estatisticas[1:, cv2.CC_STAT_AREA]
        manter[1:] = (areas >= config.AREA_MINIMA_COMPONENTE) & (areas <= area_maxima)
    return manter[rotulos].astype(np.uint8) * 255


def filtrar_regioes_votacao(mascara: np.ndarray) -> np.ndarray:
    """
    Mantém só evidências com continuidade espacial compatível com um traço.
    
    Args:
        mascara (np.ndarray): A máscara binária de entrada com as evidências.
        
    Returns:
        np.ndarray: A máscara contendo apenas as regiões que parecem ser traços.
    """
    binaria = (mascara > 0).astype(np.uint8)
    quantidade, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(binaria, 8)
    comprimento_minimo = max(
        config.COMPRIMENTO_MINIMO_REGIAO_VOTO,
        round(max(mascara.shape[:2]) * config.FRACAO_COMPRIMENTO_MINIMO_VOTO),
    )
    manter = np.zeros(quantidade, dtype=bool)
    for indice in range(1, quantidade):
        _x, _y, largura, altura, area = estatisticas[indice]
        manter[indice] = (
            area >= config.AREA_MINIMA_REGIAO
            and max(largura, altura) >= comprimento_minimo
        )
    return manter[rotulos].astype(np.uint8) * 255


def _limiar_realce(realce: np.ndarray, percentil: float) -> np.ndarray:
    """
    Combina Otsu e percentil alto para não transformar textura em rachadura.
    
    Args:
        realce (np.ndarray): A imagem realçada em tons de cinza.
        percentil (float): O percentil a ser utilizado como limite inferior.
        
    Returns:
        np.ndarray: A máscara binária resultante.
    """
    if not np.any(realce):
        return np.zeros_like(realce)
    limiar_otsu, otsu = cv2.threshold(realce, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    limite = max(float(np.percentile(realce, percentil)), float(limiar_otsu))
    return ((realce >= limite) & (otsu > 0)).astype(np.uint8) * 255


def por_canny(
    cinza_suave: np.ndarray,
    baixo: int = config.LIMIAR_CANNY_BAIXO,
    alto: int = config.LIMIAR_CANNY_ALTO,
) -> np.ndarray:
    """
    Aplica o detector de bordas Canny.
    
    Args:
        cinza_suave (np.ndarray): Imagem em tons de cinza suavizada.
        baixo (int): Limiar inferior para o Canny.
        alto (int): Limiar superior para o Canny.
        
    Returns:
        np.ndarray: A máscara segmentada pelo Canny.
    """
    bordas = cv2.Canny(cinza_suave, baixo, alto)
    limpa = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, _kernel(), iterations=1)
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_OPEN, _kernel(), iterations=1)
    return limpar_componentes(limpa)


def por_otsu(cinza_suave: np.ndarray) -> np.ndarray:
    """
    Aplica a limiarização de Otsu.
    
    Args:
        cinza_suave (np.ndarray): Imagem em tons de cinza suavizada.
        
    Returns:
        np.ndarray: A máscara segmentada por Otsu.
    """
    _, binaria = cv2.threshold(cinza_suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # rachaduras costumam ser escuras: inverte se o fundo ficou branco
    if np.mean(binaria) > 127:
        binaria = cv2.bitwise_not(binaria)
    limpa = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _kernel(), iterations=1)
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, _kernel(), iterations=2)
    return limpar_componentes(limpa)


def por_adaptativo(cinza_suave: np.ndarray) -> np.ndarray:
    """
    Destaca marcas escuras mesmo quando a iluminação muda na imagem.
    
    Args:
        cinza_suave (np.ndarray): Imagem em tons de cinza suavizada.
        
    Returns:
        np.ndarray: A máscara segmentada pelo threshold adaptativo.
    """
    binaria = cv2.adaptiveThreshold(
        cinza_suave,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.ADAPTATIVO_BLOCO,
        config.ADAPTATIVO_C,
    )
    limpa = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _kernel(), iterations=1)
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, _kernel(), iterations=1)
    return limpar_componentes(limpa)


def por_blackhat(cinza_suave: np.ndarray) -> np.ndarray:
    """
    Realça traços escuros finos sobre superfícies de concreto claras.
    
    Args:
        cinza_suave (np.ndarray): Imagem em tons de cinza suavizada.
        
    Returns:
        np.ndarray: A máscara segmentada por transformação Blackhat.
    """
    kernel_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.KERNEL_MORFOLOGIA_GRANDE, config.KERNEL_MORFOLOGIA_GRANDE))
    realce = cv2.morphologyEx(cinza_suave, cv2.MORPH_BLACKHAT, kernel_grande)
    binaria = _limiar_realce(realce, percentil=config.PERCENTIL_BLACKHAT)
    limpa = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _kernel(), iterations=1)
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, _kernel(), iterations=1)
    return limpar_componentes(limpa)


def por_tophat(cinza_suave: np.ndarray) -> np.ndarray:
    """
    Realça traços claros finos, comuns em algumas imagens de concreto.
    
    Args:
        cinza_suave (np.ndarray): Imagem em tons de cinza suavizada.
        
    Returns:
        np.ndarray: A máscara segmentada por transformação Tophat.
    """
    kernel_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.KERNEL_MORFOLOGIA_GRANDE, config.KERNEL_MORFOLOGIA_GRANDE))
    realce = cv2.morphologyEx(cinza_suave, cv2.MORPH_TOPHAT, kernel_grande)
    binaria = _limiar_realce(realce, percentil=config.PERCENTIL_TOPHAT)
    limpa = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _kernel(), iterations=1)
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, _kernel(), iterations=1)
    return limpar_componentes(limpa)


def por_contraste_local(cinza_suave: np.ndarray) -> np.ndarray:
    """
    Combina realces claros e escuros sem assumir a cor da rachadura.
    
    Args:
        cinza_suave (np.ndarray): Imagem em tons de cinza suavizada.
        
    Returns:
        np.ndarray: A máscara segmentada pela combinação de realces.
    """
    return cv2.bitwise_or(por_blackhat(cinza_suave), por_tophat(cinza_suave))


def por_sobel(cinza_suave: np.ndarray) -> np.ndarray:
    """
    Destaca mudanças fortes de intensidade em qualquer direção.
    
    Args:
        cinza_suave (np.ndarray): Imagem em tons de cinza suavizada.
        
    Returns:
        np.ndarray: A máscara segmentada usando o operador de Sobel.
    """
    gradiente_x = cv2.Sobel(cinza_suave, cv2.CV_32F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(cinza_suave, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradiente_x, gradiente_y)
    if float(magnitude.max()) <= 0:
        return np.zeros_like(cinza_suave)
    normalizada = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    binaria = _limiar_realce(normalizada, percentil=config.PERCENTIL_SOBEL)
    limpa = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, _kernel(), iterations=1)
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_OPEN, _kernel(), iterations=1)
    return limpar_componentes(limpa)


def por_frangi(cinza_suave: np.ndarray) -> np.ndarray:
    """
    Realça estruturas escuras e alongadas em várias espessuras usando o filtro de Frangi.
    
    Args:
        cinza_suave (np.ndarray): Imagem em tons de cinza suavizada.
        
    Returns:
        np.ndarray: A máscara segmentada usando o filtro de Frangi.
    """
    # Frangi é o método mais caro. Em meia resolução mantém os traços maiores
    # e reduz o custo aproximadamente quatro vezes em câmeras comuns.
    altura, largura = cinza_suave.shape[:2]
    escala = config.ESCALA_FRANGI if max(altura, largura) > config.FRANGI_LIMIAR_RESOLUCAO else 1.0
    if escala < 1.0:
        tamanho = (round(largura * escala), round(altura * escala))
        entrada = cv2.resize(cinza_suave, tamanho, interpolation=cv2.INTER_AREA)
    else:
        entrada = cinza_suave
    entrada_float = entrada.astype(np.float32) / 255.0
    escura = frangi(
        entrada_float,
        sigmas=(1, 2, 3),
        black_ridges=True,
    )
    clara = frangi(
        entrada_float,
        sigmas=(1, 2, 3),
        black_ridges=False,
    )
    resposta = np.maximum(escura, clara)
    if escala < 1.0:
        resposta = cv2.resize(resposta, (largura, altura), interpolation=cv2.INTER_LINEAR)
    if not np.isfinite(resposta).any() or float(np.nanmax(resposta)) <= 0:
        return np.zeros_like(cinza_suave)
    normalizada = cv2.normalize(resposta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    binaria = _limiar_realce(normalizada, percentil=config.PERCENTIL_FRANGI)
    limpa = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _kernel(), iterations=1)
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, _kernel(), iterations=1)
    return limpar_componentes(limpa)


def combinada(cinza_suave: np.ndarray) -> np.ndarray:
    """
    Une Canny e Otsu — baseline rápida, não um comparativo formal.
    
    Args:
        cinza_suave (np.ndarray): Imagem em tons de cinza suavizada.
        
    Returns:
        np.ndarray: A máscara segmentada por Canny e Otsu combinados.
    """
    return cv2.bitwise_or(por_canny(cinza_suave), por_otsu(cinza_suave))
