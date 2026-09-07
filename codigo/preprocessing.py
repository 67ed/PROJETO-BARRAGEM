"""Pré-processamento leve, imagem por imagem."""

from __future__ import annotations

import logging
import cv2
import numpy as np

from codigo import config

logger = logging.getLogger(__name__)


def redimensionar_se_preciso(
    imagem: np.ndarray | None,
    tamanho_maximo: int = config.TAMANHO_MAXIMO,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Redimensiona a imagem preservando a proporção, se o maior lado exceder tamanho_maximo.
    
    Args:
        imagem: Imagem de entrada.
        tamanho_maximo: Tamanho máximo permitido para o maior lado.
        interpolation: Método de interpolação do OpenCV.
        
    Returns:
        Imagem redimensionada ou a original se for menor ou igual ao máximo.
        
    Raises:
        ValueError: Se a imagem for None ou tiver dimensões nulas.
    """
    if imagem is None:
        logger.error("A imagem fornecida é None.")
        raise ValueError("A imagem não pode ser None.")
        
    altura, largura = imagem.shape[:2]
    if altura == 0 or largura == 0:
        logger.error("A imagem possui dimensões nulas.")
        raise ValueError("Dimensões da imagem não podem ser zero.")
        
    maior = max(altura, largura)
    if maior <= tamanho_maximo:
        return imagem
    escala = tamanho_maximo / float(maior)
    novo = (int(largura * escala), int(altura * escala))
    
    if novo[0] == 0 or novo[1] == 0:
        logger.warning("Redimensionamento levaria a dimensão nula. Retornando original.")
        return imagem
        
    return cv2.resize(imagem, novo, interpolation=interpolation)


def para_cinza(imagem_bgr: np.ndarray | None) -> np.ndarray:
    """Converte a imagem de BGR para tons de cinza.
    
    Args:
        imagem_bgr: Imagem colorida BGR ou já em tons de cinza.
        
    Returns:
        Imagem em tons de cinza.
        
    Raises:
        ValueError: Se a imagem for None.
    """
    if imagem_bgr is None:
        raise ValueError("A imagem não pode ser None.")
        
    if imagem_bgr.ndim == 2:
        return imagem_bgr
    return cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)


def suavizar(
    cinza: np.ndarray | None,
    kernel: tuple[int, int] = config.KERNEL_BLUR,
) -> np.ndarray:
    """Aplica suavização Gaussiana para reduzir pequenos ruídos.
    
    Args:
        cinza: Imagem em tons de cinza.
        kernel: Tamanho do kernel gaussiano.
        
    Returns:
        Imagem suavizada.
        
    Raises:
        ValueError: Se a imagem for None.
    """
    if cinza is None:
        raise ValueError("A imagem não pode ser None.")
    return cv2.GaussianBlur(cinza, kernel, 0)


def reduzir_ruido(cinza: np.ndarray | None) -> np.ndarray:
    """Remove granulação preservando as bordas mais relevantes usando filtro bilateral.
    
    Args:
        cinza: Imagem em tons de cinza.
        
    Returns:
        Imagem com ruído reduzido e bordas preservadas.
        
    Raises:
        ValueError: Se a imagem for None.
    """
    if cinza is None:
        raise ValueError("A imagem não pode ser None.")
    return cv2.bilateralFilter(
        cinza,
        d=config.BILATERAL_DIAMETRO,
        sigmaColor=config.BILATERAL_SIGMA_COR,
        sigmaSpace=config.BILATERAL_SIGMA_ESPACO,
    )


def normalizar_iluminacao(cinza: np.ndarray | None) -> np.ndarray:
    """Compensa sombras e diferenças locais de iluminação com baixo custo usando CLAHE.
    
    Args:
        cinza: Imagem em tons de cinza.
        
    Returns:
        Imagem equalizada adaptativamente.
        
    Raises:
        ValueError: Se a imagem for None.
    """
    if cinza is None:
        raise ValueError("A imagem não pode ser None.")
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_GRADE,
    )
    return clahe.apply(cinza)


def preparar(imagem_bgr: np.ndarray | None) -> dict[str, np.ndarray]:
    """Prepara a imagem original gerando versões usadas pelos detectores.
    
    Args:
        imagem_bgr: Imagem original colorida.
        
    Returns:
        Dicionário contendo a imagem original limitada e suas versões (cinza,
        sem ruído, normalizada, suave).
        
    Raises:
        ValueError: Se a imagem for None.
    """
    if imagem_bgr is None:
        logger.error("Falha ao preparar imagem: input None.")
        raise ValueError("A imagem não pode ser None.")
        
    limitada = redimensionar_se_preciso(imagem_bgr)
    cinza = para_cinza(limitada)
    sem_ruido = reduzir_ruido(cinza)
    normalizada = normalizar_iluminacao(sem_ruido)
    suave = suavizar(normalizada)
    return {
        "bgr": limitada,
        "cinza": cinza,
        "sem_ruido": sem_ruido,
        "normalizada": normalizada,
        "suave": suave,
    }
