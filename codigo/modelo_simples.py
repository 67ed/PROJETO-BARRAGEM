"""Classificador leve treinado com as máscaras do DamCrack, sem rede neural."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from codigo import config

BINS = config.BINS_MODELO_SIMPLES


def caracteristicas_visuais(imagem_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna escurecimento local e força de borda, ambos de 0 a 255.
    
    Args:
        imagem_bgr (np.ndarray): Imagem original no formato BGR.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: O canal de escurecimento e o canal de bordas.
    """
    if imagem_bgr is None or not isinstance(imagem_bgr, np.ndarray):
        raise ValueError("Imagem inválida para características visuais.")
        
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    fundo = cv2.GaussianBlur(cinza, (0, 0), sigmaX=5)
    escuro = cv2.subtract(fundo, cinza)
    gradiente_x = cv2.Sobel(cinza, cv2.CV_32F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(cinza, cv2.CV_32F, 0, 1, ksize=3)
    borda = np.clip(cv2.magnitude(gradiente_x, gradiente_y), 0, 255).astype(np.uint8)
    return escuro, borda


def _indices_caracteristicas(imagem_bgr: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna os índices quantizados com base nas características visuais da imagem.
    
    Args:
        imagem_bgr (np.ndarray): A imagem original no formato BGR.
        bins (int): O número de subdivisões para a discretização.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: Matrizes contendo os índices de escurecimento e de borda discretizados.
    """
    if bins <= 0:
        raise ValueError("bins deve ser um valor positivo.")
        
    escuro, borda = caracteristicas_visuais(imagem_bgr)
    return escuro.astype(np.int16) * bins // 256, borda.astype(np.int16) * bins // 256


@lru_cache(maxsize=2)
def _carregar_tabela(caminho_texto: str) -> tuple[np.ndarray, int]:
    """
    Carrega a tabela de classificação treinada, em cache para uso recorrente.
    
    Args:
        caminho_texto (str): O caminho do arquivo .npz com a tabela.
        
    Returns:
        tuple[np.ndarray, int]: A tabela de pesos e o número de bins utilizado.
    """
    caminho = Path(caminho_texto)
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Modelo simples não encontrado: {caminho}. "
            "Execute `python treinar_modelo_simples.py` primeiro."
        )
    try:
        with np.load(caminho, allow_pickle=False) as dados:
            tabela = dados["tabela"].astype(bool)
            bins = int(dados["bins"])
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Não foi possível ler o modelo simples: {caminho}") from exc
        
    if bins < 2 or tabela.shape != (bins, bins):
        raise RuntimeError("Tabela inválida no modelo simples.")
    return tabela, bins


def segmentar_com_modelo(
    imagem_bgr: np.ndarray,
    caminho_modelo: str | Path | None = None,
) -> np.ndarray:
    """
    Segmenta a imagem por uma tabela leve de escurecimento local e bordas aprendida.
    
    Args:
        imagem_bgr (np.ndarray): Imagem original no formato BGR.
        caminho_modelo (str | Path | None): Caminho do modelo, usa config caso nulo.
        
    Returns:
        np.ndarray: A máscara de segmentação gerada.
    """
    if imagem_bgr is None or not isinstance(imagem_bgr, np.ndarray):
        raise ValueError("A imagem de entrada não é válida.")
        
    caminho = Path(caminho_modelo or config.MODELO_SIMPLES_PADRAO)
    tabela, bins = _carregar_tabela(str(caminho.resolve()))
    indice_escuro, indice_borda = _indices_caracteristicas(imagem_bgr, bins)
    mascara = tabela[indice_escuro, indice_borda].astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)
