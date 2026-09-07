"""Captura de frames: webcam, vídeo ou pasta de imagens (teste no VS Code)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FonteFrames:
    """
    Gerencia a obtenção de frames a partir de captura de vídeo ou de imagens.
    
    Attributes:
        captura (cv2.VideoCapture | None): O objeto de captura de vídeo se aplicável.
        imagens (list[Path]): A lista de caminhos de imagens.
        indice (int): O índice atual para a leitura da imagem sequencial.
        nome (str): O nome representativo da fonte.
    """
    def __init__(self, captura: cv2.VideoCapture | None, imagens: list[Path] | None, nome: str):
        self.captura = captura
        self.imagens = imagens or []
        self.indice = 0
        self.nome = nome

    @property
    def ativa(self) -> bool:
        """
        Verifica se a fonte de frames está ativa.
        
        Returns:
            bool: True se a fonte possui imagens ou a captura está aberta, False caso contrário.
        """
        if self.captura is not None:
            return self.captura.isOpened()
        return len(self.imagens) > 0

    def ler(self) -> np.ndarray | None:
        """
        Lê um frame a partir da fonte.
        
        Returns:
            np.ndarray | None: A matriz do frame obtido ou None se não houver mais frames.
        """
        if self.captura is not None:
            ok, frame = self.captura.read()
            if not ok:
                return None
            return frame
        if not self.imagens:
            return None
        caminho = self.imagens[self.indice % len(self.imagens)]
        self.indice += 1
        frame = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
        return frame

    def soltar(self) -> None:
        """Libera os recursos associados à fonte."""
        if self.captura is not None:
            self.captura.release()
            
    def __enter__(self) -> FonteFrames:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.soltar()


def abrir_webcam(indice: int = 0, largura: int = 640, altura: int = 480) -> FonteFrames:
    """
    Abre uma captura a partir da webcam do sistema.
    
    Args:
        indice (int): O índice da câmera conectada.
        largura (int): A largura desejada da imagem.
        altura (int): A altura desejada da imagem.
        
    Returns:
        FonteFrames: A fonte criada e pronta para ser utilizada.
    """
    cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        cap.release()
        logger.error(f"Erro ao abrir a câmera {indice}.")
        raise RuntimeError(
            f"Não abriu a câmera {indice}. No VS Code tente --demo ou --video arquivo.mp4"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    return FonteFrames(cap, None, f"CAM{indice:02d}")


def abrir_video(caminho: Path) -> FonteFrames:
    """
    Abre uma captura a partir de um arquivo de vídeo.
    
    Args:
        caminho (Path): Caminho absoluto ou relativo para o vídeo.
        
    Returns:
        FonteFrames: A fonte de frames lidos a partir do vídeo.
    """
    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        cap.release()
        logger.error(f"Erro ao abrir o vídeo: {caminho}")
        raise RuntimeError(f"Não abriu o vídeo: {caminho}")
    return FonteFrames(cap, None, caminho.stem)


def abrir_pasta_imagens(pasta: Path) -> FonteFrames:
    """
    Prepara uma lista de imagens para leitura sequencial.
    
    Args:
        pasta (Path): Pasta com arquivos de imagens (jpg, jpeg, png).
        
    Returns:
        FonteFrames: A fonte configurada para retornar sequencialmente as imagens do diretório.
    """
    # Usamos list comprehension com iteradores de padrões p/ case-insensitive match (manualmente)
    extensoes = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    arquivos = []
    for ext in extensoes:
        arquivos.extend(pasta.glob(ext))
        
    arquivos = sorted(set(arquivos))
    
    if not arquivos:
        logger.error(f"Nenhuma imagem em {pasta}")
        raise RuntimeError(f"Nenhuma imagem em {pasta}")
    return FonteFrames(None, arquivos, pasta.name)
