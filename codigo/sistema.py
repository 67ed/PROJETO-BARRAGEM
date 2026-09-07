"""Sistema completo: câmera → detecção → persistência → alerta → bot."""

from __future__ import annotations

import logging
import numpy as np

from codigo import bot
from codigo.detector import detectar, ResultadoDetector
from codigo.pipeline import montar_resultado
from codigo.temporal import RastreadorPersistencia
from codigo.ui import desenhar_painel

logger = logging.getLogger(__name__)


class SistemaMonitoramento:
    """
    Controla o pipeline de rastreamento completo incluindo temporalidade e alertas.
    
    Attributes:
        camera_id (str): Identificador único da câmera em uso.
        metodo (str): O método que a detecção utilizará.
        caminho_modelo (str | None): O caminho do modelo de classificação utilizado.
        rastreador (RastreadorPersistencia): Utilitário de persistência temporal dos traços.
    """
    def __init__(
        self,
        camera_id: str = "CAM01",
        metodo: str = "votacao",
        caminho_modelo: str | None = None,
    ):
        """
        Inicializa os componentes e prepara o ambiente para o processamento.
        
        Args:
            camera_id (str): O ID da câmera. Padrão "CAM01".
            metodo (str): O método de análise. Padrão "votacao".
            caminho_modelo (str | None): Caminho do modelo para classificação simples.
        """
        self.camera_id = camera_id
        self.metodo = metodo
        self.caminho_modelo = caminho_modelo
        self.rastreador = RastreadorPersistencia()
        logger.info(f"Sistema inicializado com câmera: {camera_id}, método: {metodo}")

    def processar(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, ResultadoDetector, dict]:
        """
        Processa um frame completo: detecta os danos, atualiza as regiões rastreadas,
        gera alertas e constrói um painel interativo.
        
        Args:
            frame_bgr (np.ndarray): O quadro obtido da câmera ou do vídeo em BGR.
            
        Returns:
            tuple[np.ndarray, ResultadoDetector, dict]: Uma tupla contendo o painel renderizado, a detecção completa e o dicionário de resultados.
        """
        det = detectar(
            frame_bgr,
            metodo=self.metodo,
            caminho_modelo=self.caminho_modelo,
        )
        persistencia = self.rastreador.atualizar(det.regioes)
        dados = montar_resultado(
            self.camera_id,
            det,
            frames_persistentes=persistencia,
        )
        dados["explicacao"] = bot.explicar(dados)
        tela = desenhar_painel(det.overlay, dados)
        return tela, det, dados
