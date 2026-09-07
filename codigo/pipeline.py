"""Orquestra um exemplo: dados → detector → alerta → texto do bot."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
import numpy as np

from codigo import alerts, bot
from codigo.analysis import classificar_tamanho_visual
from codigo.detector import detectar, ResultadoDetector

logger = logging.getLogger(__name__)


def montar_resultado(
    camera_id: str,
    deteccao: ResultadoDetector,
    frames_persistentes: int = 1,
) -> dict:
    """
    Monta o dicionário de resultados e contexto baseado na detecção.
    
    Args:
        camera_id (str): Identificador da câmera.
        deteccao (ResultadoDetector): Instância de ResultadoDetector gerada pela detecção.
        frames_persistentes (int): O número de frames persistentes consecutivos.
        
    Returns:
        dict: O contexto de saída consolidado.
    """
    area = sum(r.area for r in deteccao.regioes)
    confianca = alerts.confianca_simples(area, deteccao.iou_referencia)
    principal = deteccao.regioes[0] if deteccao.regioes else None
    tamanho = classificar_tamanho_visual(principal, deteccao.mascara.shape)
    nivel = alerts.classificar(
        tem_regiao=bool(deteccao.regioes),
        confianca=confianca,
        frames_persistentes=frames_persistentes,
        confirmado_por_metodos=deteccao.extra.get("confirmado_por_metodos", False),
        tamanho_visual=tamanho["codigo"],
    )
    return {
        "camera_id": camera_id,
        "timestamp": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z"),
        "alert_level": nivel,
        "alert_label": bot.rotulo_nivel(nivel),
        "confidence": confianca,
        "region": None
        if principal is None
        else {
            "x": principal.x,
            "y": principal.y,
            "width": principal.largura,
            "height": principal.altura,
        },
        "visual_size": tamanho,
        "persistent_frames": frames_persistentes,
        "area": area,
        "iou": deteccao.iou_referencia,
        "n_regioes": len(deteccao.regioes),
        "tempo_s": round(deteccao.tempo_s, 4),
        "metodo": deteccao.extra.get("metodo", "-"),
        "metodos_votacao": deteccao.extra.get("metodos_votacao", []),
        "coberturas_por_metodo": deteccao.extra.get("coberturas_por_metodo", {}),
        "metodos_descartados": deteccao.extra.get("metodos_descartados", []),
        "metodos_substituidos": deteccao.extra.get("metodos_substituidos", {}),
        "votos_minimos": deteccao.extra.get("votos_minimos", 0),
        "maior_votacao": deteccao.extra.get("maior_votacao", 0),
        "confirmado_por_metodos": deteccao.extra.get("confirmado_por_metodos", False),
        "explicacao": "",
    }


def analisar_imagem(
    imagem_bgr: np.ndarray,
    mascara_ref: np.ndarray | None = None,
    camera_id: str = "CAM01",
    metodo: str = "votacao",
    caminho_modelo: str | None = None,
    frames_persistentes: int = 1,
) -> tuple[ResultadoDetector, dict]:
    """
    Analisa a imagem e retorna o resultado da detecção em conjunto com a estrutura de dados.
    
    Args:
        imagem_bgr (np.ndarray): Imagem original no formato BGR.
        mascara_ref (np.ndarray | None): Máscara de referência para medir IoU.
        camera_id (str): Identificador da câmera.
        metodo (str): Método de detecção utilizado.
        caminho_modelo (str | None): Caminho do modelo simples para segmentação, se usado.
        frames_persistentes (int): O número de frames persistentes informados.
        
    Returns:
        tuple[ResultadoDetector, dict]: Instância do resultado da detecção e o dicionário de saída.
    """
    det = detectar(
        imagem_bgr,
        mascara_referencia=mascara_ref,
        metodo=metodo,
        caminho_modelo=caminho_modelo,
    )
    dados = montar_resultado(camera_id, det, frames_persistentes=frames_persistentes)
    dados["explicacao"] = bot.explicar(dados)
    return det, dados
