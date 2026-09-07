"""Níveis internos de alerta do software — não são risco estrutural."""

from __future__ import annotations

import logging
from enum import Enum

from codigo import config

logger = logging.getLogger(__name__)


class NivelAlerta(Enum):
    """Representa os possíveis níveis de atenção para alterações detectadas."""
    NENHUM = "nenhum"
    BAIXA_ATENCAO = "baixa_atencao"
    INSPECAO = "inspecao"
    PRIORITARIO = "prioritario"


NIVEIS: dict[str, str] = {
    NivelAlerta.BAIXA_ATENCAO.value: "baixa atenção",
    NivelAlerta.INSPECAO.value: "necessita inspeção",
    NivelAlerta.PRIORITARIO.value: "aviso urgente — inspeção prioritária",
    NivelAlerta.NENHUM.value: "nenhuma alteração relevante",
}


def classificar(
    tem_regiao: bool,
    confianca: float,
    frames_persistentes: int = 1,
    confirmado_por_metodos: bool = False,
    tamanho_visual: str = "nenhuma",
) -> str:
    """Classifica o nível de alerta baseado em heurísticas visuais.
    
    Atenção: Não é diagnóstico de risco estrutural.
    
    Args:
        tem_regiao: Verdadeiro se houver região candidata.
        confianca: Nível de confiança da detecção (0 a 1).
        frames_persistentes: Número de quadros seguidos com a detecção.
        confirmado_por_metodos: Se a detecção teve concordância de múltiplos métodos.
        tamanho_visual: Classificação do tamanho visual.
        
    Returns:
        O código string do nível de alerta.
    """
    if not tem_regiao:
        return NivelAlerta.NENHUM.value
        
    # O nível prioritário exige confirmação independente e região visualmente
    # extensa. Ainda assim, não é diagnóstico de risco estrutural.
    if confirmado_por_metodos and tamanho_visual == "grande":
        return NivelAlerta.PRIORITARIO.value
        
    if confirmado_por_metodos or tamanho_visual in {"media", "grande"}:
        return NivelAlerta.INSPECAO.value
        
    if (
        frames_persistentes >= config.PERSISTENCIA_INSPECAO
        or confianca >= config.CONFIANCA_INSPECAO
    ):
        return NivelAlerta.INSPECAO.value
        
    return NivelAlerta.BAIXA_ATENCAO.value


def confianca_simples(area_total: float, iou: float | None) -> float:
    """Heurística da Semana 2: área + IoU com a máscara (se houver).
    
    Args:
        area_total: Área total identificada.
        iou: Intersection over Union se a referência estiver disponível, ou None.
        
    Returns:
        Um valor float entre 0.0 e 1.0 representando a confiança.
    """
    base = min(1.0, area_total / 8000.0)
    if iou is None:
        return round(0.35 + 0.4 * base, 3)
    return round(min(1.0, 0.25 + 0.75 * iou), 3)
