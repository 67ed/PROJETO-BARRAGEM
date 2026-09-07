"""Tela de teste: overlay + alerta + texto do bot."""

from __future__ import annotations

import cv2
import numpy as np

CORES = {
    "nenhum": (160, 160, 160),
    "baixa_atencao": (0, 200, 255),
    "inspecao": (0, 140, 255),
    "prioritario": (0, 0, 220),
}

# Constantes de Interface (UI)
LARGURA_TEXTO_PADRAO = 62
ALTURA_FAIXA_TOPO = 158
MARGEM_ESQUERDA = 16
MARGEM_SUPERIOR = 22
LARGURA_BARRA_LATERAL = 8
FONTE_PADRAO = cv2.FONT_HERSHEY_SIMPLEX
ESCALA_FONTE_TITULO = 0.55
ESCALA_FONTE_TEXTO = 0.48
ESCALA_FONTE_BOT = 0.42


def _quebrar(texto: str, max_chars: int = LARGURA_TEXTO_PADRAO) -> list[str]:
    """
    Divide um texto em uma lista de linhas para não ultrapassar a largura informada.
    
    Args:
        texto (str): O texto a ser quebrado.
        max_chars (int): Largura máxima em caracteres para cada linha.
        
    Returns:
        list[str]: As linhas textuais ajustadas ao tamanho especificado (até 8 linhas).
    """
    palavras = texto.split()
    linhas: list[str] = []
    atual = ""
    for p in palavras:
        teste = f"{atual} {p}".strip()
        if len(teste) <= max_chars:
            atual = teste
        else:
            if atual:
                linhas.append(atual)
            atual = p
    if atual:
        linhas.append(atual)
    return linhas[:8]


def desenhar_painel(overlay_bgr: np.ndarray, dados: dict) -> np.ndarray:
    """
    Desenha os elementos da interface de usuário sobre a imagem da detecção.
    
    Args:
        overlay_bgr (np.ndarray): Imagem base com marcações visuais dos resultados.
        dados (dict): O contexto de saída com informações relevantes a serem mostradas no painel.
        
    Returns:
        np.ndarray: O frame final pronto para renderização.
    """
    vis = overlay_bgr.copy()
    h, w = vis.shape[:2]
    nivel = dados.get("alert_level", "nenhum")
    cor = CORES.get(nivel, (255, 255, 255))

    cv2.rectangle(vis, (0, 0), (w, ALTURA_FAIXA_TOPO), (20, 20, 20), -1)
    cv2.rectangle(vis, (0, 0), (LARGURA_BARRA_LATERAL, ALTURA_FAIXA_TOPO), cor, -1)

    linhas_topo = [
        "MONITORAMENTO BARRAGEM  (prototipo — nao e diagnostico)",
        f"Camera: {dados.get('camera_id', '-')}   {dados.get('timestamp', '')}",
        (
            f"Nivel: {dados.get('alert_label', '-')}   "
            f"tamanho: {dados.get('visual_size', {}).get('rotulo', '-')}"
        ),
        f"Metodo: {dados.get('metodo', '-')}   regioes: {dados.get('n_regioes', 0)}   confianca: {dados.get('confidence', 0)}",
        (
            f"Votacao: {dados.get('maior_votacao', 0)}/{len(dados.get('metodos_votacao', []))} "
            f"metodos   minimo urgente: {dados.get('votos_minimos', 0)}"
            if dados.get("metodos_votacao")
            else "Votacao: nao aplicada neste metodo"
        ),
        (
            "Métodos desconsiderados por excesso de ruído: "
            f"{', '.join(dados.get('metodos_descartados', []))}"
            if dados.get("metodos_descartados")
            else "Filtro de ruído e iluminação aplicado a todos os métodos"
        ),
    ]
    y = MARGEM_SUPERIOR
    for i, linha in enumerate(linhas_topo):
        cv2.putText(
            vis,
            linha,
            (MARGEM_ESQUERDA, y),
            FONTE_PADRAO,
            ESCALA_FONTE_TEXTO if i else ESCALA_FONTE_TITULO,
            (240, 240, 240) if i else cor,
            1,
            cv2.LINE_AA,
        )
        y += 22

    texto = dados.get("explicacao") or ""
    linhas_bot = _quebrar(texto, max_chars=max(40, w // 11))
    caixa_h = 22 + 18 * len(linhas_bot)
    y0 = h - caixa_h
    cv2.rectangle(vis, (0, y0), (w, h), (15, 15, 15), -1)
    cv2.putText(vis, "BOT:", (MARGEM_ESQUERDA, y0 + 18), FONTE_PADRAO, 0.5, cor, 1, cv2.LINE_AA)
    yy = y0 + 38
    for linha in linhas_bot:
        cv2.putText(vis, linha, (MARGEM_ESQUERDA, yy), FONTE_PADRAO, ESCALA_FONTE_BOT, (230, 230, 230), 1, cv2.LINE_AA)
        yy += 18
    return vis
