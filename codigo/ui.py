"""HUD do monitoramento: status, votação, bot e controles."""

from __future__ import annotations

import cv2
import numpy as np

from codigo.texto import para_hud

CORES = {
    "nenhum": (168, 168, 168),
    "baixa_atencao": (0, 196, 255),
    "inspecao": (0, 140, 255),
    "prioritario": (36, 36, 255),
}

CHIP = {
    "nenhum": "OK",
    "baixa_atencao": "ATENCAO",
    "inspecao": "INSPECAO",
    "prioritario": "URGENTE",
}

RESUMO = {
    "nenhum": "Sem evidencia suficiente neste quadro. Isso nao garante que a estrutura esteja segura.",
    "baixa_atencao": "Possivel alteracao visual com pouca evidencia. Pode ser sombra, sujeira ou textura.",
    "inspecao": "Alteracao persistente. Encaminhar para inspecao da equipe responsavel.",
    "prioritario": "Tres ou mais metodos concordaram numa alteracao extensa. Priorizar avaliacao humana.",
}

ALTURA_CONTROLES = 56
FONTE = cv2.FONT_HERSHEY_SIMPLEX
AJUDA = (
    "q/Esc sair   espaco pausar   s salvar   m metodo   h ajuda   n proximo"
)


def _retangulo_alpha(
    vis: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    cor: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = vis.copy()
    cv2.rectangle(overlay, p1, p2, cor, -1)
    cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)


def _texto(
    vis: np.ndarray,
    texto: str,
    origem: tuple[int, int],
    escala: float,
    cor: tuple[int, int, int],
    espessura: int = 1,
) -> None:
    cv2.putText(vis, para_hud(texto), origem, FONTE, escala, cor, espessura, cv2.LINE_AA)


def _pips_voto(vis: np.ndarray, x: int, y: int, preenchidos: int, total: int, cor: tuple[int, int, int]) -> None:
    total = max(0, total)
    for i in range(total):
        cx = x + i * 18
        if i < preenchidos:
            cv2.circle(vis, (cx, y), 6, cor, -1, cv2.LINE_AA)
        else:
            cv2.circle(vis, (cx, y), 6, (90, 90, 90), 1, cv2.LINE_AA)


def desenhar_painel(
    overlay_bgr: np.ndarray,
    dados: dict,
    *,
    fps: float = 0.0,
    pausado: bool = False,
    ajuda: bool = False,
    rodape_px: int = ALTURA_CONTROLES,
) -> np.ndarray:
    """Desenha o HUD sobre o overlay da detecção."""
    vis = overlay_bgr.copy()
    h, w = vis.shape[:2]
    nivel = str(dados.get("alert_level", "nenhum"))
    cor = CORES.get(nivel, (255, 255, 255))

    altura_topo = 74
    _retangulo_alpha(vis, (0, 0), (w, altura_topo), (12, 12, 12), 0.72)
    cv2.rectangle(vis, (0, 0), (7, altura_topo), cor, -1)

    rotulo = CHIP.get(nivel, "STATUS")
    (tw, _th), _ = cv2.getTextSize(rotulo, FONTE, 0.55, 1)
    chip_w = tw + 24
    chip_x = w - chip_w - 12
    _retangulo_alpha(vis, (chip_x, 10), (chip_x + chip_w, 40), cor, 0.9)
    _texto(vis, rotulo, (chip_x + 12, 32), 0.55, (255, 255, 255), 1)

    status = "PAUSADO" if pausado else "AO VIVO"
    _texto(vis, "BARRAGEM  |  DETECTOR", (18, 28), 0.58, cor, 1)
    fps_txt = f"{fps:.1f} fps" if fps > 0 else f"{dados.get('tempo_s', 0):.2f}s"
    _texto(
        vis,
        (
            f"{status}  {dados.get('camera_id', '-')}  {fps_txt}  "
            f"{dados.get('metodo', '-')}  R{dados.get('n_regioes', 0)}"
        ),
        (18, 52),
        0.45,
        (220, 220, 220),
    )
    metodos = dados.get("metodos_votacao") or []
    if metodos:
        _pips_voto(
            vis,
            w - 18 * len(metodos) - 10,
            64,
            int(dados.get("maior_votacao", 0) or 0),
            len(metodos),
            cor,
        )

    texto = RESUMO.get(nivel, str(dados.get("explicacao") or ""))
    if dados.get("maior_votacao") and metodos:
        texto += f"  Votos {dados.get('maior_votacao')}/{len(metodos)}."
    linhas = _quebrar(para_hud(texto), max(48, w // 11))
    caixa_h = 28 + 16 * max(0, len(linhas) - 1)
    y0 = h - rodape_px - caixa_h
    _retangulo_alpha(vis, (0, y0), (w, h - rodape_px), (10, 10, 10), 0.78)
    _texto(vis, "BOT", (16, y0 + 18), 0.45, cor, 1)
    for i, linha in enumerate(linhas):
        x = 62 if i == 0 else 16
        y = y0 + 18 + i * 16
        _texto(vis, linha, (x, y), 0.42, (230, 230, 230))

    if ajuda:
        _retangulo_alpha(vis, (16, altura_topo + 12), (w - 16, altura_topo + 48), (0, 0, 0), 0.55)
        _texto(vis, AJUDA, (28, altura_topo + 36), 0.45, (240, 240, 240))

    return vis


def _quebrar(texto: str, max_chars: int = 62) -> list[str]:
    palavras = texto.split()
    linhas: list[str] = []
    atual = ""
    for palavra in palavras:
        teste = f"{atual} {palavra}".strip()
        if len(teste) <= max_chars:
            atual = teste
        else:
            if atual:
                linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return linhas[:6]
