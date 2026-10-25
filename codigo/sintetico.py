"""Frames sintéticos de concreto com traço tipo rachadura — demo sem o DamCrack."""

from __future__ import annotations

import cv2
import numpy as np


def gerar_frame(
    seed: int = 0,
    largura: int = 640,
    altura: int = 480,
    com_rachadura: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera uma foto falsa de concreto e, opcionalmente, um traço escuro alongado.

    Returns:
        Imagem BGR e máscara binária da rachadura sintética (pode ser vazia).
    """
    rng = np.random.default_rng(seed)
    if com_rachadura is None:
        com_rachadura = bool(seed % 3)

    base = rng.normal(162, 16, (altura, largura)).clip(95, 215).astype(np.uint8)
    granulo = rng.integers(0, 28, (altura, largura), dtype=np.uint8)
    granulo = cv2.GaussianBlur(granulo, (9, 9), 0)
    concreto = cv2.addWeighted(base, 0.82, granulo, 0.18, 0)

    junta = int(altura * (0.38 + 0.08 * ((seed % 5) - 2) / 2))
    cv2.line(concreto, (0, junta), (largura, junta + rng.integers(-6, 7)), int(rng.integers(120, 145)), 1)

    mascara = np.zeros((altura, largura), dtype=np.uint8)
    if com_rachadura:
        x = int(rng.integers(largura // 6, largura // 3))
        y = int(rng.integers(altura // 10, altura // 4))
        pontos: list[list[int]] = []
        for _ in range(22):
            x = int(np.clip(x + int(rng.integers(-6, 16)), 8, largura - 8))
            y = int(np.clip(y + int(rng.integers(6, 22)), 8, altura - 8))
            pontos.append([x, y])
        poli = np.array(pontos, dtype=np.int32)
        tom = int(rng.integers(22, 48))
        cv2.polylines(concreto, [poli], False, tom, thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(mascara, [poli], False, 255, thickness=3, lineType=cv2.LINE_AA)

    bgr = cv2.cvtColor(concreto, cv2.COLOR_GRAY2BGR)
    matiz = np.zeros_like(bgr)
    matiz[:, :] = (18, 22, 12)
    bgr = cv2.addWeighted(bgr, 0.92, matiz, 0.08, 0)
    return bgr, mascara


def gerar_lote(n: int = 12, semente: int = 7) -> list[np.ndarray]:
    """Sequência curta para o modo demo, misturando frames com e sem traço."""
    return [gerar_frame(seed=semente + i)[0] for i in range(max(1, n))]
