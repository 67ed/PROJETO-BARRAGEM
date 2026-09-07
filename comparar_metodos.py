#!/usr/bin/env python3
"""Gera uma grade visual com todos os detectores para uma mesma imagem.

Exemplos:
    python comparar_metodos.py
    python comparar_metodos.py --imagem caminho/para/foto.jpg --mostrar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from codigo import config
from codigo.data import carregar_bgr, extrair_subconjunto, listar_pares_extraidos
from codigo.detector import METODOS_VOTACAO, detectar


def _imagem_para_comparar(args: argparse.Namespace) -> Path:
    if args.imagem is not None:
        if not args.imagem.is_file():
            raise SystemExit(f"Imagem não encontrada: {args.imagem}")
        return args.imagem

    pares = listar_pares_extraidos()
    if pares and pares[0].caminho_imagem.is_file():
        return pares[0].caminho_imagem

    if not args.zip.is_file():
        raise SystemExit(
            "Informe --imagem ou coloque o ZIP DamCrack no caminho configurado: "
            f"{args.zip}"
        )
    return extrair_subconjunto(n=1, caminho_zip=args.zip)[0].caminho_imagem


def _cartao(imagem: np.ndarray, titulo: str, largura: int = 320, altura: int = 250) -> np.ndarray:
    """Padroniza imagens de tamanhos diferentes em um cartão com título."""
    painel = np.full((altura, largura, 3), 18, dtype=np.uint8)
    area_altura = altura - 34
    escala = min(largura / imagem.shape[1], area_altura / imagem.shape[0])
    novo_tamanho = (max(1, round(imagem.shape[1] * escala)), max(1, round(imagem.shape[0] * escala)))
    redimensionada = cv2.resize(imagem, novo_tamanho, interpolation=cv2.INTER_AREA)
    y = 34 + (area_altura - redimensionada.shape[0]) // 2
    x = (largura - redimensionada.shape[1]) // 2
    painel[y : y + redimensionada.shape[0], x : x + redimensionada.shape[1]] = redimensionada
    cv2.rectangle(painel, (0, 0), (largura, 33), (35, 35, 35), -1)
    cv2.putText(painel, titulo, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
    return painel


def _grade(cartoes: list[np.ndarray], colunas: int = 4) -> np.ndarray:
    while len(cartoes) % colunas:
        cartoes.append(np.full_like(cartoes[0], 18))
    linhas = [
        cv2.hconcat(cartoes[indice : indice + colunas])
        for indice in range(0, len(cartoes), colunas)
    ]
    return cv2.vconcat(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparação visual dos métodos de detecção")
    parser.add_argument("--imagem", type=Path, help="foto a analisar; sem ela usa a primeira amostra")
    parser.add_argument("--zip", type=Path, default=config.ZIP_PADRAO)
    parser.add_argument("--modelo", type=Path, default=config.MODELO_SIMPLES_PADRAO)
    parser.add_argument("--saida", type=Path, help="arquivo PNG da grade gerada")
    parser.add_argument("--mostrar", action="store_true", help="abre a grade em uma janela OpenCV")
    args = parser.parse_args()

    caminho_imagem = _imagem_para_comparar(args)
    imagem = carregar_bgr(caminho_imagem)
    cartoes = [_cartao(imagem, "Imagem original")]

    # Estes são os cinco métodos que participam do aviso urgente 3 de 5.
    for metodo in METODOS_VOTACAO:
        try:
            resultado = detectar(imagem, metodo=metodo, caminho_modelo=str(args.modelo))
        except FileNotFoundError:
            print(f"Pulando {metodo}: modelo não encontrado em {args.modelo}")
            continue
        cartoes.append(_cartao(resultado.overlay, metodo.capitalize()))

    # Os métodos abaixo continuam disponíveis para comparação, mas não entram
    # na votação urgente: Otsu e o contraste local são mais sensíveis à textura.
    for metodo in ("otsu", "adaptativo", "blackhat", "contraste_local", "combinada", "consenso"):
        try:
            resultado = detectar(imagem, metodo=metodo, caminho_modelo=str(args.modelo))
        except FileNotFoundError:
            print(f"Pulando {metodo}: modelo não encontrado em {args.modelo}")
            continue
        cartoes.append(_cartao(resultado.overlay, metodo.capitalize()))

    votacao = detectar(imagem, metodo="votacao", caminho_modelo=str(args.modelo))
    votos = votacao.extra["maior_votacao"]
    minimo = votacao.extra["votos_minimos"]
    cartoes.append(_cartao(votacao.overlay, f"Votação {minimo}/5 (máx. {votos})"))
    grade = _grade(cartoes)

    saida = args.saida or (
        config.PASTA_RESULTADOS / "comparacoes" / f"{caminho_imagem.stem}_metodos.png"
    )
    saida.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(saida), grade):
        raise RuntimeError(f"Não foi possível salvar a comparação em {saida}")

    print(f"Imagem: {caminho_imagem}")
    print(f"Comparação salva em: {saida}")
    print(
        f"Votação urgente: {votos}/5 votos no mesmo ponto; mínimo configurado: {minimo}. "
        f"Regiões confirmadas: {len(votacao.regioes)}."
    )
    if args.mostrar:
        cv2.namedWindow("Comparação dos métodos", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Comparação dos métodos", 1280, 800)
        cv2.imshow("Comparação dos métodos", grade)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
