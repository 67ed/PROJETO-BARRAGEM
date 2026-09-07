#!/usr/bin/env python3
"""Treina uma tabela visual leve usando o ZIP DamCrack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codigo import config
from codigo.treino_simples import treinar_modelo_simples


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino leve para o DamCrack")
    parser.add_argument("--zip", type=Path, default=config.ZIP_PADRAO)
    parser.add_argument("--saida", type=Path, default=config.MODELO_SIMPLES_PADRAO)
    parser.add_argument(
        "--amostras",
        type=int,
        default=300,
        help="quantidade de imagens; 0 usa as 4000, 300 é mais leve",
    )
    parser.add_argument("--validacao", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Treino leve: processa uma imagem por vez e não usa rede neural.")
    resultado = treinar_modelo_simples(
        caminho_zip=args.zip,
        saida_modelo=args.saida,
        amostras=args.amostras,
        validacao=args.validacao,
        seed=args.seed,
    )
    print(f"\nModelo salvo em: {args.saida}")
    print(f"Corte aprendido: {resultado['corte']}")
    print(
        f"Validação — IoU: {resultado['iou_validacao']:.3f}, "
        f"precisão: {resultado['precisao_validacao']:.3f}, "
        f"recall: {resultado['recall_validacao']:.3f}"
    )
    print("Use no app: python app.py --metodo treinado")


if __name__ == "__main__":
    main()
