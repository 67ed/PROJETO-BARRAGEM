#!/usr/bin/env python3
"""
Semana 2 — extrai um subconjunto pequeno do DamCrack, roda o detector
clássico e grava overlays + um log em JSON.

Uso (na pasta do projeto, com o venv ativo):

    python run_semana2.py
    python run_semana2.py --n 8 --metodo canny
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Garante que `import codigo` funcione mesmo rodando o script direto
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from codigo import config
from codigo.data import carregar_bgr, carregar_mascara_binaria, extrair_subconjunto
from codigo.pipeline import analisar_imagem


def main() -> None:
    parser = argparse.ArgumentParser(description="Primeiros testes — Semana 2")
    parser.add_argument("--n", type=int, default=8, help="Quantas imagens extrair do zip")
    parser.add_argument("--zip", type=Path, default=config.ZIP_PADRAO)
    parser.add_argument(
        "--metodo",
        choices=["canny", "otsu", "adaptativo", "blackhat", "tophat", "contraste_local", "sobel", "frangi", "combinada", "treinado", "votacao", "consenso"],
        default="votacao",
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=config.MODELO_SIMPLES_PADRAO,
        help="modelo .npz ao usar --metodo treinado",
    )
    parser.add_argument("--camera", default="CAM01")
    args = parser.parse_args()

    if not args.zip.is_file():
        raise SystemExit(
            f"Não achei o zip em {args.zip}. "
            "Coloque Damage Segmentaion.zip nesse caminho ou passe --zip."
        )

    print(f"Extraindo {args.n} pares imagem+máscara (não o zip inteiro)...")
    pares = extrair_subconjunto(n=args.n, caminho_zip=args.zip)
    pasta_img = config.PASTA_RESULTADOS / "imagens"
    pasta_msk = config.PASTA_RESULTADOS / "mascaras"
    pasta_log = config.PASTA_RESULTADOS / "registros"
    pasta_img.mkdir(parents=True, exist_ok=True)
    pasta_msk.mkdir(parents=True, exist_ok=True)
    pasta_log.mkdir(parents=True, exist_ok=True)

    registros = []
    for par in pares:
        imagem = carregar_bgr(par.caminho_imagem)
        mascara = carregar_mascara_binaria(par.caminho_mascara, so_classe_0=True)
        det, dados = analisar_imagem(
            imagem,
            mascara_ref=mascara,
            camera_id=args.camera,
            metodo=args.metodo,
            caminho_modelo=str(args.modelo),
        )
        dados["id_imagem"] = par.id_imagem
        dados["classes_xml"] = par.classes_xml
        cv2.imwrite(str(pasta_img / f"{par.id_imagem}_overlay.png"), det.overlay)
        cv2.imwrite(str(pasta_msk / f"{par.id_imagem}_pred.png"), det.mascara)
        registros.append(dados)
        iou = dados["iou"]
        print(
            f"{par.id_imagem:6s}  nivel={dados['alert_label']:26s}  "
            f"iou={iou:.3f}  tempo={dados['tempo_s']:.3f}s  regioes={dados['n_regioes']}"
        )

    saida = pasta_log / "semana2.json"
    saida.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nLog: {saida}")
    print("Sobreposições em resultados/imagens/")
    print("Lembrete: IoU baixo no Canny/Otsu é comum. Isso não invalida a Semana 2.")


if __name__ == "__main__":
    main()
