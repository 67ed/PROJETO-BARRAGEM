#!/usr/bin/env python3
"""
Sistema completo para testar no VS Code.

Câmera (ou vídeo / imagens) → detector → persistência → alerta → bot na tela.

    cd ~/projeto_barragem
    source .venv/bin/activate
    python app.py                 # webcam
    python app.py --demo          # fotos do DamCrack (sem câmera)
    python app.py --video arquivo.mp4
    python app.py --camera 0
    python app.py --metodo treinado  # após rodar treinar_modelo_simples.py

Teclas:  q = sair   s = salvar frame   espaço = pausar
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from codigo import config
from codigo.camera import FonteFrames, abrir_pasta_imagens, abrir_video, abrir_webcam
from codigo.data import extrair_subconjunto
from codigo.logging_config import configurar_logging
from codigo.sistema import SistemaMonitoramento

logger = logging.getLogger(__name__)


def _fonte(args: argparse.Namespace) -> tuple[FonteFrames, str]:
    if args.demo:
        pares = extrair_subconjunto(n=args.n, caminho_zip=args.zip)
        pasta = pares[0].caminho_imagem.parent
        fonte = abrir_pasta_imagens(pasta)
        return fonte, "CAM-DEMO"
    if args.video:
        fonte = abrir_video(Path(args.video))
        return fonte, Path(args.video).stem
    fonte = abrir_webcam(args.camera)
    return fonte, f"CAM{args.camera:02d}"


def _desenhar_botoes(tela: np.ndarray, pausado: bool) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Desenha botões de controle com um estilo mais moderno e bonito."""
    vis = tela.copy()
    altura, largura = vis.shape[:2]
    
    # Adicionar uma barra inferior semitransparente para os controles
    altura_barra = 50
    y_barra = altura - altura_barra
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, y_barra), (largura, altura), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

    botao_largura, botao_altura = 190, 34
    margem = 10
    x_captura = largura - botao_largura - margem
    y = altura - botao_altura - (altura_barra - botao_altura) // 2

    rect_captura = (x_captura, y, botao_largura, botao_altura)
    largura_sair = 80
    x_sair = x_captura - largura_sair - margem
    rect_sair = (x_sair, y, largura_sair, botao_altura)

    # Cores modernas (BGR)
    cor_btn_pausado = (60, 180, 60)      # Verde suave
    cor_btn_rodando = (60, 60, 220)      # Vermelho suave
    cor_btn_sair = (80, 80, 80)          # Cinza escuro
    cor_texto = (255, 255, 255)
    cor_borda = (200, 200, 200)

    # Botão de captura
    cor_fundo = cor_btn_pausado if pausado else cor_btn_rodando
    texto_captura = "RETOMAR CAPTURA" if pausado else "PARAR CAPTURA"
    
    # Sombra do botão captura
    cv2.rectangle(vis, (x_captura + 2, y + 2), (x_captura + botao_largura + 2, y + botao_altura + 2), (0, 0, 0), -1)
    # Fundo do botão captura
    cv2.rectangle(vis, (x_captura, y), (x_captura + botao_largura, y + botao_altura), cor_fundo, -1)
    # Borda do botão captura
    cv2.rectangle(vis, (x_captura, y), (x_captura + botao_largura, y + botao_altura), cor_borda, 1, cv2.LINE_AA)
    
    # Texto botão captura
    fonte = cv2.FONT_HERSHEY_DUPLEX
    escala_fonte = 0.5
    espessura = 1
    (tw, th), _ = cv2.getTextSize(texto_captura, fonte, escala_fonte, espessura)
    tx = x_captura + (botao_largura - tw) // 2
    ty = y + (botao_altura + th) // 2 - 1
    cv2.putText(vis, texto_captura, (tx, ty), fonte, escala_fonte, cor_texto, espessura, cv2.LINE_AA)

    # Botão de sair
    # Sombra
    cv2.rectangle(vis, (x_sair + 2, y + 2), (x_sair + largura_sair + 2, y + botao_altura + 2), (0, 0, 0), -1)
    # Fundo
    cv2.rectangle(vis, (x_sair, y), (x_sair + largura_sair, y + botao_altura), cor_btn_sair, -1)
    # Borda
    cv2.rectangle(vis, (x_sair, y), (x_sair + largura_sair, y + botao_altura), cor_borda, 1, cv2.LINE_AA)
    
    texto_sair = "SAIR"
    (tw, th), _ = cv2.getTextSize(texto_sair, fonte, escala_fonte, espessura)
    tx = x_sair + (largura_sair - tw) // 2
    ty = y + (botao_altura + th) // 2 - 1
    cv2.putText(vis, texto_sair, (tx, ty), fonte, escala_fonte, cor_texto, espessura, cv2.LINE_AA)

    return vis, {"captura": rect_captura, "sair": rect_sair}


def _clicou_no_retangulo(x: int, y: int, rect: tuple[int, int, int, int] | None) -> bool:
    if rect is None:
        return False
    rx, ry, largura, altura = rect
    return rx <= x <= rx + largura and ry <= y <= ry + altura


def main() -> None:
    configurar_logging("INFO")
    parser = argparse.ArgumentParser(description="Monitoramento barragem — teste local")
    parser.add_argument("--camera", type=int, default=0, help="índice da webcam")
    parser.add_argument("--video", type=str, default="", help="arquivo de vídeo")
    parser.add_argument("--demo", action="store_true", help="usar imagens do DamCrack")
    parser.add_argument("--n", type=int, default=12, help="quantas imagens no --demo")
    parser.add_argument("--zip", type=Path, default=config.ZIP_PADRAO)
    parser.add_argument(
        "--metodo",
        choices=["canny", "otsu", "adaptativo", "blackhat", "tophat", "contraste_local", "sobel", "frangi", "combinada", "treinado", "votacao", "consenso"],
        default="votacao",
        help="votacao emite aviso urgente quando 3 de 5 métodos concordam na mesma região",
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=config.MODELO_SIMPLES_PADRAO,
        help="arquivo .npz gerado por treinar_modelo_simples.py",
    )
    parser.add_argument("--cada", type=int, default=1, help="analisar 1 a cada N frames")
    parser.add_argument(
        "--intervalo",
        type=float,
        default=2.0,
        help="segundos entre capturas analisadas; 2 é mais leve e reduz alertas repetidos",
    )
    parser.add_argument("--atraso", type=int, default=30, help="ms entre frames (demo/vídeo)")
    args = parser.parse_args()

    if args.demo and not args.zip.is_file():
        logger.error(f"Para --demo precisa do zip: {args.zip}")
        raise SystemExit(f"Para --demo precisa do zip: {args.zip}")

    try:
        fonte, camera_id = _fonte(args)
    except RuntimeError as exc:
        logger.warning(str(exc))
        logger.info("Caindo para --demo com o conjunto de dados...")
        if not args.zip.is_file():
            logger.error("Sem câmera e sem zip do DamCrack. Não dá para testar.")
            raise SystemExit("Sem câmera e sem zip do DamCrack. Não dá para testar.")
        args.demo = True
        fonte, camera_id = _fonte(args)

    sistema = SistemaMonitoramento(
        camera_id=camera_id,
        metodo=args.metodo,
        caminho_modelo=str(args.modelo),
    )
    pasta_log = config.PASTA_RESULTADOS / "registros"
    pasta_img = config.PASTA_RESULTADOS / "imagens"
    pasta_log.mkdir(parents=True, exist_ok=True)
    pasta_img.mkdir(parents=True, exist_ok=True)
    log_path = pasta_log / "ao_vivo.jsonl"

    nome_janela = "Monitoramento Barragem"
    logger.info(
        "Janela do OpenCV. Clique em PARAR CAPTURA para pausar. "
        "Clique em SAIR, q ou Esc para fechar."
    )
    estado = {"pausado": False, "botoes": {}, "encerrar": False}

    def ao_clicar(evento: int, x: int, y: int, _flags: int, _param: any) -> None:
        if evento != cv2.EVENT_LBUTTONDOWN:
            return
        if _clicou_no_retangulo(x, y, estado["botoes"].get("captura")):
            estado["pausado"] = not estado["pausado"]
        elif _clicou_no_retangulo(x, y, estado["botoes"].get("sair")):
            estado["encerrar"] = True

    cv2.namedWindow(nome_janela, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(nome_janela, ao_clicar)
    n_frame = 0
    ultimo_processamento = float("-inf")
    tela = None
    dados = None

    try:
        with log_path.open("a", encoding="utf-8") as f:
            while not estado["encerrar"]:
                if not estado["pausado"]:
                    frame = fonte.ler()
                    if frame is None:
                        if args.demo:
                            time.sleep(0.1) # Evitar consumo 100% de CPU
                            continue
                        logger.info("Fim da fonte (vídeo/câmera).")
                        break
                    n_frame += 1
                    agora = time.monotonic()
                    deve_analisar = (
                        tela is None
                        or (
                            n_frame % max(1, args.cada) == 0
                            and agora - ultimo_processamento >= max(0.0, args.intervalo)
                        )
                    )
                    if deve_analisar:
                        tela, _det, dados = sistema.processar(frame)
                        ultimo_processamento = agora
                        f.write(json.dumps(dados, ensure_ascii=False) + "\n")
                        f.flush()

                if tela is not None:
                    tela_exibida, estado["botoes"] = _desenhar_botoes(
                        tela, estado["pausado"]
                    )
                    cv2.imshow(nome_janela, tela_exibida)

                tecla = cv2.waitKey(args.atraso if args.demo or args.video else 1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord(" "):
                    estado["pausado"] = not estado["pausado"]
                if tecla == ord("s") and tela is not None:
                    nome = pasta_img / f"captura_{int(time.time())}.png"
                    cv2.imwrite(str(nome), tela)
                    logger.info(f"Frame salvo em {nome}")
                try:
                    if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário.")
    finally:
        fonte.soltar()
        cv2.destroyAllWindows()
        logger.info(f"Log de detecção salvo em: {log_path}")


if __name__ == "__main__":
    main()
