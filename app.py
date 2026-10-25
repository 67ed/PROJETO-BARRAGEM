#!/usr/bin/env python3
"""
Monitoramento visual de barragens.

    python app.py                 # webcam (cai para demo se não houver câmera)
    python app.py --demo          # DamCrack ou concreto sintético
    python app.py --video arquivo.mp4
    python app.py --imagem foto.jpg
    python app.py --metodo votacao

Teclas: q sair | espaço pausar | s salvar | m método | h ajuda | n próximo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

if "WAYLAND_DISPLAY" in os.environ and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from codigo import config
from codigo.camera import (
    FonteFrames,
    abrir_pasta_imagens,
    abrir_sintetico,
    abrir_video,
    abrir_webcam,
)
from codigo.data import extrair_subconjunto, listar_pares_extraidos
from codigo.detector import METODOS_VOTACAO
from codigo.logging_config import configurar_logging
from codigo.sistema import SistemaMonitoramento
from codigo.ui import desenhar_painel

logger = logging.getLogger(__name__)

METODOS_CICLO = (
    "votacao",
    *METODOS_VOTACAO,
    "consenso",
    "combinada",
)


def _fonte(args: argparse.Namespace) -> tuple[FonteFrames, str]:
    if args.imagem:
        caminho = Path(args.imagem)
        if not caminho.is_file():
            raise SystemExit(f"Imagem não encontrada: {caminho}")
        fonte = abrir_pasta_imagens(caminho.parent)
        fonte.imagens = [caminho]
        fonte.indice = 0
        return fonte, caminho.stem
    if args.demo:
        if args.zip.is_file():
            pares = extrair_subconjunto(n=args.n, caminho_zip=args.zip)
            pasta = pares[0].caminho_imagem.parent
            return abrir_pasta_imagens(pasta), "CAM-DEMO"
        pares = listar_pares_extraidos()
        if pares:
            pasta = pares[0].caminho_imagem.parent
            return abrir_pasta_imagens(pasta), "CAM-DEMO"
        logger.warning("DamCrack não encontrado; usando concreto sintético.")
        return abrir_sintetico(n=args.n), "CAM-SYN"
    if args.video:
        return abrir_video(Path(args.video)), Path(args.video).stem
    return abrir_webcam(args.camera), f"CAM{args.camera:02d}"


def _desenhar_botoes(tela: np.ndarray, pausado: bool) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    vis = tela.copy()
    altura, largura = vis.shape[:2]
    altura_barra = 50
    y_barra = altura - altura_barra
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, y_barra), (largura, altura), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)

    botao_largura, botao_altura = 190, 34
    margem = 12
    x_captura = largura - botao_largura - margem
    y = altura - botao_altura - (altura_barra - botao_altura) // 2
    rect_captura = (x_captura, y, botao_largura, botao_altura)
    largura_sair = 86
    x_sair = x_captura - largura_sair - margem
    rect_sair = (x_sair, y, largura_sair, botao_altura)

    fonte = cv2.FONT_HERSHEY_DUPLEX
    for rect, fundo, texto in (
        (rect_captura, (52, 168, 72) if pausado else (46, 46, 210), "RETOMAR" if pausado else "PAUSAR"),
        (rect_sair, (70, 70, 70), "SAIR"),
    ):
        x, yb, bw, bh = rect
        cv2.rectangle(vis, (x + 2, yb + 2), (x + bw + 2, yb + bh + 2), (0, 0, 0), -1)
        cv2.rectangle(vis, (x, yb), (x + bw, yb + bh), fundo, -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(texto, fonte, 0.5, 1)
        cv2.putText(
            vis,
            texto,
            (x + (bw - tw) // 2, yb + (bh + th) // 2 - 1),
            fonte,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    dica = "q sair  espaco pausar  s salvar  m metodo  h ajuda  n proximo"
    cv2.putText(vis, dica, (14, altura - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return vis, {"captura": rect_captura, "sair": rect_sair}


def _clicou_no_retangulo(x: int, y: int, rect: tuple[int, int, int, int] | None) -> bool:
    if rect is None:
        return False
    rx, ry, largura, altura = rect
    return rx <= x <= rx + largura and ry <= y <= ry + altura


def _proximo_metodo(atual: str) -> str:
    if atual in METODOS_CICLO:
        return METODOS_CICLO[(METODOS_CICLO.index(atual) + 1) % len(METODOS_CICLO)]
    return METODOS_CICLO[0]


def main() -> None:
    configurar_logging("INFO")
    parser = argparse.ArgumentParser(description="Detector visual de rachaduras em barragens")
    parser.add_argument("--camera", type=int, default=0, help="índice da webcam")
    parser.add_argument("--video", type=str, default="", help="arquivo de vídeo")
    parser.add_argument("--imagem", type=str, default="", help="analisar uma foto")
    parser.add_argument("--demo", action="store_true", help="DamCrack ou concreto sintético")
    parser.add_argument("--n", type=int, default=12, help="quantas imagens no --demo")
    parser.add_argument("--zip", type=Path, default=config.ZIP_PADRAO)
    parser.add_argument(
        "--metodo",
        choices=[
            "canny",
            "otsu",
            "adaptativo",
            "blackhat",
            "tophat",
            "contraste_local",
            "sobel",
            "frangi",
            "combinada",
            "treinado",
            "votacao",
            "consenso",
        ],
        default="votacao",
        help="votacao emite aviso urgente quando 3 de 5 métodos concordam",
    )
    parser.add_argument("--modelo", type=Path, default=config.MODELO_SIMPLES_PADRAO)
    parser.add_argument("--cada", type=int, default=1, help="analisar 1 a cada N frames")
    parser.add_argument("--intervalo", type=float, default=2.0, help="segundos entre análises")
    parser.add_argument("--atraso", type=int, default=30, help="ms entre frames (demo/vídeo)")
    args = parser.parse_args()

    try:
        fonte, camera_id = _fonte(args)
    except RuntimeError as exc:
        logger.warning("%s", exc)
        logger.info("Caindo para demo...")
        args.demo = True
        try:
            fonte, camera_id = _fonte(args)
        except Exception:
            fonte, camera_id = abrir_sintetico(n=args.n), "CAM-SYN"

    sistema = SistemaMonitoramento(
        camera_id=camera_id,
        metodo=args.metodo,
        caminho_modelo=str(args.modelo),
    )
    pasta_log = config.PASTA_RESULTADOS / "registros"
    pasta_img = config.PASTA_RESULTADOS / "deteccoes"
    pasta_log.mkdir(parents=True, exist_ok=True)
    pasta_img.mkdir(parents=True, exist_ok=True)
    log_path = pasta_log / "ao_vivo.jsonl"

    nome_janela = "Barragem - detector de rachaduras"
    logger.info("Janela OpenCV pronta. Tecla h mostra os atalhos.")
    estado = {"pausado": False, "botoes": {}, "encerrar": False, "ajuda": False, "forcar": False}
    n_frame = 0
    n_alertas = 0
    ultimo_processamento = float("-inf")
    overlay = None
    dados = None
    fps = 0.0
    t_fps = time.monotonic()

    def ao_clicar(evento: int, x: int, y: int, _flags: int, _param: object) -> None:
        if evento != cv2.EVENT_LBUTTONDOWN:
            return
        if _clicou_no_retangulo(x, y, estado["botoes"].get("captura")):
            estado["pausado"] = not estado["pausado"]
        elif _clicou_no_retangulo(x, y, estado["botoes"].get("sair")):
            estado["encerrar"] = True

    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome_janela, 1100, 720)
    try:
        cv2.setMouseCallback(nome_janela, ao_clicar)
    except cv2.error as exc:
        logger.warning("Nao foi possivel registrar clique do mouse: %s", exc)

    try:
        with log_path.open("a", encoding="utf-8") as f:
            while not estado["encerrar"]:
                if not estado["pausado"] or overlay is None or estado["forcar"]:
                    frame = fonte.ler()
                    if frame is None:
                        if args.demo or args.imagem:
                            time.sleep(0.05)
                            continue
                        logger.info("Fim da fonte (vídeo/câmera).")
                        break
                    n_frame += 1
                    agora = time.monotonic()
                    deve_analisar = (
                        overlay is None
                        or estado["forcar"]
                        or (
                            n_frame % max(1, args.cada) == 0
                            and agora - ultimo_processamento >= max(0.0, args.intervalo)
                        )
                    )
                    estado["forcar"] = False
                    if deve_analisar:
                        overlay, _det, dados = sistema.processar(frame)
                        ultimo_processamento = agora
                        dt = agora - t_fps
                        fps = (1.0 / dt) if dt > 0 else 0.0
                        t_fps = agora
                        f.write(json.dumps(dados, ensure_ascii=False) + "\n")
                        f.flush()
                        if dados.get("alert_level") not in (None, "nenhum"):
                            n_alertas += 1

                if overlay is not None and dados is not None:
                    tela = desenhar_painel(
                        overlay,
                        dados,
                        fps=fps,
                        pausado=estado["pausado"],
                        ajuda=estado["ajuda"],
                    )
                    tela_exibida, estado["botoes"] = _desenhar_botoes(tela, estado["pausado"])
                    cv2.imshow(nome_janela, tela_exibida)

                tecla = cv2.waitKey(args.atraso if args.demo or args.video or args.imagem else 1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord(" "):
                    estado["pausado"] = not estado["pausado"]
                if tecla == ord("h"):
                    estado["ajuda"] = not estado["ajuda"]
                if tecla == ord("m"):
                    sistema.metodo = _proximo_metodo(sistema.metodo)
                    sistema.rastreador.contagem = 0
                    estado["forcar"] = True
                    logger.info("Método: %s", sistema.metodo)
                if tecla == ord("n"):
                    fonte.pular()
                    estado["forcar"] = True
                    estado["pausado"] = False
                if tecla == ord("s") and overlay is not None and dados is not None:
                    nome = pasta_img / f"captura_{int(time.time())}.png"
                    captura = desenhar_painel(overlay, dados, fps=fps, pausado=estado["pausado"])
                    cv2.imwrite(str(nome), captura)
                    logger.info("Frame salvo em %s", nome)
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
        logger.info("Frames analisados: %s | alertas: %s | log: %s", n_frame, n_alertas, log_path)


if __name__ == "__main__":
    main()
