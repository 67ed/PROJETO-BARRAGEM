"""Detector: junta pré-processamento + segmentação + regiões."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import time

import cv2
import numpy as np

from codigo.analysis import Regiao, iou_mascaras, regioes_da_mascara, sobrepor
from codigo import preprocessing, segmentation
from codigo import config
from codigo.modelo_simples import segmentar_com_modelo

logger = logging.getLogger(__name__)

# Métodos independentes usados para o aviso prioritário. A votação é espacial:
# os votos precisam coincidir na mesma região da imagem.
METODOS_VOTACAO = ("canny", "tophat", "sobel", "frangi", "treinado")
METODOS_CONSENSO_ESTRITO = ("canny", "otsu", "adaptativo", "blackhat", "treinado")


@dataclass
class ResultadoDetector:
    """
    Armazena o resultado de uma detecção e fornece metadados do processamento.
    
    Attributes:
        mascara (np.ndarray): A máscara binária indicando danos.
        regioes (list[Regiao]): Lista de regiões de danos identificadas.
        overlay (np.ndarray): A imagem de visualização com os danos sobrepostos.
        tempo_s (float): Tempo em segundos gasto na detecção.
        iou_referencia (float | None): O valor IoU, caso máscara de referência exista.
        extra (dict): Informações adicionais do processo, como votação.
    """
    mascara: np.ndarray
    regioes: list[Regiao]
    overlay: np.ndarray
    tempo_s: float
    iou_referencia: float | None = None
    extra: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"ResultadoDetector(regioes={len(self.regioes)}, tempo_s={self.tempo_s:.3f})"


def _segmentar(
    metodo: str,
    cinza_suave: np.ndarray,
    imagem_bgr: np.ndarray,
    caminho_modelo: str | None,
) -> np.ndarray:
    """
    Aplica o método de segmentação especificado.
    
    Args:
        metodo (str): Nome do método a ser usado.
        cinza_suave (np.ndarray): Imagem de entrada em escala de cinza e suavizada.
        imagem_bgr (np.ndarray): Imagem original BGR.
        caminho_modelo (str | None): Caminho para o arquivo do modelo simples, caso usado.
        
    Returns:
        np.ndarray: A máscara de segmentação gerada.
    """
    if metodo == "canny":
        return segmentation.por_canny(cinza_suave)
    if metodo == "otsu":
        return segmentation.por_otsu(cinza_suave)
    if metodo == "adaptativo":
        return segmentation.por_adaptativo(cinza_suave)
    if metodo == "blackhat":
        return segmentation.por_blackhat(cinza_suave)
    if metodo == "tophat":
        return segmentation.por_tophat(cinza_suave)
    if metodo == "contraste_local":
        return segmentation.por_contraste_local(cinza_suave)
    if metodo == "sobel":
        return segmentation.por_sobel(cinza_suave)
    if metodo == "frangi":
        return segmentation.por_frangi(cinza_suave)
    if metodo == "combinada":
        return segmentation.combinada(cinza_suave)
    if metodo == "treinado":
        return segmentar_com_modelo(imagem_bgr, caminho_modelo=caminho_modelo)
    
    opcoes = ", ".join(
        (
            *METODOS_VOTACAO,
            "otsu",
            "blackhat",
            "tophat",
            "combinada",
            "treinado",
            "votacao",
            "consenso",
        )
    )
    raise ValueError(f"Método desconhecido: {metodo}. Opções: {opcoes}.")


def _votar(mascaras: dict[str, np.ndarray], votos_minimos: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Conta votos com tolerância de um pixel entre as bordas dos métodos.
    
    Args:
        mascaras (dict[str, np.ndarray]): Dicionário com máscaras geradas por cada método.
        votos_minimos (int): O mínimo de votos necessários para considerar um pixel como dano.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: Uma tupla com a máscara filtrada e a imagem de votos brutos.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    votos = np.zeros_like(next(iter(mascaras.values())), dtype=np.uint8)
    for mascara_metodo in mascaras.values():
        votos += (cv2.dilate(mascara_metodo, kernel, iterations=1) > 0).astype(np.uint8)
    combinada = (votos >= votos_minimos).astype(np.uint8) * 255
    combinada = segmentation.limpar_componentes(combinada)
    return segmentation.filtrar_regioes_votacao(combinada), votos


def detectar(
    imagem_bgr: np.ndarray,
    mascara_referencia: np.ndarray | None = None,
    metodo: str = "votacao",
    caminho_modelo: str | None = None,
) -> ResultadoDetector:
    """
    Executa a pipeline de detecção completa em uma imagem.
    
    metodo: 'canny' | 'otsu' | 'adaptativo' | 'blackhat' | 'tophat' |
            'contraste_local' | 'sobel' | 'frangi' | 'combinada' | 'treinado' |
            'votacao' | 'consenso'
    Se mascara_referencia for dada, calcula IoU (avaliação, não diagnóstico).
    
    Args:
        imagem_bgr (np.ndarray): Imagem de entrada no formato BGR.
        mascara_referencia (np.ndarray | None): Máscara de referência para calcular o IoU.
        metodo (str): O método de detecção a ser usado.
        caminho_modelo (str | None): O caminho do arquivo do modelo treinado.
        
    Returns:
        ResultadoDetector: Os resultados e as métricas da detecção efetuada.
    """
    inicio = time.perf_counter()
    preparado = preprocessing.preparar(imagem_bgr)
    suave = preparado["suave"]
    bgr = preparado["bgr"]

    metodos_votacao: list[str] = []
    coberturas_por_metodo: dict[str, float] = {}
    metodos_descartados: list[str] = []
    metodos_substituidos: dict[str, str] = {}
    votos_minimos = 0
    maior_votacao = 0
    
    if metodo in {"votacao", "consenso"}:
        nomes = METODOS_VOTACAO if metodo == "votacao" else METODOS_CONSENSO_ESTRITO
        votos_minimos = config.VOTOS_MINIMOS_URGENTE if metodo == "votacao" else len(nomes)
        mascaras = {}
        for nome in nomes:
            try:
                mascaras[nome] = _segmentar(nome, suave, bgr, caminho_modelo)
            except FileNotFoundError:
                if metodo != "votacao" or nome != "treinado":
                    raise
                substituto = "contraste_local"
                mascaras[substituto] = _segmentar(substituto, suave, bgr, caminho_modelo)
                metodos_substituidos[nome] = substituto
                logger.warning(f"Método '{nome}' não encontrado, substituindo por '{substituto}'.")
                
        mascaras_para_voto = {}
        for nome, mascara_metodo in mascaras.items():
            cobertura = float(np.count_nonzero(mascara_metodo) / mascara_metodo.size)
            coberturas_por_metodo[nome] = round(cobertura, 4)
            if metodo == "votacao" and cobertura > config.COBERTURA_MAXIMA_METODO_VOTO:
                metodos_descartados.append(nome)
                mascaras_para_voto[nome] = np.zeros_like(mascara_metodo)
            else:
                mascaras_para_voto[nome] = mascara_metodo
                
        mascara, votos = _votar(mascaras_para_voto, votos_minimos)
        metodos_votacao = list(mascaras)
        maior_votacao = int(votos.max())
    else:
        mascara = _segmentar(metodo, suave, bgr, caminho_modelo)

    regioes = regioes_da_mascara(mascara)
    overlay = sobrepor(bgr, mascara)
    iou = None
    
    if mascara_referencia is not None:
        ref = preprocessing.redimensionar_se_preciso(
            mascara_referencia, interpolation=cv2.INTER_NEAREST
        )
        if ref.ndim == 3:
            raise ValueError("A máscara de referência deve ser binária (1 canal).")
        if ref.shape[:2] != mascara.shape[:2]:
            ref = cv2.resize(
                ref, (mascara.shape[1], mascara.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        iou = iou_mascaras(mascara, ref)

    return ResultadoDetector(
        mascara=mascara,
        regioes=regioes,
        overlay=overlay,
        tempo_s=time.perf_counter() - inicio,
        iou_referencia=iou,
        extra={
            "metodo": metodo,
            "metodos_votacao": metodos_votacao,
            "coberturas_por_metodo": coberturas_por_metodo,
            "metodos_descartados": metodos_descartados,
            "metodos_substituidos": metodos_substituidos,
            "votos_minimos": votos_minimos,
            "maior_votacao": maior_votacao,
            "confirmado_por_metodos": bool(regioes) and maior_votacao >= votos_minimos > 0,
        },
    )
