"""Calibração visual simples usando o DamCrack, uma imagem por vez."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import cv2
import numpy as np

from codigo import config
from codigo.data import garantir_zip_legivel, listar_ids, mascara_para_binaria
from codigo.modelo_simples import BINS, _indices_caracteristicas

logger = logging.getLogger(__name__)


def _ler_imagem(zf: zipfile.ZipFile, membro: str) -> np.ndarray:
    """
    Lê a matriz de bytes correspondente de dentro de um ZIP e decodifica para BGR.
    
    Args:
        zf (zipfile.ZipFile): O arquivo zip.
        membro (str): Caminho do membro dentro do zip.
        
    Returns:
        np.ndarray: A imagem carregada do arquivo.
    """
    dados = np.frombuffer(zf.read(membro), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if imagem is None:
        raise RuntimeError(f"Não foi possível ler {membro} dentro do ZIP.")
    return imagem


def _imagem_e_mascara(zf: zipfile.ZipFile, id_imagem: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Recupera a imagem base e sua respectiva máscara binária de classificação.
    
    Args:
        zf (zipfile.ZipFile): O arquivo zip.
        id_imagem (str): O identificador da imagem alvo.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: A imagem original BGR e a máscara binária booleana.
    """
    imagem = _ler_imagem(zf, f"{config.PASTA_IMAGENS}/{id_imagem}.jpg")
    rotulo = _ler_imagem(zf, f"{config.PASTA_MASCARAS}/{id_imagem}.png")
    return imagem, mascara_para_binaria(rotulo, so_classe_0=True) > 0


def _separar_ids(
    caminho_zip: Path, amostras: int, validacao: float, seed: int
) -> tuple[list[str], list[str]]:
    """
    Divide a coleção de imagens nos grupos de treinamento e validação.
    
    Args:
        caminho_zip (Path): O caminho absoluto para o zip.
        amostras (int): Total de amostras a considerar.
        validacao (float): A fração (0.0-1.0) alocada à validação.
        seed (int): Semente de aleatoriedade para reprodutibilidade.
        
    Returns:
        tuple[list[str], list[str]]: Listas com IDs para treino e validação, respectivamente.
    """
    ids = listar_ids(caminho_zip)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    if amostras > 0:
        ids = ids[:amostras]
    if len(ids) < 2:
        raise ValueError("Use pelo menos duas imagens para treino e validação.")
    n_validacao = max(1, min(len(ids) - 1, round(len(ids) * validacao)))
    return ids[n_validacao:], ids[:n_validacao]


def _histogramas(caminho_zip: Path, ids: list[str], mostrar_progresso: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Acumula somente uma tabela 32 × 32 para cada classe (positivo/negativo).
    
    Args:
        caminho_zip (Path): O caminho absoluto para o zip.
        ids (list[str]): Os identificadores das imagens a processar.
        mostrar_progresso (bool): Ativa o envio de logs informativos durante o processo.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: Matrizes contendo o total de pixels por bin (positivo, negativo).
    """
    hist_positivo = np.zeros((BINS, BINS), dtype=np.int64)
    hist_negativo = np.zeros((BINS, BINS), dtype=np.int64)
    with zipfile.ZipFile(caminho_zip) as zf:
        for indice, id_imagem in enumerate(ids, start=1):
            imagem, mascara = _imagem_e_mascara(zf, id_imagem)
            indice_escuro, indice_borda = _indices_caracteristicas(imagem, BINS)
            combinados = indice_escuro * BINS + indice_borda
            hist_positivo += np.bincount(combinados[mascara], minlength=BINS * BINS).reshape(BINS, BINS)
            hist_negativo += np.bincount(combinados[~mascara], minlength=BINS * BINS).reshape(BINS, BINS)
            if mostrar_progresso and indice % 100 == 0:
                logger.info(f"Calibrando: {indice}/{len(ids)} imagens")
    return hist_positivo, hist_negativo


def _treinar_tabela(
    hist_positivo: np.ndarray, hist_negativo: np.ndarray, hist_val_positivo: np.ndarray, hist_val_negativo: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Escolhe células da tabela que dão o maior IoU no conjunto de validação.
    
    Args:
        hist_positivo (np.ndarray): Ocultamento das amostras positivas no treino.
        hist_negativo (np.ndarray): Ocultamento das amostras negativas no treino.
        hist_val_positivo (np.ndarray): Ocultamento positivo da validação.
        hist_val_negativo (np.ndarray): Ocultamento negativo da validação.
        
    Returns:
        tuple[np.ndarray, float]: A tabela binária das escolhas ideais e o valor limite utilizado.
    """
    probabilidade = (hist_positivo + 1.0) / (hist_positivo + hist_negativo + 2.0)
    ordem = np.argsort(probabilidade.ravel())[::-1]
    positivos_validacao = hist_val_positivo.ravel()[ordem]
    negativos_validacao = hist_val_negativo.ravel()[ordem]
    verdadeiros_positivos = np.cumsum(positivos_validacao)
    falsos_positivos = np.cumsum(negativos_validacao)
    total_positivos = int(hist_val_positivo.sum())
    iou = verdadeiros_positivos / np.maximum(total_positivos + falsos_positivos, 1)
    melhor = int(np.argmax(iou))
    corte = float(probabilidade.ravel()[ordem[melhor]])
    return probabilidade >= corte, corte

def _avaliar(tabela: np.ndarray, hist_positivo: np.ndarray, hist_negativo: np.ndarray) -> dict:
    """
    Avalia a tabela final do modelo em comparação à realidade de validação.
    
    Args:
        tabela (np.ndarray): Os bins classificados como positivos.
        hist_positivo (np.ndarray): Contagem correta da validação.
        hist_negativo (np.ndarray): Contagem incorreta da validação.
        
    Returns:
        dict: Métricas baseadas em precisão, recall e iou_validacao.
    """
    verdadeiros_positivos = int(hist_positivo[tabela].sum())
    falsos_positivos = int(hist_negativo[tabela].sum())
    falsos_negativos = int(hist_positivo[~tabela].sum())
    intersecao = verdadeiros_positivos
    uniao = verdadeiros_positivos + falsos_positivos + falsos_negativos
    return {
        "iou_validacao": round(intersecao / uniao, 4) if uniao else 1.0,
        "precisao_validacao": round(
            verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos), 4
        )
        if verdadeiros_positivos + falsos_positivos
        else 0.0,
        "recall_validacao": round(
            verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos), 4
        )
        if verdadeiros_positivos + falsos_negativos
        else 0.0,
    }


def treinar_modelo_simples(
    caminho_zip: Path | None = None,
    saida_modelo: Path | None = None,
    amostras: int = 300,
    validacao: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Treina com pouca RAM: conserva apenas duas tabelas 32 × 32.
    
    Args:
        caminho_zip (Path | None): O caminho do repositório ZIP.
        saida_modelo (Path | None): O local onde o modelo treinado final e suas métricas serão salvos.
        amostras (int): O limite da amostra.
        validacao (float): Fração atribuída aos dados de validação.
        seed (int): Estado inicial de aleatoriedade do conjunto de amostras.
        
    Returns:
        dict: O histórico resultante do treinamento contendo as métricas de validação avaliadas.
    """
    if amostras < 0 or not 0.0 < validacao < 0.5:
        raise ValueError("Amostras deve ser zero ou positivo e validação entre 0 e 0,5.")
    caminho_zip = garantir_zip_legivel(caminho_zip)
    ids_treino, ids_validacao = _separar_ids(caminho_zip, amostras, validacao, seed)
    hist_treino_positivo, hist_treino_negativo = _histogramas(caminho_zip, ids_treino, True)
    hist_val_positivo, hist_val_negativo = _histogramas(caminho_zip, ids_validacao, False)
    
    if not hist_treino_positivo.sum() or not hist_treino_negativo.sum():
        raise RuntimeError("As máscaras não têm pixels suficientes para calibrar o modelo.")
        
    tabela, corte = _treinar_tabela(
        hist_treino_positivo, hist_treino_negativo, hist_val_positivo, hist_val_negativo
    )
    metricas = _avaliar(tabela, hist_val_positivo, hist_val_negativo)

    saida_modelo = Path(saida_modelo or config.MODELO_SIMPLES_PADRAO)
    saida_modelo.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(saida_modelo, tabela=tabela.astype(np.uint8), bins=np.array(BINS))
    resultado = {
        "modelo": "tabela_escurecimento_e_borda",
        "corte": round(corte, 6),
        "amostras_treino": len(ids_treino),
        "amostras_validacao": len(ids_validacao),
        "pixels_positivos_treino": int(hist_treino_positivo.sum()),
        "pixels_negativos_treino": int(hist_treino_negativo.sum()),
        **metricas,
    }
    saida_modelo.with_suffix(".json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resultado
