"""Leitura do DamCrack sem carregar o zip inteiro na RAM."""

from __future__ import annotations

import logging
import shutil
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from codigo import config

logger = logging.getLogger(__name__)


def _no_colab() -> bool:
    """
    Verifica se o ambiente de execução é o Google Colab.
    
    Returns:
        bool: True se for o Colab, False caso contrário.
    """
    return Path("/content").is_dir() and (
        "google.colab" in __import__("sys").modules or Path("/content/drive").exists()
    )


def garantir_zip_legivel(caminho_zip: Path | None = None) -> Path:
    """
    O Drive do Colab costuma quebrar ZipFile (BadZipFile): o FUSE não lê o
    diretório no fim do arquivo. Copia o zip para o disco local do Colab.
    
    Args:
        caminho_zip (Path | None): O caminho do arquivo zip. Usa o padrão se None.
        
    Returns:
        Path: O caminho validado ou o novo caminho copiado.
    """
    zip_path = Path(caminho_zip or config.ZIP_PADRAO)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip do DamCrack não encontrado: {zip_path}")

    tamanho = zip_path.stat().st_size
    with zip_path.open("rb") as f:
        magia = f.read(8)

    if tamanho < 1_000_000:
        raise zipfile.BadZipFile(
            f"Arquivo pequeno demais ({tamanho} bytes) para ser o DamCrack (~359 MB). "
            "Provavelmente o upload no Drive não terminou, ou é uma página HTML."
        )
    if magia.lstrip().lower().startswith(b"<") or magia.lower().startswith(b"<!do"):
        raise zipfile.BadZipFile(
            "O 'zip' na verdade é uma página HTML (download incompleto). "
            "Baixe de novo o Damage Segmentaion.zip e envie ao Drive até aparecer ~359 MB."
        )

    no_drive = "/drive/" in str(zip_path).replace("\\", "/")
    if _no_colab() and no_drive:
        local = Path("/content/damcrack_local.zip")
        precisa = (not local.is_file()) or local.stat().st_size != tamanho
        if precisa:
            logger.info(
                f"Copiando {tamanho / 1e6:.0f} MB do Drive para /content/damcrack_local.zip "
                "(evita BadZipFile no Colab). Pode levar uns minutos..."
            )
            shutil.copyfile(zip_path, local)
        zip_path = local
        config.definir_zip(zip_path)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.namelist()[:1]
    except zipfile.BadZipFile as exc:
        logger.error(f"BadZipFile ao abrir: {zip_path}")
        raise zipfile.BadZipFile(
            f"O Python não reconheceu como zip: {zip_path} ({tamanho / 1e6:.1f} MB). "
            "No Colab, espere o Drive terminar de sincronizar e rode de novo. "
            "O arquivo certo tem cerca de 359 MB e o nome Damage Segmentaion.zip."
        ) from exc
    return zip_path


@dataclass
class ParImagemMascara:
    """Um exemplo: imagem + máscara refinada + caixas do XML.
    
    Attributes:
        id_imagem (str): O identificador da imagem.
        caminho_imagem (Path): Caminho para a imagem extraída.
        caminho_mascara (Path): Caminho para a máscara binária.
        classes_xml (list[str]): Lista de rótulos definidos no arquivo XML.
    """
    id_imagem: str
    caminho_imagem: Path
    caminho_mascara: Path
    classes_xml: list[str]
    
    def __repr__(self) -> str:
        return f"ParImagemMascara(id={self.id_imagem}, classes={len(self.classes_xml)})"


def _ids_no_zip(zf: zipfile.ZipFile) -> list[str]:
    """
    Extrai todos os IDs únicos das imagens válidas de dentro do zip.
    
    Args:
        zf (zipfile.ZipFile): Arquivo ZIP aberto.
        
    Returns:
        list[str]: Lista ordenada de IDs de imagens.
    """
    ids = []
    prefixo = config.PASTA_IMAGENS + "/"
    for nome in zf.namelist():
        n = nome.replace("\\", "/")
        if n.lower().endswith(".jpg") and "/images/" in n.lower():
            ids.append(Path(n).stem)
            continue
        if n.startswith(prefixo) and n.lower().endswith(".jpg"):
            ids.append(Path(n).stem)
    return sorted(set(ids), key=lambda x: (len(x), x))


def listar_ids(caminho_zip: Path | None = None) -> list[str]:
    """
    Lista os IDs (ex.: A1, A10) sem extrair arquivos.
    
    Args:
        caminho_zip (Path | None): O caminho do arquivo zip. Usa o padrão se None.
        
    Returns:
        list[str]: Uma lista ordenada contendo os IDs.
    """
    zip_path = garantir_zip_legivel(caminho_zip)
    with zipfile.ZipFile(zip_path) as zf:
        return _ids_no_zip(zf)


def _ler_classes_xml(zf: zipfile.ZipFile, id_imagem: str) -> list[str]:
    """
    Lê os rótulos armazenados em um arquivo XML para a imagem especificada.
    
    Args:
        zf (zipfile.ZipFile): O arquivo ZIP já aberto.
        id_imagem (str): O ID da imagem.
        
    Returns:
        list[str]: Lista de classes presentes.
    """
    membro = f"{config.PASTA_XML}/{id_imagem}.xml"
    try:
        raiz = ET.fromstring(zf.read(membro))
    except KeyError:
        return []
    nomes = []
    for obj in raiz.findall("object"):
        nome = obj.findtext("name")
        if nome:
            nomes.append(nome)
    return nomes


def extrair_subconjunto(
    n: int = 20,
    caminho_zip: Path | None = None,
    destino: Path | None = None,
    ids: list[str] | None = None,
) -> list[ParImagemMascara]:
    """
    Copia só N exemplos para disco (imagem + máscara).
    Não descompacta os ~4000 pares de uma vez.
    
    Args:
        n (int): O número de imagens para extrair.
        caminho_zip (Path | None): O caminho do arquivo zip.
        destino (Path | None): O diretório destino.
        ids (list[str] | None): Lista com os IDs selecionados, se não os N primeiros são lidos.
        
    Returns:
        list[ParImagemMascara]: A lista de pares representativos da amostra extraída.
    """
    zip_path = garantir_zip_legivel(caminho_zip)
    destino = Path(destino or config.PASTA_AMOSTRA)
    pasta_img = destino / "imagens"
    pasta_msk = destino / "mascaras"
    pasta_img.mkdir(parents=True, exist_ok=True)
    pasta_msk.mkdir(parents=True, exist_ok=True)

    pares: list[ParImagemMascara] = []
    with zipfile.ZipFile(zip_path) as zf:
        todos = ids or _ids_no_zip(zf)
        escolhidos = todos[: max(1, n)]
        for id_imagem in escolhidos:
            membro_img = f"{config.PASTA_IMAGENS}/{id_imagem}.jpg"
            membro_msk = f"{config.PASTA_MASCARAS}/{id_imagem}.png"
            saida_img = pasta_img / f"{id_imagem}.jpg"
            saida_msk = pasta_msk / f"{id_imagem}.png"
            saida_img.write_bytes(zf.read(membro_img))
            saida_msk.write_bytes(zf.read(membro_msk))
            pares.append(
                ParImagemMascara(
                    id_imagem=id_imagem,
                    caminho_imagem=saida_img,
                    caminho_mascara=saida_msk,
                    classes_xml=_ler_classes_xml(zf, id_imagem),
                )
            )
    return pares


def listar_pares_extraidos(destino: Path | None = None) -> list[ParImagemMascara]:
    """
    Lista todos os pares de imagens e máscaras que já foram extraídas no diretório.
    
    Args:
        destino (Path | None): Diretório raiz da extração.
        
    Returns:
        list[ParImagemMascara]: Os pares de imagens extraídos disponíveis.
    """
    destino = Path(destino or config.PASTA_AMOSTRA)
    pasta_img = destino / "imagens"
    pasta_msk = destino / "mascaras"
    if not pasta_img.is_dir():
        return []
    pares = []
    for jpg in sorted(pasta_img.glob("*.jpg")):
        msk = pasta_msk / f"{jpg.stem}.png"
        pares.append(
            ParImagemMascara(
                id_imagem=jpg.stem,
                caminho_imagem=jpg,
                caminho_mascara=msk if msk.is_file() else Path(""),
                classes_xml=[],
            )
        )
    return pares


def carregar_bgr(caminho: Path) -> np.ndarray:
    """
    Carrega a imagem de um arquivo para matriz BGR.
    
    Args:
        caminho (Path): O caminho da imagem.
        
    Returns:
        np.ndarray: A imagem lida.
    """
    imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")
    return imagem


def mascara_para_binaria(
    mascara_bgr: np.ndarray,
    so_classe_0: bool = True,
) -> np.ndarray:
    """
    Converte máscara RGB do DamCrack em binário (0/255).

    so_classe_0=True: só o vermelho (possível rachadura).
    so_classe_0=False: qualquer dano (vermelho ou azul).
    
    Args:
        mascara_bgr (np.ndarray): A máscara multicor de entrada.
        so_classe_0 (bool): Filtro para pegar a classe 0 (vermelha).
        
    Returns:
        np.ndarray: Máscara resultante transformada em binária (255 nas detecções).
    """
    rgb = cv2.cvtColor(mascara_bgr, cv2.COLOR_BGR2RGB)
    vermelho = np.all(rgb == np.array(config.COR_CLASSE_0), axis=2)
    if so_classe_0:
        binaria = vermelho
    else:
        azul = np.all(rgb == np.array(config.COR_CLASSE_1), axis=2)
        binaria = vermelho | azul
    return (binaria.astype(np.uint8) * 255)


def carregar_mascara_binaria(caminho: Path, so_classe_0: bool = True) -> np.ndarray:
    """
    Lê um arquivo de máscara e converte para binária.
    
    Args:
        caminho (Path): Caminho para a imagem de máscara.
        so_classe_0 (bool): Retorna somente as anotações referentes à rachadura.
        
    Returns:
        np.ndarray: A máscara em formato binário.
    """
    mascara = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if mascara is None:
        raise FileNotFoundError(f"Não foi possível ler a máscara: {caminho}")
    return mascara_para_binaria(mascara, so_classe_0=so_classe_0)
