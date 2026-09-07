"""Parâmetros configuráveis do protótipo (Semana 2)."""

from __future__ import annotations

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RAIZ_PROJETO: Path = Path(__file__).resolve().parent.parent

# O nome original da pasta no dataset tem um erro de digitação ("Segmentaion")
PASTA_DENTRO_DO_ZIP: str = "Damage Segmentaion"

# Caminhos dentro do zip
PASTA_IMAGENS: str = f"{PASTA_DENTRO_DO_ZIP}/Images"
PASTA_MASCARAS: str = f"{PASTA_DENTRO_DO_ZIP}/Refined Labels/Mask"
PASTA_XML: str = f"{PASTA_DENTRO_DO_ZIP}/Refined Labels/Pascal Voc"

# Máscaras coloridas do DamCrack (Refined Labels/Mask):
# vermelho = classe 0 (tratamos como possível rachadura)
# azul     = classe 1 (outro dano, em geral spalling — confirmar no artigo)
COR_CLASSE_0: tuple[int, int, int] = (255, 0, 0)  # RGB
COR_CLASSE_1: tuple[int, int, int] = (0, 0, 255)

NOMES_CLASSE: dict[int, str] = {
    0: "possivel_rachadura",
    1: "outro_dano",
}

# Constantes de pré-processamento e análise de imagem
TAMANHO_MAXIMO: int = 640
KERNEL_BLUR: tuple[int, int] = (5, 5)
LIMIAR_CANNY_BAIXO: int = 70
LIMIAR_CANNY_ALTO: int = 180
KERNEL_MORFOLOGIA: int = 3
AREA_MINIMA_REGIAO: float = 80.0

# Pré-processamento e remoção de ruído. São operações locais de CPU e não usam
# rede neural; são aplicadas a qualquer foto, webcam ou vídeo antes da detecção.
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_GRADE: tuple[int, int] = (8, 8)
BILATERAL_DIAMETRO: int = 7
BILATERAL_SIGMA_COR: float = 35.0
BILATERAL_SIGMA_ESPACO: float = 35.0
AREA_MINIMA_COMPONENTE: int = 30
FRACAO_MAXIMA_COMPONENTE: float = 0.15
COBERTURA_MAXIMA_METODO_VOTO: float = 0.25
ESCALA_FRANGI: float = 0.5
COMPRIMENTO_MINIMO_REGIAO_VOTO: int = 24
FRACAO_COMPRIMENTO_MINIMO_VOTO: float = 0.078

# Constantes para alertas
PERSISTENCIA_INSPECAO: int = 3
CONFIANCA_INSPECAO: float = 0.45
VOTOS_MINIMOS_URGENTE: int = 3

# Faixas de tamanho VISUAL na imagem. Não são limites geotécnicos nem servem
# para diagnosticar a estabilidade da barragem; apenas definem prioridade de
# inspeção para a maior região encontrada pela câmera.
AREA_RELATIVA_MEDIA: float = 0.0015
AREA_RELATIVA_GRANDE: float = 0.008
COMPRIMENTO_RELATIVO_MEDIO: float = 0.10
COMPRIMENTO_RELATIVO_GRANDE: float = 0.30

# Constantes mágicas da segmentação
KERNEL_MORFOLOGIA_GRANDE: int = 15
ADAPTATIVO_BLOCO: int = 51
ADAPTATIVO_C: int = 9
PERCENTIL_BLACKHAT: float = 88.0
PERCENTIL_TOPHAT: float = 88.0
PERCENTIL_SOBEL: float = 85.0
PERCENTIL_FRANGI: float = 92.0
FRANGI_LIMIAR_RESOLUCAO: int = 360
BINS_MODELO_SIMPLES: int = 32


def _zip_padrao() -> Path:
    """Retorna o caminho padrão do arquivo ZIP contendo os dados."""
    env = os.environ.get("DAMCRACK_ZIP")
    if env:
        return Path(env)
    candidatos = [
        Path("/content/drive/MyDrive/Damage Segmentaion.zip"),
        Path("/content/drive/MyDrive/Documentos/Damage Segmentaion.zip"),
        Path("/content/Damage Segmentaion.zip"),
        Path.home() / "Documentos/Damage Segmentaion.zip",
        RAIZ_PROJETO / "Damage Segmentaion.zip",
    ]
    for caminho in candidatos:
        if caminho.is_file():
            return caminho
    return candidatos[0]


ZIP_PADRAO: Path = _zip_padrao()
PASTA_AMOSTRA: Path = Path(os.environ.get("DAMCRACK_AMOSTRA", str(RAIZ_PROJETO / "dados" / "amostra")))
PASTA_RESULTADOS: Path = RAIZ_PROJETO / "resultados"
PASTA_MODELOS: Path = PASTA_RESULTADOS / "modelos"
MODELO_SIMPLES_PADRAO: Path = PASTA_MODELOS / "limiar_damcrack.npz"


def definir_zip(caminho: str | Path) -> Path:
    """Define globalmente o caminho para o dataset. Use no Colab depois de montar o Drive.
    
    Args:
        caminho: Caminho para o arquivo zip do dataset.
        
    Returns:
        Objeto Path representando o caminho definido.
    """
    global ZIP_PADRAO
    ZIP_PADRAO = Path(caminho)
    os.environ["DAMCRACK_ZIP"] = str(ZIP_PADRAO)
    logger.info("Caminho do ZIP definido para: %s", ZIP_PADRAO)
    return ZIP_PADRAO
