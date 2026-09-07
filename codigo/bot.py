"""Bot: transforma o resultado estruturado em texto simples. Sem diagnóstico."""

from __future__ import annotations

import logging
from codigo.alerts import NIVEIS, NivelAlerta

logger = logging.getLogger(__name__)

FRASES_NIVEL: dict[str, str] = {
    NivelAlerta.NENHUM.value: (
        "Não foi destacada uma região com evidência suficiente nesta imagem. "
        "Isso não significa que a barragem esteja segura."
    ),
    NivelAlerta.BAIXA_ATENCAO.value: (
        "Foi identificada uma possível alteração visual, mas a evidência atual é baixa. "
        "Pode ser sombra, sujeira, textura do concreto ou outro artefato."
    ),
    NivelAlerta.INSPECAO.value: (
        "Foi identificada uma possível alteração persistente. "
        "Recomenda-se inspeção pela equipe responsável."
    ),
    NivelAlerta.PRIORITARIO.value: (
        "Ao menos três métodos visuais concordaram sobre uma alteração visual extensa na mesma região. "
        "Recomenda-se avaliação prioritária pela equipe responsável."
    ),
}


def explicar(resultado: dict | None) -> str:
    """Interpreta os resultados do detector e gera um resumo em texto humano.
    
    Não emite laudo estrutural, apenas sumariza os achados da visão computacional.
    
    Args:
        resultado: Dicionário contendo os dados da análise.
                   
    Returns:
        Um texto descrevendo a situação de forma legível.
    """
    if not resultado:
        logger.warning("Resultado vazio fornecido.")
        return FRASES_NIVEL[NivelAlerta.NENHUM.value]
        
    nivel = resultado.get("alert_level", NivelAlerta.BAIXA_ATENCAO.value)
    partes = [FRASES_NIVEL.get(nivel, FRASES_NIVEL[NivelAlerta.BAIXA_ATENCAO.value])]

    camera = resultado.get("camera_id")
    if camera:
        partes.append(f"Câmera: {camera}.")

    frames = resultado.get("persistent_frames")
    if frames and isinstance(frames, int) and frames > 1:
        partes.append(f"A alteração foi associada em {frames} frame(s) consecutivos.")

    metodos = resultado.get("metodos_votacao") or []
    if resultado.get("confirmado_por_metodos") and metodos:
        maior_voto = resultado.get('maior_votacao', 0)
        votos_min = resultado.get('votos_minimos', 0)
        partes.append(
            f"Votação máxima: {maior_voto} de {len(metodos)} métodos "
            f"(mínimo exigido: {votos_min})."
        )

    descartados = resultado.get("metodos_descartados") or []
    if descartados and isinstance(descartados, list):
        partes.append(
            "Foram desconsiderados nesta votação: "
            f"{', '.join(str(d) for d in descartados)}; eles marcaram uma área excessiva da imagem."
        )

    regiao = resultado.get("region")
    if isinstance(regiao, dict) and "x" in regiao:
        partes.append(
            f"Região aproximada: x={regiao['x']}, y={regiao['y']}, "
            f"{regiao.get('width', '?')}×{regiao.get('height', '?')} px."
        )

    tamanho = resultado.get("visual_size")
    if isinstance(tamanho, dict) and tamanho.get("codigo") != "nenhuma":
        rotulo = tamanho.get('rotulo', '-')
        comp_px = tamanho.get('comprimento_px', 0)
        comp_perc = tamanho.get('comprimento_percentual', 0)
        area_perc = tamanho.get('area_percentual', 0)
        partes.append(
            f"Tamanho visual: {rotulo}; "
            f"comprimento aproximado de {comp_px} px "
            f"({comp_perc}% da imagem) e "
            f"área de {area_perc}%."
        )

    partes.append(
        "O sistema não afirma que exista uma rachadura estrutural "
        "nem estima risco de ruptura. O tamanho visual na foto não substitui "
        "inspeção técnica, instrumentação ou análise de engenharia."
    )
    return " ".join(partes)


def rotulo_nivel(codigo: str | None) -> str:
    """Traduz o código do nível de alerta para descrição.
    
    Args:
        codigo: Chave do nível.
        
    Returns:
        String de texto descritivo.
    """
    if codigo is None:
        return "desconhecido"
    return NIVEIS.get(codigo, codigo)
