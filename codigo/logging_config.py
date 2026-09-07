"""Configuração centralizada de logging para o projeto barragem.

Uso típico em qualquer módulo::

    import logging
    logger = logging.getLogger(__name__)

Para ativar o logging no script principal::

    from codigo.logging_config import configurar_logging
    configurar_logging()          # INFO no console
    configurar_logging("DEBUG")   # DEBUG no console
"""

from __future__ import annotations

import logging
import sys


_CONFIGURADO = False

FORMATO_PADRAO = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
FORMATO_DATA = "%H:%M:%S"


def configurar_logging(
    nivel: str = "INFO",
    formato: str = FORMATO_PADRAO,
    formato_data: str = FORMATO_DATA,
) -> None:
    """Configura o logging raiz do projeto.

    Deve ser chamada uma única vez, no início do script principal.
    Chamadas subsequentes são ignoradas para evitar handlers duplicados.

    Args:
        nivel: Nível mínimo de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        formato: String de formatação das mensagens.
        formato_data: Formato do timestamp nas mensagens.
    """
    global _CONFIGURADO
    if _CONFIGURADO:
        return
    _CONFIGURADO = True

    nivel_numerico = getattr(logging, nivel.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(formato, datefmt=formato_data))

    raiz = logging.getLogger()
    raiz.setLevel(nivel_numerico)
    raiz.addHandler(handler)

    # Reduz verbosidade de bibliotecas externas
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
