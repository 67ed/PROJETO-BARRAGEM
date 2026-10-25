"""Texto seguro para HUD do OpenCV (Hershey não desenha acentos)."""

from __future__ import annotations

_MAPA_ACENTOS = str.maketrans(
    "áàâãäéèêëíìîïóòôõöúùûüçñÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇÑ",
    "aaaaaeeeeiiiiooooouuuucnAAAAAEEEEIIIIOOOOOUUUUCN",
)


def para_hud(texto: object) -> str:
    """Remove acentos e caracteres que o OpenCV não renderiza."""
    return str(texto).translate(_MAPA_ACENTOS)
