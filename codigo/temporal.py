"""Persistência entre frames. Não prova que a alteração é uma rachadura."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from codigo.analysis import Regiao

logger = logging.getLogger(__name__)


def _iou_caixas(a: Regiao | None, b: Regiao | None) -> float:
    """Calcula a Intersection over Union (IoU) entre os retângulos delimitadores (Bounding Boxes).
    
    Args:
        a: Primeira região.
        b: Segunda região.
        
    Returns:
        Valor de IoU, entre 0.0 e 1.0. Retorna 0.0 se não houver intersecção.
    """
    if a is None or b is None:
        return 0.0
        
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.largura, b.x + b.largura)
    y2 = min(a.y + a.altura, b.y + b.altura)
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    uniao = a.largura * a.altura + b.largura * b.altura - inter
    
    if uniao <= 0:
        return 0.0
    return inter / uniao


@dataclass
class RastreadorPersistencia:
    """Acompanha se as alterações visuais ocorrem em frames consecutivos na mesma área.
    
    Attributes:
        iou_minimo: Valor mínimo de sobreposição (IoU) para considerar a mesma detecção.
        contagem: Total de frames seguidos onde as regiões se mantiveram.
        ultima: Lista das regiões salvas do frame anterior.
    """
    iou_minimo: float = 0.2
    contagem: int = 0
    ultima: list[Regiao] = field(default_factory=list)

    def atualizar(self, regioes: list[Regiao] | None) -> int:
        """Atualiza a contagem baseada na sobreposição das regiões atuais com as anteriores.
        
        Args:
            regioes: Lista de regiões detectadas no frame atual.
            
        Returns:
            O número de frames seguidos em que uma região equivalente foi detectada.
        """
        if not regioes:
            self.contagem = 0
            self.ultima = []
            return 0
            
        if not self.ultima:
            self.contagem = 1
            self.ultima = regioes
            return self.contagem
            
        combinou = any(
            _iou_caixas(nova, antiga) >= self.iou_minimo
            for nova in regioes
            for antiga in self.ultima
        )
        self.contagem = self.contagem + 1 if combinou else 1
        self.ultima = regioes
        return self.contagem
