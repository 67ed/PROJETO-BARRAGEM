from codigo.detector import detectar
from codigo.sintetico import gerar_frame
from codigo.texto import para_hud
from codigo.ui import desenhar_painel


def test_para_hud_remove_acentos() -> None:
    assert para_hud("avaliação") == "avaliacao"
    assert para_hud("Nível") == "Nivel"


def test_frame_sintetico_com_rachadura() -> None:
    imagem, mascara = gerar_frame(seed=3, com_rachadura=True)
    assert imagem.ndim == 3
    assert mascara.max() == 255


def test_canny_encontra_traco_sintetico() -> None:
    imagem, _mascara = gerar_frame(seed=11, com_rachadura=True)
    resultado = detectar(imagem, metodo="canny")
    assert resultado.mascara.shape[:2] == imagem.shape[:2]
    assert resultado.tempo_s > 0


def test_hud_preserva_tamanho() -> None:
    imagem, _ = gerar_frame(seed=1, com_rachadura=True)
    resultado = detectar(imagem, metodo="canny")
    dados = {
        "alert_level": "inspecao",
        "alert_label": "necessita inspeção",
        "camera_id": "CAM-SYN",
        "timestamp": "2026-01-01 00:00:00-0300",
        "metodo": "canny",
        "n_regioes": len(resultado.regioes),
        "confidence": 0.5,
        "explicacao": "Possível alteração visual no concreto.",
        "metodos_votacao": ["canny", "tophat", "sobel", "frangi", "treinado"],
        "maior_votacao": 3,
        "tempo_s": resultado.tempo_s,
    }
    tela = desenhar_painel(resultado.overlay, dados, fps=2.0, ajuda=True)
    assert tela.shape == resultado.overlay.shape
