from codigo.alerts import classificar, confianca_simples


def test_sem_regiao_e_nenhum() -> None:
    assert classificar(False, 0.9, frames_persistentes=10) == "nenhum"


def test_confirmado_grande_e_prioritario() -> None:
    nivel = classificar(True, 0.4, confirmado_por_metodos=True, tamanho_visual="grande")
    assert nivel == "prioritario"


def test_confirmado_media_e_inspecao() -> None:
    nivel = classificar(True, 0.2, confirmado_por_metodos=True, tamanho_visual="media")
    assert nivel == "inspecao"


def test_confianca_limitada() -> None:
    assert 0.0 <= confianca_simples(100.0, None) <= 1.0
    assert 0.0 <= confianca_simples(10.0, 0.8) <= 1.0
