# Projeto Barragem — detector de rachaduras

Protótipo de visão computacional para **apoiar o monitoramento** de barragens de concreto.  
Não diagnostica estrutura e **não** estima risco de ruptura.

A câmera observa, o sistema analisa, o alerta informa e o bot explica.

## Início rápido

```bash
cd projeto_barragem
python -m venv .venv
source .venv/bin/activate      # se usar bash / zsh
# source .venv/bin/activate.fish # se usar fish shell
pip install -r requirements.txt
python app.py --demo
```

`--demo` usa o DamCrack se o ZIP estiver disponível. Sem o dataset, gera **concreto sintético** na hora — dá para apresentar o sistema sem câmera e sem download.

```bash
python app.py                  # webcam (cai para demo se falhar)
python app.py --video clipe.mp4
python app.py --imagem foto.jpg
python app.py --metodo votacao
```

ZIP do DamCrack (opcional):

```bash
export DAMCRACK_ZIP=/caminho/para/Damage\ Segmentaion.zip
```

## Como funciona

Cinco métodos clássicos votam na **mesma região**. O aviso urgente só sai com pelo menos 3 concordâncias.

| Método | Papel |
|--------|--------|
| Canny | Bordas |
| Top-Hat | Traços claros |
| Sobel | Gradiente |
| Frangi | Estruturas alongadas |
| Modelo treinado | Escurecimento + borda |

Antes da votação: filtro bilateral, CLAHE, remoção de pontos isolados e descarte de método com cobertura > 25%.

## Controles

| Tecla / clique | Ação |
|----------------|------|
| `q` ou `Esc` | Sair |
| `espaço` | Pausar / retomar |
| `s` | Salvar frame com HUD |
| `m` | Trocar método |
| `h` | Mostrar / ocultar atalhos |
| `n` | Próxima imagem (demo) |
| **PAUSAR** / **SAIR** | Clique na barra inferior |

## Níveis de alerta (internos do software)

| Nível | Significado |
|-------|-------------|
| nenhum | Sem anomalia suficiente |
| baixa atenção | Pouca evidência |
| necessita inspeção | Concordância ou região persistente |
| monitoramento prioritário | ≥3 métodos + região extensa |

Esses níveis **não** são classificação oficial de segurança.

## Scripts

```bash
python avaliar_dataset.py --n 50
python comparar_metodos.py --imagem foto.jpg --mostrar
python treinar_modelo_simples.py
pytest
```

## Estrutura

```
app.py                 interface ao vivo
avaliar_dataset.py     IoU em lote no DamCrack
comparar_metodos.py    grade visual dos detectores
codigo/                pipeline, votação, HUD, bot
tests/                 testes de alerta, HUD e sintético
resultados/            overlays, máscaras, logs JSON
```

## Dataset DamCrack

~4.000 pares imagem/máscara. Vermelho = possível rachadura; azul = outro dano.  
O ZIP original se chama `Damage Segmentaion.zip` (typo do dataset).

## Licença

Projeto acadêmico. Consulte os autores para uso comercial.
