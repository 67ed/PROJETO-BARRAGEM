# DamGuard

Detector de rachaduras em barragens de concreto, feito com visão computacional em Python.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-protótipo-orange)
![Testes](https://img.shields.io/badge/testes-pytest-0A9EDC?logo=pytest&logoColor=white)
![Licença](https://img.shields.io/badge/uso-acadêmico-lightgrey)

> **Aviso:** o DamGuard é um protótipo que ajuda no monitoramento visual. Ele não diagnostica a estrutura nem estima risco de ruptura. Os níveis de alerta são internos do software e não substituem a inspeção de profissionais habilitados.

## Sobre o projeto

O DamGuard analisa imagens, vídeos ou a webcam procurando possíveis rachaduras em superfícies de concreto. Em vez de depender de um único detector, ele usa cinco métodos que votam na mesma região da imagem. O alerta mais urgente só aparece quando pelo menos 3 métodos concordam, o que reduz os falsos alarmes.

O programa funciona com webcam, vídeo ou imagem. Também tem um modo demo, que usa o dataset DamCrack ou gera concreto sintético, então dá para testar sem câmera.

## Demonstração

Capturas do sistema rodando no modo demo. Em cada uma aparecem o nível de alerta (canto superior direito), as regiões detectadas (R1, R2...), os votos dos métodos (as bolinhas) e a explicação do bot na parte de baixo.

**Monitoramento prioritário**

<table>
  <tr>
    <td align="center"><img src="docs/img/urgente-fissura-linear.png" width="300"><br><sub>Fissura longa em concreto escuro, 5/5 votos</sub></td>
    <td align="center"><img src="docs/img/urgente-muro.png" width="300"><br><sub>Fissura em muro de pedra, 4/5 votos</sub></td>
    <td align="center"><img src="docs/img/urgente-fissura-vertical.png" width="300"><br><sub>Fissura vertical com sombra, 5/5 votos</sub></td>
  </tr>
</table>

**Necessita inspeção**

<table>
  <tr>
    <td align="center"><img src="docs/img/inspecao-encosta.png" width="300"><br><sub>Superfície rochosa, 5/5 votos</sub></td>
    <td align="center"><img src="docs/img/inspecao-concreto-claro.png" width="300"><br><sub>Concreto com marcas claras, 5/5 votos</sub></td>
    <td align="center"><img src="docs/img/inspecao-concreto-escuro.png" width="300"><br><sub>Concreto em escala de cinza, 4/5 votos</sub></td>
  </tr>
</table>

**Um método sozinho pode falhar**

Na mesma imagem sintética, o Canny sozinho não destacou nada, enquanto a votação encontrou a rachadura.

<table>
  <tr>
    <td align="center"><img src="docs/img/sintetico-canny.png" width="400"><br><sub>Só Canny: nenhuma alteração relevante</sub></td>
    <td align="center"><img src="docs/img/sintetico-votacao.png" width="400"><br><sub>Votação: rachadura detectada, 4/5 votos</sub></td>
  </tr>
</table>

## Como rodar

```bash
git clone https://github.com/67ed/PROJETO-BARRAGEM.git
cd PROJETO-BARRAGEM

python -m venv .venv
source .venv/bin/activate.fish   # fish shell
# source .venv/bin/activate      # bash / zsh
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
python app.py
```

Sem argumentos, o programa usa a webcam. Se ela não for encontrada, cai para o modo demo.

### Opções

| Comando | O que faz |
|---|---|
| `python app.py --demo` | Roda a demonstração (DamCrack ou concreto sintético) |
| `python app.py --video clipe.mp4` | Analisa um vídeo |
| `python app.py --imagem foto.jpg` | Analisa uma imagem |
| `python app.py --metodo votacao` | Escolhe o método de detecção |

Para usar o dataset DamCrack no modo demo, aponte o caminho do ZIP (opcional):

```bash
export DAMCRACK_ZIP="/caminho/para/Damage Segmentaion.zip"
```

No fish, use `set -x DAMCRACK_ZIP "/caminho/para/Damage Segmentaion.zip"`.

## Como funciona

Cinco métodos analisam a imagem e votam na mesma região:

| Método | O que detecta |
|---|---|
| Canny | Bordas |
| Top-Hat | Traços claros |
| Sobel | Gradiente |
| Frangi | Estruturas alongadas |
| Modelo treinado | Escurecimento e borda |

Antes da votação, a imagem passa por filtro bilateral, CLAHE (contraste local) e remoção de pontos isolados. Se um método marcar mais de 25% da imagem, ele é descartado.

## Níveis de alerta

| Nível | Significado |
|---|---|
| Nenhum | Sem anomalia suficiente |
| Baixa atenção | Pouca evidência |
| Necessita inspeção | Métodos concordam ou a região persiste |
| Monitoramento prioritário | 3 ou mais métodos concordando e região extensa |

## Controles

| Tecla / clique | Ação |
|---|---|
| `q` ou `Esc` | Sair |
| `espaço` | Pausar / retomar |
| `s` | Salvar frame com HUD |
| `m` | Trocar método |
| `h` | Mostrar / ocultar atalhos |
| `n` | Próxima imagem (modo demo) |
| PAUSAR / SAIR | Botões na barra inferior |

## Scripts auxiliares

```bash
python avaliar_dataset.py --n 50                         # IoU em lote no DamCrack
python comparar_metodos.py --imagem foto.jpg --mostrar   # grade visual dos detectores
python treinar_modelo_simples.py                         # treina o modelo simples
pytest                                                   # roda os testes
```

## Estrutura do projeto

```
PROJETO-BARRAGEM/
├── app.py                      # interface ao vivo
├── avaliar_dataset.py          # IoU em lote no DamCrack
├── comparar_metodos.py         # grade visual dos detectores
├── treinar_modelo_simples.py   # treino do modelo simples
├── codigo/                     # pipeline, votação, HUD, bot
├── dados/                      # dados do projeto
├── resultados/                 # overlays, máscaras, logs JSON
├── tests/                      # testes de alerta, HUD e sintético
├── requirements.txt
└── pyproject.toml
```

## Dataset DamCrack

O dataset tem cerca de 4.000 pares de imagem e máscara. Nas máscaras, vermelho marca possível rachadura e azul marca outro tipo de dano. O ZIP original se chama `Damage Segmentaion.zip`, com esse erro de digitação mesmo, vindo do próprio dataset.

## Licença

Projeto acadêmico. Consulte os autores para uso comercial.
