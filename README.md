# 🏗️ Projeto Barragem — Detecção de Rachaduras

Protótipo de visão computacional para **apoiar o monitoramento** de barragens de concreto.  
O sistema **não** diagnostica estrutura e **não** estima risco de ruptura.

> *A câmera observa, o sistema analisa, o alerta informa e o bot explica.*

---

## ⚡ Início Rápido

```bash
git clone https://github.com/SEU_USUARIO/projeto_barragem.git
cd projeto_barragem
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Rodar com imagens de teste (sem câmera)

```bash
python app.py --demo
```

### Rodar com webcam

```bash
python app.py
```

### Rodar com vídeo

```bash
python app.py --video caminho/para/video.mp4
```

> **Nota:** O modo `--demo` precisa do arquivo `Damage Segmentaion.zip` do dataset DamCrack.
> Configure o caminho via variável de ambiente: `export DAMCRACK_ZIP=/caminho/para/arquivo.zip`

---

## 🧠 Como Funciona

O sistema usa **5 métodos clássicos** de segmentação (sem rede neural):

| Método | Tipo |
|--------|------|
| **Canny** | Detecção de bordas |
| **Top-Hat** | Realce de traços claros |
| **Sobel** | Gradiente de intensidade |
| **Frangi** | Filtro para estruturas tubulares |
| **Modelo Treinado** | Tabela de escurecimento + borda |

### Sistema de Votação

O alerta urgente só aparece quando **pelo menos 3 dos 5 métodos** concordam na **mesma região** da imagem.

Antes da votação, cada imagem passa por:
- ✅ Redução de ruído bilateral
- ✅ Compensação de iluminação (CLAHE)
- ✅ Remoção de pontos isolados
- ✅ Descarte de métodos com cobertura > 25% (evita falsos positivos por textura)

---

## 🔧 Scripts Disponíveis

| Script | Descrição |
|--------|-----------|
| `app.py` | Aplicação principal com interface gráfica |
| `avaliar_dataset.py` | Avaliação em lote no DamCrack com cálculo de IoU |
| `comparar_metodos.py` | Grade visual comparando todos os métodos |
| `treinar_modelo_simples.py` | Treina o modelo tabular leve |

### Avaliar no Dataset

```bash
# Avaliar 50 imagens (padrão) com votação
python avaliar_dataset.py

# Avaliar com método específico
python avaliar_dataset.py --n 20 --metodo canny
```

### Treinar o Modelo Leve

```bash
# Treino com 300 imagens (padrão, leve)
python treinar_modelo_simples.py

# Usar todas as 4.000 imagens
python treinar_modelo_simples.py --amostras 0
```

O modelo é salvo em `resultados/modelos/limiar_damcrack.npz`.

### Comparar Métodos Visualmente

```bash
python comparar_metodos.py --imagem caminho/para/foto.jpg --mostrar
```

---

## 📁 Estrutura do Projeto

```
projeto_barragem/
├── app.py                      # Interface gráfica (webcam/vídeo/demo)
├── avaliar_dataset.py          # Avaliação em lote no DamCrack
├── comparar_metodos.py         # Comparação visual dos métodos
├── treinar_modelo_simples.py   # Treino do modelo tabular
├── requirements.txt
│
├── codigo/                     # Pacote principal
│   ├── __init__.py
│   ├── config.py               # Parâmetros e constantes
│   ├── logging_config.py       # Configuração centralizada de log
│   ├── preprocessing.py        # Pré-processamento de imagem
│   ├── segmentation.py         # Algoritmos de segmentação
│   ├── detector.py             # Orquestrador de detecção + votação
│   ├── analysis.py             # Análise geométrica das regiões
│   ├── alerts.py               # Classificação de nível de alerta
│   ├── bot.py                  # Geração de texto explicativo
│   ├── pipeline.py             # Pipeline imagem → resultado
│   ├── sistema.py              # Fachada do sistema completo
│   ├── camera.py               # Abstração de fontes de vídeo
│   ├── data.py                 # Leitura do DamCrack (ZIP)
│   ├── temporal.py             # Persistência entre frames
│   ├── modelo_simples.py       # Classificador tabular leve
│   ├── treino_simples.py       # Lógica de treino do modelo
│   └── ui.py                   # HUD / overlay na tela
│
├── dados/                      # Amostras extraídas (gerado em runtime)
├── resultados/
│   ├── imagens/                # Overlays gerados
│   ├── mascaras/               # Máscaras preditas
│   ├── comparacoes/            # Grades comparativas
│   ├── modelos/                # Modelo treinado (.npz)
│   └── registros/              # Logs JSON de cada execução
└── notebooks/                  # Notebooks Jupyter (opcional)
```

---

## 🎯 Controles da Interface

| Controle | Ação |
|----------|------|
| `q` ou `Esc` | Sair |
| `s` | Salvar frame atual |
| `Espaço` | Pausar/retomar captura |
| Clique em **PARAR CAPTURA** | Pausar/retomar |
| Clique em **SAIR** | Fechar aplicação |

---

## 📊 Níveis de Alerta

| Nível | Significado |
|-------|-------------|
| `nenhum` | Sem anomalia detectada |
| `baixa atenção` | Possível anomalia, poucos métodos detectaram |
| `necessita inspeção` | Anomalia confirmada por múltiplos métodos |
| `monitoramento prioritário` | ≥3 métodos concordam + região extensa |

> ⚠️ Esses níveis são **internos do software** e não representam classificação oficial de segurança.

---

## 📋 Dataset DamCrack

O projeto usa o dataset **DamCrack** (Damage Segmentation) com ~4.000 pares imagem/máscara.

| Cor da Máscara | Classe | Tratamento |
|----------------|--------|------------|
| Vermelho `(255,0,0)` | `0` | Possível rachadura |
| Azul `(0,0,255)` | `1` | Outro dano (spalling) |

> O nome do ZIP é `Damage Segmentaion.zip` (com typo original do dataset).

---

## 🛠️ Requisitos

- Python 3.10+
- OpenCV >= 4.8
- NumPy >= 1.24
- scikit-image >= 0.22
- Matplotlib >= 3.7

Instale tudo com:

```bash
pip install -r requirements.txt
```

---

## 📄 Licença

Projeto acadêmico. Consulte os autores para uso comercial.
