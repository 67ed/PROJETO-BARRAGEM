# Projeto Barragem — possíveis rachaduras

Protótipo de visão computacional para **apoiar o monitoramento** de uma barragem de concreto.  
O sistema **não** diagnostica estrutura e **não** estima risco de ruptura.

Frase-guia: *a câmera observa, o sistema analisa, o alerta informa e o bot explica.*

## Testar no VS Code (sistema completo)

No terminal do projeto, com o venv:

```bash
cd ~/projeto_barragem
source .venv/bin/activate
python app.py --demo
```

`--demo` usa fotos do DamCrack (não precisa de webcam).  
Webcam: `python app.py`  
Vídeo: `python app.py --video arquivo.mp4`

Ou no VS Code: Run and Debug → **Barragem: demo DamCrack**.

Teclas na janela: `q` sair, `s` salvar, espaço pausar.

Além do detector clássico, o projeto tem um **modelo simples treinado por tabela**. Ele
aprende, pelas máscaras do DamCrack, quais combinações de pixel escuro e borda visual
devem ser destacadas como possível rachadura. Câmera, persistência, alerta e bot estão
no `app.py`.

1. Lê o DamCrack (`Damage Segmentaion.zip`) **sem** descompactar as 4000 imagens de uma vez.
2. Emparelha `A123.jpg` com a máscara refinada `A123.png`.
3. Pré-processa (redimensiona, cinza, blur).
4. Roda um detector clássico (Canny / Otsu / os dois).
5. Destaca regiões, grava overlay e gera um texto de alerta.

O detector clássico é a linha de base. O modelo treinado é leve e não usa rede neural.

## Treinar o modelo leve com o DamCrack

O ZIP informado contém 4.000 pares de imagens e máscaras. O treino processa uma imagem
por vez, conserva apenas duas tabelas 32 × 32 na memória e não descompacta o conjunto de dados.

```bash
cd ~/projeto_barragem
source .venv/bin/activate

# treino leve com 300 imagens (padrão)
python treinar_modelo_simples.py

# opcional: usar as 4.000 imagens, ainda uma por vez
python treinar_modelo_simples.py --amostras 0
```

O modelo será salvo em `resultados/modelos/limiar_damcrack.npz`. Para usá-lo na câmera,
vídeo ou demo:

```bash
python app.py --metodo treinado
python app.py --demo --metodo treinado
python run_semana2.py --n 20 --metodo treinado
```

O modelo treinado continua sendo uma ferramenta de apoio à inspeção visual: a máscara
prevista não confirma uma rachadura estrutural nem estima risco de ruptura.

## Votação entre métodos e aviso urgente

O `app.py` usa `--metodo votacao` por padrão. O aviso urgente só aparece quando **pelo
menos 3 dos 5 métodos** concordam na mesma região: Canny, top-hat (traços claros), Sobel,
Frangi e o modelo simples treinado. Um ou dois métodos isolados podem gerar `baixa atenção`
ou `necessita inspeção`, mas não um aviso urgente.

Antes da votação, toda foto passa por redução de ruído bilateral, compensação local de
iluminação (CLAHE), remoção de pontos isolados e remoção de blocos grandes de textura ou
sombra. Um método que marque mais de 25% da imagem também é retirado da votação daquela
foto; isso evita que textura do concreto transforme sozinha a análise em alerta urgente.
Por fim, a votação só conserva regiões com continuidade espacial mínima, reduzindo pontos
soltos que recebem votos por coincidência.

Otsu, limiar adaptativo, black-hat, contraste local e a combinação Canny+Otsu continuam
disponíveis para testes individuais. Eles não entram nos 3 de 5 porque são mais sensíveis
à iluminação e à textura. Se o arquivo do modelo treinado não existir, ele é substituído
automaticamente por `contraste_local`, e isso fica registrado no log.

Para comparar visualmente todos os métodos em uma foto, use:

```bash
python comparar_metodos.py --imagem caminho/para/foto.jpg --mostrar

# Sem --imagem, usa a primeira imagem em dados/amostra/
python comparar_metodos.py --mostrar
```

A grade também é salva em `resultados/comparacoes/`.

## Prioridade visual pelo tamanho da região

Além da votação, o sistema mede a maior região detectada na foto: área percentual e
comprimento percentual da imagem. Ela é mostrada como **pequena**, **média** ou **extensa**.
Uma região extensa confirmada por pelo menos 3 dos 5 métodos recebe `aviso urgente —
inspeção prioritária`; confirmação de 3/5 com região pequena ou média recebe `necessita
inspeção`.

Essas faixas estão em `codigo/config.py` e são apenas critérios visuais internos. Tamanho
em pixels não informa profundidade, abertura real, causa, evolução, estabilidade ou risco
de ruptura. A prioridade real exige inspeção de profissional habilitado e os procedimentos
aplicáveis à barragem.

Por padrão, uma nova imagem é analisada a cada 2 segundos. Para deixar ainda mais lento:

```bash
python app.py --intervalo 5
```

As fotos salvas pela tecla `s` continuam manuais. O intervalo controla a frequência de
captura analisada e de novos registros no log.

## Google Colab (recomendado para testar)

1. Envie `Damage Segmentaion.zip` para o **Google Drive** (não precisa descompactar).
2. Abra no Colab o arquivo `notebooks/semana_2_colab.ipynb` (Arquivo → Upload do notebook).
3. Rode **Executar tudo**. Autorize o Drive quando pedir.
4. Se não achar o zip, edite `CAMINHO_ZIP` (ex.: `/content/drive/MyDrive/Documentos/Damage Segmentaion.zip`).

O notebook do Colab já carrega o código; não precisa enviar a pasta `codigo/` junto.

## Como rodar no computador

```bash
cd ~/projeto_barragem
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_semana2.py --n 8
```

Saídas:

- `resultados/imagens/` — imagem com a região destacada
- `resultados/mascaras/` — máscara prevista
- `resultados/registros/semana2.json` — nível de alerta, IoU, tempo, texto do bot

Notebook de experimentos:

```bash
jupyter notebook notebooks/semana_2.ipynb
```

O zip padrão está em `/home/gustavo/Documentos/Damage Segmentaion.zip`.  
Se estiver em outro lugar: `python run_semana2.py --zip /caminho/arquivo.zip`.

## Pastas

```
projeto_barragem/
├── dados/
│   └── amostra/        # criado na hora (poucas imagens)
├── notebooks/semana_2.ipynb
├── codigo/
│   ├── data.py             # zip, pares imagem+máscara
│   ├── preprocessing.py
│   ├── segmentation.py     # Canny / Otsu
│   ├── detector.py
│   ├── analysis.py         # área, IoU, overlay
│   ├── temporal.py         # persistência (para a câmera depois)
│   ├── alerts.py
│   ├── bot.py
│   └── pipeline.py
├── resultados/
│   ├── imagens/
│   ├── mascaras/
│   ├── modelos/
│   └── registros/
└── run_semana2.py
```

## Máscaras do DamCrack

Nas máscaras refinadas (RGB):

| Cor | Classe no XML | Tratamento neste código |
|-----|----------------|-------------------------|
| vermelho `(255,0,0)` | `0` | possível rachadura |
| azul `(0,0,255)` | `1` | outro dano (em geral spalling) |

Confira no artigo do conjunto de dados se essa associação bate 100%. O código **não** trata todo dano como rachadura.

## Linguagem dos alertas

Os níveis (`baixa atenção`, `necessita inspeção`, `monitoramento prioritário`) são **internos do software**, não níveis oficiais de segurança da barragem.

## Hardware

Processamento **uma imagem por vez**. Não carregue o zip inteiro no notebook. Comece com `--n 8` ou `--n 20`.
# PROJETO-BARRAGEM
