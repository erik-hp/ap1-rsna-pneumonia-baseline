# TP1 — RSNA 2018 Pneumonia Detection Baseline

Baseline clássico para o **RSNA 2018 Pneumonia Detection Challenge**, desenvolvido pela **Equipe E1** na disciplina **Tópicos Especiais em Sistemas de Informação**, do curso de Sistemas de Informação.

O objetivo do trabalho é construir, avaliar e documentar um piso de desempenho reproduzível para análise de radiografias de tórax usando exclusivamente **visão computacional clássica, características hand-crafted e modelos tradicionais de aprendizado de máquina**, sem redes neurais profundas ou extratores profundos pré-treinados.

**Repositório:** https://github.com/erik-hp/ap1-rsna-pneumonia-baseline

## Equipe e divisão de trabalho

| Integrante | Frente principal |
|---|---|
| Erik Holanda Pires | Modelagem, protocolo experimental, avaliação, detecção e integração final |
| Ruan Pablo Saraiva Ripardo | Dados, pré-processamento e extração de características |
| Natanael Douglas Alves Feijão | EDA, revisão de literatura, análise de erro e redação científica |

A tabela oficial de contribuição individual deve ser entregue separadamente, assinada pelos integrantes, seguindo o Anexo A do enunciado. Cada atividade da matriz de responsabilidade deve ter exatamente um responsável principal `R`.

## Formulação do problema

O desafio original envolve **detecção/localização de opacidades compatíveis com pneumonia em radiografias de tórax** por meio de bounding boxes.

Neste baseline, o problema foi tratado em duas etapas complementares:

1. **Classificação por radiografia:** entrada = uma radiografia; saída = `Target` binário (0/1).
2. **Detecção/localização:** entrada = regiões candidatas da radiografia; saída = escore de pneumonia e bounding boxes após pós-processamento.

A classificação global não substitui a avaliação de localização; por isso o projeto também calcula métricas de detecção.

## Restrições metodológicas do trabalho

Este projeto respeita as restrições centrais do TP1:

- não utiliza redes neurais profundas;
- não utiliza ResNet, VGG, CLIP, DINO ou outros extratores profundos pré-treinados;
- utiliza apenas descritores explícitos e modelos clássicos;
- inclui baseline trivial;
- compara pelo menos três famílias de descritores;
- compara pelo menos três modelos clássicos;
- separa treino e avaliação por paciente;
- ajusta hiperparâmetros apenas dentro do treino;
- usa seed fixa;
- reporta média e desvio-padrão entre folds;
- inclui métricas adequadas para classificação desbalanceada e localização.

## Dados

### Origem

Foi utilizado o conjunto de treino do **RSNA Pneumonia Detection Challenge 2018**, hospedado no Kaggle.

Competição Kaggle:

```text
rsna-pneumonia-detection-challenge
```

Os dados brutos não são versionados neste repositório.

O pipeline espera os arquivos:

```text
data/raw/
├── stage_2_train_labels.csv
└── stage_2_train_images/
    ├── <patientId>.dcm
    └── ...
```

O conjunto de teste oficial do Kaggle **não é usado para avaliação**, pois seus rótulos não são públicos. Todo o protocolo experimental é construído sobre o conjunto de treino oficial com partições próprias.

### Como obter os dados

É necessário possuir conta no Kaggle e aceitar os termos da competição.

No Google Colab, o notebook:

```text
notebooks/pipeline_preprocessamento_extracao_features_colab.ipynb
```

usa o comando:

```bash
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data
```

e extrai apenas:

```text
stage_2_train_images/*
stage_2_train_labels.csv
```

Também é possível baixar manualmente os arquivos pela página da competição e organizá-los conforme a estrutura acima.

## Amostra científica usada nos resultados finais

Embora o script aceite outros tamanhos de amostra, **os resultados finais registrados no notebook de modelagem foram produzidos com 500 pacientes**.

Distribuição validada:

| Classe | Pacientes |
|---|---:|
| Negativo (`Target=0`) | 387 |
| Positivo (`Target=1`) | 113 |
| **Total** | **500** |

Prevalência positiva: **22,6%**.

A amostragem é feita por paciente, de forma estratificada e reprodutível, com **seed 42**.

Para regenerar a mesma configuração de tamanho usada no experimento final:

```bash
python scripts/extract_features.py \
  --raw data/raw \
  --out data/processed \
  --sampling natural \
  --n-total 500 \
  --jobs -1
```

O arquivo `sample_manifest.csv` registra exatamente os pacientes selecionados naquela execução.

> Importante: o valor padrão do script é configurável e pode ser diferente. Para reproduzir os números finais deste TP1, use `--sampling natural --n-total 500`.

## Pipeline de pré-processamento

Cada radiografia DICOM é processada de forma independente:

```text
DICOM
  ↓
leitura com pydicom
  ↓
validação do PatientID
  ↓
tratamento de MONOCHROME1
  ↓
resize para 256 × 256
  ↓
normalização robusta pelos percentis 1 e 99
  ↓
escala de intensidade [0,1]
  ↓
CLAHE (clip_limit=0.01)
  ↓
extração de características
```

O manifesto da imagem preserva metadados DICOM úteis para auditoria quando disponíveis.

## Características hand-crafted

São utilizadas três famílias de descritores.

| Família | Configuração resumida | Nº de atributos |
|---|---|---:|
| HOG | 9 orientações, células 16×16, blocos 2×2 | 324 |
| LBP | P=8, R=1, método uniforme, histograma normalizado | 10 |
| GLCM | 32 níveis, distâncias 1 e 3, 4 ângulos, 6 propriedades | 12 |
| **Total** |  | **346** |

As regiões são redimensionadas para 64×64 antes da extração dos descritores.

O estudo de ablação compara automaticamente:

```text
HOG
LBP
GLCM
HOG + LBP
HOG + GLCM
LBP + GLCM
HOG + LBP + GLCM
```

## Regiões candidatas para detecção

A localização usa janelas quadradas multiescala sobre a imagem pré-processada de 256×256:

| Tamanho | Regiões por imagem |
|---:|---:|
| 48×48 | 49 |
| 64×64 | 49 |
| 96×96 | 36 |
| 128×128 | 25 |
| 160×160 | 16 |
| **Total** | **175** |

Stride: **32 pixels**.

Uma região é rotulada como positiva durante o treinamento quando:

```text
IoU máximo com alguma bounding box verdadeira >= 0,30
```

O campo `max_iou` é salvo exclusivamente para auditoria/rotulagem e **não entra nas features do modelo**.

Com 500 pacientes, a matriz regional possui 87.500 regiões candidatas.

## Arquivos derivados

A etapa de extração produz:

```text
data/processed/
├── sample_manifest.csv
├── image_manifest.csv
├── features_all.csv
├── features_regions.csv
└── ground_truth_boxes.csv
```

Função de cada arquivo:

| Arquivo | Conteúdo |
|---|---|
| `sample_manifest.csv` | pacientes e rótulos da amostra |
| `image_manifest.csv` | identificadores e metadados DICOM |
| `features_all.csv` | uma linha por radiografia com HOG/LBP/GLCM |
| `features_regions.csv` | 175 regiões por paciente, caixas e descritores |
| `ground_truth_boxes.csv` | bounding boxes verdadeiras dos positivos amostrados |

Esses CSVs podem ser grandes e não precisam ser versionados; o código que os regenera está no repositório.

## Validação dos dados antes do treino

Antes de executar os experimentos:

```bash
python scripts/validate_features.py \
  --features data/processed/features_all.csv \
  --auto-ablation
```

A execução final validou:

```text
500 linhas
500 pacientes únicos
387 negativos
113 positivos
346 características
0 linhas duplicadas
0 linhas adicionais por paciente
```

O validador também falha antecipadamente em caso de NaN, infinito, alvo inválido ou ausência das famílias de features esperadas.

## Modelagem

### Baseline trivial

É executado `DummyClassifier(strategy="prior")`, obrigatório para estabelecer o piso de comparação em uma base desbalanceada.

### Modelos clássicos

O estudo principal avalia:

- SVM linear;
- SVM RBF;
- Random Forest;
- HistGradientBoosting.

O projeto possui suporte opcional a PCA e à repetição sem pesos de classe, mas **essas análises opcionais não fazem parte dos números finais atualmente registrados no notebook**.

## Protocolo experimental

Seed global:

```text
42
```

Classificação:

```text
CV externo: 5 folds
CV interno: 3 folds
seleção interna: Average Precision (AUC-PR)
unidade de separação: paciente
```

Os hiperparâmetros são escolhidos somente dentro do CV interno. O fold externo não é usado para ajuste.

Transformações ajustáveis aos dados ficam dentro do pipeline. Os SVMs usam `StandardScaler`.

As predições finais de classificação são **out-of-fold**: cada observação é prevista por um modelo que não a utilizou em seu treino.

## Tratamento de desbalanceamento

No estudo principal de classificação, são usados pesos de classe nos estimadores compatíveis.

Existe suporte para análise de sensibilidade sem pesos:

```bash
python scripts/run_experiment.py \
  --features data/processed/features_all.csv \
  --auto-ablation \
  --class-weight none
```

Essa execução é opcional e **não deve ser descrita no artigo como realizada caso não tenha sido executada e registrada**.

Na detecção regional, o desbalanceamento é tratado por undersampling de negativos **somente no treino**, limitado por padrão a aproximadamente 3 negativos para cada região positiva.

## Métricas

### Classificação

São calculadas por fold:

- AUC-ROC;
- AUC-PR;
- sensibilidade;
- especificidade;
- F1;
- balanced accuracy.

Acurácia simples não é usada como métrica principal.

### Detecção/localização

São calculados:

- IoU;
- AP em múltiplos limiares;
- mAP 0.50:0.95;
- FROC em diferentes taxas de falsos positivos por imagem.

Também é aplicado Non-Maximum Suppression (NMS) por imagem.

## Resultados finais registrados

### Baseline trivial

```text
ROC-AUC            0.500 ± 0.000
PR-AUC             0.226 ± 0.049
Sensibilidade      0.000 ± 0.000
Especificidade     1.000 ± 0.000
F1                 0.000 ± 0.000
Balanced Accuracy  0.500 ± 0.000
```

### Maior PR-AUC média observada

**HOG + LBP + Random Forest**

```text
ROC-AUC            0.757 ± 0.054
PR-AUC             0.494 ± 0.111
Sensibilidade      0.085 ± 0.058
Especificidade     0.975 ± 0.033
F1                 0.137 ± 0.090
Balanced Accuracy  0.530 ± 0.025
```

A alta especificidade e a baixa sensibilidade mostram que essa configuração é conservadora no limiar padrão, embora tenha apresentado a maior PR-AUC média.

### Configuração com maior equilíbrio entre sensibilidade e especificidade

**LBP + GLCM + SVM linear**

```text
ROC-AUC            0.734 ± 0.092
PR-AUC             0.465 ± 0.145
Sensibilidade      0.691 ± 0.128
Especificidade     0.689 ± 0.051
F1                 0.499 ± 0.125
Balanced Accuracy  0.690 ± 0.081
```

### Ablação relevante

```text
HOG + LBP + Random Forest
PR-AUC = 0.494 ± 0.111

HOG + LBP + GLCM + Random Forest
PR-AUC = 0.493 ± 0.118
```

Adicionar GLCM ao conjunto HOG+LBP não produziu ganho relevante de PR-AUC nessa configuração.

### Detecção/localização

PR-AUC regional por fold:

```text
Fold 1: 0.1129
Fold 2: 0.1509
Fold 3: 0.1405
Fold 4: 0.1548
Fold 5: 0.1326
```

Média aproximada:

```text
0.138 ± 0.017
```

AP/mAP:

```text
AP@IoU 0.50 = 0.02298
AP@IoU 0.55 = 0.01020
AP@IoU 0.60 = 0.00439
AP@IoU 0.65 = 0.00049
AP@IoU 0.70 = 0.00009
AP@IoU >= 0.75 = 0

mAP@0.50:0.95 = 0.00382
```

FROC:

| FP por imagem | Sensibilidade |
|---:|---:|
| 0.125 | 0.0536 |
| 0.25 | 0.0893 |
| 0.5 | 0.1131 |
| 1 | 0.1250 |
| 2 | 0.1548 |
| 4 | 0.2024 |
| 8 | 0.2381 |

O resultado evidencia que o baseline clássico consegue extrair sinal para classificação global, mas apresenta forte limitação para localização espacial precisa com janelas candidatas fixas.

## Limitações conhecidas

- janelas candidatas quadradas, fixas e não aprendidas;
- ausência de segmentação explícita dos pulmões;
- redução da radiografia para 256×256;
- redução das regiões para 64×64 antes dos descritores;
- variabilidade de tamanho e forma das opacidades;
- desbalanceamento entre classes e entre regiões;
- anotações de pneumonia com subjetividade e ruído;
- `Target=0` agrega diferentes condições sem pneumonia;
- mAP reduz rapidamente com aumento do limiar de IoU.

Essas limitações devem ser discutidas no artigo e na análise qualitativa de erro.

## Como reproduzir localmente

Recomendado: **Python 3.11**.

### 1. Criar ambiente

Windows/PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Executar testes

```bash
pytest -q
```

O GitHub Actions também executa testes, validação do contrato e um smoke test ponta a ponta em cada push/PR relevante.

### 3. Obter os dados

Baixe o conjunto de treino do Kaggle e organize em `data/raw/`.

### 4. Gerar a amostra e as features finais

```bash
python scripts/extract_features.py \
  --raw data/raw \
  --out data/processed \
  --sampling natural \
  --n-total 500 \
  --jobs -1
```

### 5. Validar a matriz

```bash
python scripts/validate_features.py \
  --features data/processed/features_all.csv \
  --auto-ablation
```

### 6. Rodar o estudo principal de classificação

```bash
python scripts/run_experiment.py \
  --features data/processed/features_all.csv \
  --auto-ablation \
  --outer-splits 5 \
  --inner-splits 3 \
  --seed 42 \
  --output-dir results \
  --figures-dir figures \
  --plots best
```

### 7. Rodar detecção/localização

```bash
python scripts/run_detection_experiment.py \
  --features-regions data/processed/features_regions.csv \
  --ground-truth data/processed/ground_truth_boxes.csv \
  --image-manifest data/processed/image_manifest.csv \
  --model hist_gradient_boosting \
  --family "HOG+LBP+GLCM" \
  --outer-splits 5 \
  --inner-splits 3 \
  --max-neg-pos-ratio 3 \
  --nms-iou 0.30 \
  --top-k 20 \
  --seed 42 \
  --output-dir results_detection
```

## Execução pelo Google Colab

Há dois notebooks principais.

### Pré-processamento e extração

```text
notebooks/pipeline_preprocessamento_extracao_features_colab.ipynb
```

Responsável por:

- instalar o projeto;
- autenticar no Kaggle;
- baixar os DICOMs;
- testar o pipeline;
- extrair HOG/LBP/GLCM;
- gerar matrizes de imagem e de regiões;
- salvar arquivos derivados.

Para reproduzir exatamente o experimento final, confirme que a etapa de extração está configurada para:

```text
--sampling natural --n-total 500
```

### Modelagem, avaliação e detecção

```text
notebooks/pipeline_modelagem_avaliacao_colab.ipynb
```

O notebook final:

- clona a branch `main`;
- instala dependências;
- executa os testes;
- monta o Google Drive;
- valida `features_all.csv`;
- roda classificação completa;
- executa a ablação;
- executa detecção/localização;
- salva métricas e figuras;
- consolida os números usados no artigo.

## Saídas da classificação

```text
results/
├── dataset_audit.json
├── experiment_manifest.json
├── metrics_folds.csv
├── summary_numeric.csv
├── summary_formatted.csv
├── ablation_summary.csv
├── predictions_oof.csv
├── error_cases_top.csv
├── best_params.json
└── article_table.md

figures/
├── roc_*.png
├── pr_*.png
└── confusion_*.png
```

`error_cases_top.csv` contém falsos positivos e falsos negativos priorizados para análise qualitativa.

## Saídas da detecção

A execução da detecção produz:

```text
region_metrics_folds.csv
detection_candidates_oof.csv.gz
detection_predictions.csv
detection_metrics.json
detection_best_params.json
detection_manifest.json
```

Os artefatos OOF são salvos antes do pós-processamento final para evitar perda dos folds em caso de falha posterior.

## Estrutura do repositório

```text
.
├── .github/workflows/       # CI
├── docs/                    # protocolo, metodologia e instruções
├── notebooks/               # Colabs de extração e modelagem
├── scripts/                 # CLIs reproduzíveis
├── src/rsna_baseline/       # implementação do pipeline
├── tests/                   # testes automatizados
├── requirements.txt         # dependências fixadas
├── pyproject.toml
└── README.md
```

Documentação adicional:

```text
docs/protocolo_experimental.md
docs/metodologia_features.md
docs/execucao_modelagem_avaliacao_colab.md
docs/checklist_modelagem_avaliacao.md
```

## Smoke test sem dados RSNA

Para verificar a infraestrutura sem baixar o dataset:

```bash
python scripts/generate_smoke_features.py
python scripts/validate_features.py \
  --features data/processed/features_smoke.csv \
  --auto-ablation

python scripts/run_experiment.py \
  --features data/processed/features_smoke.csv \
  --families "HOG=hog_" \
  --models svm_linear \
  --outer-splits 2 \
  --inner-splits 2
```

**Resultados do smoke test não são resultados científicos e não devem ser usados no artigo.**

## Reprodutibilidade

Para a correção, a execução deve ser possível a partir de ambiente limpo seguindo apenas este README.

O projeto adota:

- seed fixa 42;
- dependências versionadas em `requirements.txt`;
- caminhos relativos no código;
- geração versionada das matrizes intermediárias;
- testes automatizados;
- GitHub Actions;
- validação explícita da entrada;
- predições out-of-fold;
- manifesto dos experimentos;
- notebooks e scripts para reproduzir tabelas e figuras.

Os dados brutos não ficam no GitHub, mas sua origem e processo de obtenção estão documentados.

## Análise de erro

O trabalho exige análise qualitativa de acertos e falhas.

A classificação gera:

```text
error_cases_top.csv
```

Esse arquivo deve ser cruzado com as radiografias originais para inspecionar:

- falsos positivos de maior escore;
- falsos negativos de menor escore;
- padrões visuais recorrentes;
- possíveis relações com projeção/qualidade da imagem;
- hipóteses para falha dos descritores clássicos.

A análise qualitativa final e os exemplos visuais devem ser incorporados ao artigo.

## Requisitos do artigo final

O relatório deve seguir o **template da SBC** e ter no máximo **4 páginas, incluindo as referências**.

Estrutura obrigatória:

1. título, autores e afiliação;
2. resumo em português, até 10 linhas;
3. abstract em inglês, até 10 linhas;
4. Introdução;
5. Trabalhos Relacionados;
6. Metodologia;
7. Resultados;
8. Conclusão;
9. Referências.

A seção de Resultados deve incluir:

- tabela comparativa descritor × modelo com média ± desvio-padrão;
- comparação explícita com o baseline trivial;
- ao menos uma figura;
- discussão dos resultados;
- análise qualitativa de erro.

As referências devem totalizar **pelo menos 10**, sendo **no mínimo 6 artigos de periódicos ou conferências revisados por pares**.

Cada decisão metodológica não trivial deve ser justificada por literatura ou por evidência experimental/ablação.

O link deste repositório deve aparecer no corpo do artigo, preferencialmente na Metodologia ou em nota de rodapé na primeira página.

## Entregáveis

Para a entrega final do TP1:

1. PDF do artigo, no template SBC, máximo de 4 páginas;
2. fontes do artigo (`.zip` do LaTeX ou `.docx`);
3. link público/liberado deste repositório;
4. tabela de contribuição individual assinada, em arquivo separado.

O pacote final deve ser entregue em um único ZIP com nome no padrão:

```text
TP1_E1_<desafio>.zip
```

A tabela de contribuição não conta no limite de quatro páginas.

## Checklist de fechamento em relação ao enunciado

### Já coberto pelo repositório

- [x] código versionado;
- [x] `requirements.txt` com versões;
- [x] scripts/notebooks de reprodução;
- [x] leitura e pré-processamento DICOM;
- [x] amostragem reproduzível com seed;
- [x] HOG, LBP e GLCM;
- [x] baseline trivial;
- [x] SVM linear e RBF;
- [x] Random Forest;
- [x] gradient boosting;
- [x] validação aninhada;
- [x] separação por paciente;
- [x] métricas de classificação;
- [x] mAP/IoU e FROC;
- [x] ablação de famílias de descritores;
- [x] geração de tabela média ± desvio;
- [x] curvas/matriz de confusão;
- [x] arquivo de casos de erro;
- [x] testes automatizados e CI;
- [x] README com origem dos dados e instruções de execução.

### Itens que pertencem ao fechamento científico/documental

- [ ] finalizar EDA com exemplos visuais e metadados DICOM relevantes;
- [ ] concluir análise qualitativa dos erros usando as imagens originais;
- [ ] consolidar/revisar pelo menos 10 referências, com pelo menos 6 peer-reviewed;
- [ ] concluir artigo no template SBC;
- [ ] garantir resumo e abstract dentro do limite;
- [ ] colocar o link do repositório no artigo;
- [ ] incluir nota de uso de IA generativa no final do artigo;
- [ ] gerar e assinar a tabela individual de contribuição;
- [ ] empacotar PDF, fontes, contribuição e demais entregáveis no ZIP final;
- [ ] executar uma última reprodução em ambiente limpo/Colab novo.

## Declaração de uso de IA generativa

Durante a produção do projeto foram utilizadas as ferramentas de IA generativa:

- **ChatGPT (OpenAI)**;
- **Claude (Anthropic)**.

As ferramentas foram utilizadas como **apoio ao desenvolvimento**, incluindo revisão e organização textual, auxílio em dúvidas de programação, depuração, documentação e apoio à redação.

A equipe permanece responsável pela verificação do código, dos resultados, das interpretações e das referências utilizadas. Conteúdo gerado por IA não substitui literatura científica e referências bibliográficas devem ser conferidas nas fontes originais antes da entrega.

Conforme exigido pelo enunciado, esta declaração também deve aparecer em **nota ao final do artigo**, indicando o uso das ferramentas e sua finalidade.

## Observação sobre integridade acadêmica

O uso de IA generativa é declarado como apoio. Os resultados científicos reportados devem corresponder às execuções registradas, e referências não devem ser incluídas sem verificação pela equipe.

Não utilizar resultados de smoke test no artigo e não reportar como executadas análises opcionais que não tenham sido efetivamente rodadas e registradas.
