# AP1 — RSNA 2018 Pneumonia Detection Baseline

Baseline clássico para o **RSNA 2018 Pneumonia Detection Challenge**, desenvolvido para a disciplina **Tópicos Especiais em Sistemas de Informação**.

O objetivo é estabelecer um piso de desempenho reproduzível usando somente **características hand-crafted** e **modelos clássicos de aprendizado de máquina**, sem redes neurais profundas.

## Equipe

- Erik Holanda Pires — modelagem, protocolo experimental, avaliação e integração final
- Ruan Pablo Saraiva Ripardo — dados, pré-processamento e extração de características
- Natanael Douglas Alves Feijão — EDA, literatura, análise de erro e redação científica

## Protocolo implementado

A frente de modelagem já está preparada para:

- baseline trivial com `DummyClassifier`;
- SVM linear e SVM RBF;
- Random Forest;
- HistGradientBoosting;
- validação cruzada **aninhada** com `StratifiedGroupKFold`;
- separação obrigatória por paciente/grupo;
- ajuste de hiperparâmetros somente nos dados de treino;
- AUC-ROC, AUC-PR, sensibilidade, especificidade, F1 e acurácia balanceada;
- avaliação de detecção por AP/mAP com IoU e FROC;
- geração de tabelas, predições out-of-fold e figuras para o artigo.

A semente padrão é **42**.

## Estrutura

```text
.
├── docs/
│   └── protocolo_experimental.md
├── scripts/
│   ├── evaluate_detection.py
│   └── run_experiment.py
├── src/
│   └── rsna_baseline/
│       ├── evaluation.py
│       ├── experiment.py
│       ├── io.py
│       ├── modeling.py
│       └── plots.py
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Instalação

Recomendado: Python 3.11.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

## Contrato da matriz de características

Para o baseline por imagem, o CSV deve ter uma linha por radiografia e, no mínimo:

```text
patientId,Target,hog_...,lbp_...,glcm_...
```

Para localização/detecção, a matriz pode conter uma linha por região candidata e acrescentar:

```text
patientId,x,y,width,height,Target,hog_...,lbp_...,glcm_...
```

Nesse segundo formato, as coordenadas da região candidata são preservadas nas predições out-of-fold e podem ser avaliadas com o script de detecção.

## Rodar os experimentos

Exemplo quando Ruan entregar HOG, LBP e GLCM em um único CSV:

```bash
python scripts/run_experiment.py \
  --features data/processed/features_all.csv \
  --group-col patientId \
  --label-col Target \
  --families "HOG=hog_,LBP=lbp_,GLCM=glcm_,ALL=*"
```

Resultados são gravados em `results/` e figuras em `figures/`.

## Avaliar localização

O ground truth deve conter:

```text
patientId,x,y,width,height
```

As predições devem conter:

```text
patientId,x,y,width,height,score
```

Exemplo:

```bash
python scripts/evaluate_detection.py \
  --ground-truth data/processed/ground_truth_boxes.csv \
  --predictions results/detection_predictions.csv \
  --image-col patientId
```

Se existir um manifesto contendo **todas** as imagens, incluindo negativos sem caixas, informe-o com `--image-manifest`; isso torna FP/imagem da FROC correto.

## Reprodutibilidade

Não versionar DICOMs nem dados brutos. O repositório deve conter apenas código, configurações, instruções e resultados derivados pequenos. Antes da entrega, executar:

```bash
pytest -q
```

e reproduzir os experimentos em um ambiente limpo seguindo somente este README.
