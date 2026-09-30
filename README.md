# AP1 — RSNA 2018 Pneumonia Detection Baseline

Baseline clássico para o **RSNA 2018 Pneumonia Detection Challenge**, desenvolvido para a disciplina **Tópicos Especiais em Sistemas de Informação**.

O objetivo é estabelecer um piso de desempenho reproduzível usando somente **características hand-crafted** e **modelos clássicos de aprendizado de máquina**, sem redes neurais profundas.

## Equipe

- Erik Holanda Pires — modelagem, protocolo experimental, avaliação e integração final
- Ruan Pablo Saraiva Ripardo — dados, pré-processamento e extração de características
- Natanael Douglas Alves Feijão — EDA, literatura, análise de erro e redação científica

## O que a frente de modelagem já cobre

- baseline trivial com `DummyClassifier`;
- SVM linear e RBF;
- Random Forest;
- HistGradientBoosting;
- validação cruzada aninhada com `StratifiedGroupKFold`;
- separação por paciente/grupo e verificação de overlap;
- hiperparâmetros ajustados somente dentro do treino;
- AUC-ROC, AUC-PR, sensibilidade, especificidade, F1 e balanced accuracy;
- IoU, AP/mAP e FROC;
- ablação automática de HOG/LBP/GLCM e de todas as combinações;
- sensibilidade com/sem `class_weight`;
- PCA opcional dentro do pipeline;
- auditoria do CSV antes de treinar;
- tabela média ± desvio, predições out-of-fold e casos de erro;
- manifesto do experimento com versões;
- smoke test ponta a ponta no GitHub Actions.

A seed padrão é **42**.

## Instalação

Recomendado: Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Contrato de integração das features

Arquivo esperado:

```text
data/processed/features_all.csv
```

Classificação por radiografia:

```text
patientId,Target,hog_...,lbp_...,glcm_...
```

Para detecção/localização, preservar também:

```text
patientId,x,y,width,height,Target,hog_...,lbp_...,glcm_...
```

Não versionar DICOMs nem dados brutos.

## Pré-validar antes de gastar tempo treinando

```powershell
python scripts/validate_features.py --features data/processed/features_all.csv --auto-ablation
```

Esse comando falha cedo se houver NaN, infinito, colunas ausentes ou alvo fora de 0/1.

## Estudo principal

Quando as features reais estiverem disponíveis:

```powershell
python scripts/run_experiment.py --features data/processed/features_all.csv --auto-ablation
```

A ablação automática executa:

```text
HOG
LBP
GLCM
HOG+LBP
HOG+GLCM
LBP+GLCM
HOG+LBP+GLCM
```

contra o baseline trivial e os modelos clássicos configurados.

Por padrão, as figuras são geradas apenas para a configuração de maior AUC-PR média, evitando dezenas de imagens. Use `--plots all` para gerar figuras de todas as combinações.

## Saídas

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

`article_table.md` já deixa a comparação descritor × modelo em formato fácil de transportar para o artigo.

## Sensibilidade ao desbalanceamento

O estudo principal usa pesos de classe. Para comparar com o mesmo protocolo sem pesos:

```powershell
python scripts/run_experiment.py --features data/processed/features_all.csv --auto-ablation --class-weight none --output-dir results_no_class_weight --figures-dir figures_no_class_weight
```

A justificativa acadêmica no artigo deve receber referência bibliográfica real; o código apenas deixa a comparação reproduzível.

## PCA opcional

Somente se a dimensionalidade real justificar:

```powershell
python scripts/run_experiment.py --features data/processed/features_all.csv --auto-ablation --pca-variance 0.95 --output-dir results_pca --figures-dir figures_pca
```

O PCA fica dentro do pipeline e é ajustado somente no treino de cada fold.

## Avaliar localização

Ground truth:

```text
patientId,x,y,width,height
```

Predições:

```text
patientId,x,y,width,height,score
```

```powershell
python scripts/evaluate_detection.py --ground-truth data/processed/ground_truth_boxes.csv --predictions results/detection_predictions.csv --image-col patientId --image-manifest data/processed/image_manifest.csv
```

O manifesto com todas as imagens, inclusive negativas, é recomendado para FROC correto em FP/imagem.

## Smoke test local sem dados RSNA

Isso testa o pipeline, mas **não produz resultado científico**:

```powershell
python scripts/generate_smoke_features.py
python scripts/validate_features.py --features data/processed/features_smoke.csv --auto-ablation
python scripts/run_experiment.py --features data/processed/features_smoke.csv --families "HOG=hog_" --models svm_linear --outer-splits 2 --inner-splits 2
```

## Reprodutibilidade final

Antes da entrega:

1. criar ambiente Python 3.11 limpo;
2. instalar somente pelo `requirements.txt`;
3. executar `pytest -q`;
4. gerar as features reais seguindo a frente de pré-processamento;
5. rodar a pré-validação;
6. rodar o estudo principal com seed 42;
7. verificar que tabelas e figuras coincidem com as usadas no artigo.

A lista operacional completa está em `docs/checklist_final_erik.md`.
