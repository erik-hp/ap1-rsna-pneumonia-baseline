# Execução da frente do Erik no Google Colab

Este guia executa a parte pesada de modelagem, avaliação e detecção sem depender da máquina local.

## Arquivos de entrada

No Google Drive:

```text
MyDrive/RSNA_TP1/data/processed/
├── features_all.csv
├── features_regions.csv
├── ground_truth_boxes.csv
└── image_manifest.csv
```

`features_regions.csv` precisa ser a versão regenerada com as 175 regiões candidatas da configuração atual.

## Notebook

Use:

```text
notebooks/erik_experimentos_colab.ipynb
```

Ele executa:

1. instalação e testes;
2. validação de `features_all.csv`;
3. estudo principal de classificação com nested CV 5x3;
4. ablação HOG/LBP/GLCM;
5. detecção regional OOF por paciente;
6. NMS;
7. mAP 0.50:0.95 e FROC;
8. salvamento direto no Google Drive.

## Classificação

O experimento principal continua sendo:

```bash
python scripts/run_experiment.py \
  --features data/processed/features_all.csv \
  --auto-ablation \
  --outer-splits 5 \
  --inner-splits 3 \
  --seed 42
```

## Detecção

A detecção usa `scripts/run_detection_experiment.py`.

O protocolo é:

- split externo por paciente;
- nenhum paciente aparece em treino e teste;
- undersampling de regiões negativas somente dentro do treino;
- busca de hiperparâmetros somente no treino externo;
- predições out-of-fold para todas as regiões do teste;
- NMS por imagem;
- avaliação contra `ground_truth_boxes.csv`.

Comando final:

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
  --seed 42
```

O `max_iou` presente no arquivo regional é apenas variável de auditoria/rotulagem e está explicitamente excluído das features.

## Saídas de detecção

```text
results_detection/
├── region_metrics_folds.csv
├── detection_candidates_oof.csv.gz
├── detection_predictions.csv
├── detection_metrics.json
├── detection_best_params.json
└── detection_manifest.json
```

`detection_candidates_oof.csv.gz` contém apenas coordenadas, rótulo, fold e score, não as 346 features, reduzindo bastante o tamanho.

## Execução rápida antes da final

Para verificar se tudo funciona antes de gastar horas:

```bash
python scripts/run_detection_experiment.py \
  --features-regions data/processed/features_regions.csv \
  --ground-truth data/processed/ground_truth_boxes.csv \
  --image-manifest data/processed/image_manifest.csv \
  --outer-splits 2 \
  --inner-splits 2 \
  --max-neg-pos-ratio 2 \
  --output-dir results_detection_smoke
```

Esse teste rápido serve apenas para conferir execução. Os números do artigo devem vir do protocolo final documentado.
