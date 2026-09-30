# Checklist final — frente Erik

## Pronto antes dos dados reais

- [x] Baseline trivial obrigatório.
- [x] SVM linear e RBF.
- [x] Random Forest.
- [x] Gradient boosting.
- [x] Seed fixa.
- [x] Particionamento por paciente/grupo.
- [x] Nested cross-validation.
- [x] Ajuste de hiperparâmetros somente no treino.
- [x] Verificação de overlap de grupos.
- [x] AUC-ROC, AUC-PR, sensibilidade, especificidade, F1 e balanced accuracy.
- [x] IoU, AP/mAP e FROC.
- [x] Predições out-of-fold.
- [x] Curvas ROC/PR e matriz de confusão.
- [x] Tabela média ± desvio.
- [x] Ablação automática das combinações HOG/LBP/GLCM.
- [x] Análise opcional com/sem class_weight.
- [x] PCA opcional dentro do pipeline, sem vazamento.
- [x] Auditoria automática do CSV antes do treinamento.
- [x] Seleção automática de falsos positivos e falsos negativos para análise qualitativa.
- [x] Manifesto do experimento com versões e parâmetros.
- [x] Tabela Markdown pronta para transportar ao artigo.
- [x] Smoke dataset determinístico.
- [x] Smoke test ponta a ponta no GitHub Actions.
- [x] README e protocolo documentados.

## Pendente somente quando os dados reais chegarem

- [ ] Rodar a auditoria em `features_all.csv`.
- [ ] Congelar amostra, famílias e definição das regiões candidatas.
- [ ] Rodar o estudo completo HOG/LBP/GLCM.
- [ ] Rodar ablação automática.
- [ ] Conferir média e desvio entre folds.
- [ ] Gerar figuras finais.
- [ ] Avaliar bounding boxes reais.
- [ ] Exportar os casos FP/FN para análise de erro.
- [ ] Se necessário, rodar sensibilidade sem class_weight.
- [ ] Se a dimensionalidade justificar, rodar PCA como análise adicional.
- [ ] Registrar os números usados no artigo.
- [ ] Reproduzir os números em ambiente limpo.
- [ ] Conferir que o README, sozinho, é suficiente para reprodução.

## Comandos finais

Pré-validação:

```powershell
python scripts/validate_features.py --features data/processed/features_all.csv --auto-ablation
```

Estudo principal:

```powershell
python scripts/run_experiment.py --features data/processed/features_all.csv --auto-ablation
```

Sensibilidade sem pesos de classe, se a equipe decidir reportar:

```powershell
python scripts/run_experiment.py --features data/processed/features_all.csv --auto-ablation --class-weight none --output-dir results_no_class_weight --figures-dir figures_no_class_weight
```

PCA opcional, somente se houver justificativa metodológica:

```powershell
python scripts/run_experiment.py --features data/processed/features_all.csv --auto-ablation --pca-variance 0.95 --output-dir results_pca --figures-dir figures_pca
```

Avaliação de detecção após produzir caixas candidatas pontuadas:

```powershell
python scripts/evaluate_detection.py --ground-truth data/processed/ground_truth_boxes.csv --predictions results/detection_predictions.csv --image-col patientId --image-manifest data/processed/image_manifest.csv
```
