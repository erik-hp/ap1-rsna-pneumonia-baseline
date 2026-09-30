# Protocolo experimental — frente Erik

Este documento fixa o protocolo de modelagem e avaliação antes de observar resultados finais. Alterações posteriores precisam ser justificadas e registradas.

## 1. Unidade experimental e vazamento

O conjunto oficial de teste do Kaggle não será usado porque seus rótulos não são públicos. Todos os experimentos serão feitos a partir do conjunto de treino oficial da RSNA.

A divisão será feita por **paciente**, representado por `patientId`. Nenhum paciente poderá aparecer simultaneamente no treino e no teste de uma dobra.

Mesmo que no RSNA 2018 cada `patientId` costume identificar uma radiografia, o agrupamento permanece explícito para tornar a regra de ausência de vazamento verificável.

## 2. Semente

Semente global: **42**. Toda operação aleatória que ofereça `random_state` deve usar essa semente ou uma derivação determinística.

## 3. Validação cruzada aninhada

Protocolo padrão:

- CV externo: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`;
- CV interno: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + fold)`;
- hiperparâmetros escolhidos somente no CV interno;
- desempenho estimado somente no fold externo nunca usado no ajuste;
- seleção interna por **Average Precision (AUC-PR)**.

Transformações ajustadas aos dados permanecem dentro do `Pipeline`. Os SVMs usam `StandardScaler`; PCA, quando ativado, também fica dentro do pipeline.

## 4. Baseline trivial

`DummyClassifier(strategy="prior")` é sempre executado. Ele estabelece o piso de comparação exigido no TP.

## 5. Modelos clássicos

1. SVM linear;
2. SVM RBF;
3. Random Forest;
4. HistGradientBoosting.

As grades de hiperparâmetros são deliberadamente compactas para manter reprodutibilidade e custo computacional compatível com o trabalho.

## 6. Desbalanceamento

Estratégia principal: pesos de classe (`class_weight`) nos modelos que suportam esse mecanismo.

Motivação operacional: a estratégia atua no treinamento sem criar observações sintéticas e pode ser mantida dentro do estimador de cada fold. O código permite repetir o experimento com `--class-weight none` para uma análise de sensibilidade.

**Importante para o artigo:** a justificativa metodológica final do uso de pesos de classe precisa ser acompanhada de referência bibliográfica lida pela equipe. O código não substitui essa citação.

## 7. Famílias de descritores e ablação

Contrato principal:

- `hog_*`;
- `lbp_*`;
- `glcm_*`.

Com `--auto-ablation`, o estudo gera automaticamente:

- HOG;
- LBP;
- GLCM;
- HOG+LBP;
- HOG+GLCM;
- LBP+GLCM;
- HOG+LBP+GLCM.

Isso permite medir a contribuição incremental das famílias sem mudar manualmente o protocolo.

## 8. PCA opcional

PCA não é ativado no experimento principal por padrão. O parâmetro `--pca-variance 0.95` permite executar uma análise adicional preservando 95% da variância.

Como o PCA é ajustado dentro do pipeline e dentro de cada fold, não há ajuste prévio no conjunto completo.

A decisão de reportar PCA deve depender da dimensionalidade real e de justificativa metodológica; não deve ser ativada apenas para procurar um número melhor.

## 9. Métricas de classificação

Por fold:

- AUC-ROC;
- AUC-PR;
- sensibilidade;
- especificidade;
- F1;
- acurácia balanceada.

Acurácia simples não é tratada como métrica principal.

## 10. Detecção/localização

O desafio original é de detecção. Classificação por imagem não substitui localização.

Para localização, o pipeline precisa receber regiões candidatas com:

```text
patientId,x,y,width,height,<features...>
```

O avaliador calcula AP em diferentes limiares de IoU, mAP 0.50:0.95 e FROC.

Se a frente de features entregar somente um vetor por radiografia, a classificação poderá ser avaliada, mas a localização ficará incompleta. Por isso, as regiões candidatas e suas coordenadas precisam ser preservadas para o experimento final.

## 11. Auditoria de entrada

Antes de treinar, `scripts/validate_features.py` verifica:

- presença de `patientId` e `Target`;
- alvo 0/1;
- NaN nas features;
- infinitos;
- quantidade de pacientes/grupos;
- prevalência positiva;
- quantidade de features por família;
- presença e validade básica das caixas candidatas, quando existentes.

O mesmo relatório é salvo automaticamente como `results/dataset_audit.json`.

## 12. Saídas auditáveis

O experimento gera:

- `metrics_folds.csv`;
- `summary_numeric.csv`;
- `summary_formatted.csv`;
- `predictions_oof.csv`;
- `best_params.json`;
- `dataset_audit.json`;
- `experiment_manifest.json`;
- `article_table.md`;
- `error_cases_top.csv`;
- `ablation_summary.csv`, quando a ablação está ativa;
- curva ROC;
- curva Precision-Recall;
- matriz de confusão.

As predições são out-of-fold: cada observação é prevista por um modelo que não a utilizou no treinamento daquele fold.

## 13. Análise de erro

`error_cases_top.csv` seleciona falsos positivos com maior escore e falsos negativos com menor escore, separadamente por descritor e modelo. Esse arquivo será entregue à frente de análise qualitativa para inspeção das imagens originais.

## 14. Congelamento

Depois que a equipe aprovar a amostra, a definição das regiões candidatas, as três famílias de descritores e os folds, esses elementos devem ser congelados.

Mudanças posteriores devem ser registradas como correção metodológica ou experimento de ablação/sensibilidade, não como tentativa informal de melhorar o resultado final.
