# Protocolo experimental — frente Erik

Este documento fixa o protocolo de modelagem e avaliação antes de observar resultados finais. Alterações posteriores precisam ser justificadas e registradas.

## 1. Unidade experimental e vazamento

O conjunto oficial de teste do Kaggle não será usado porque seus rótulos não são públicos. Todos os experimentos serão feitos a partir do conjunto de treino oficial da RSNA.

A divisão será feita por **paciente**, representado por `patientId`. Nenhum paciente poderá aparecer simultaneamente no treino e no teste de uma dobra.

Mesmo que no RSNA 2018 cada `patientId` costume identificar uma radiografia, o agrupamento permanece explícito para tornar a regra de ausência de vazamento verificável.

## 2. Semente

Semente global do experimento:

```text
42
```

Toda operação aleatória que ofereça `random_state` deve utilizar essa semente ou uma derivação determinística dela.

## 3. Validação cruzada aninhada

O protocolo padrão usa:

- CV externo: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`;
- CV interno: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + fold)`;
- seleção de hiperparâmetros no CV interno;
- estimativa de desempenho somente no fold externo nunca usado no ajuste.

A métrica de seleção do CV interno é **AUC-PR / Average Precision**, escolhida por ser informativa em cenários de desbalanceamento.

Normalização e qualquer transformação ajustada aos dados devem permanecer dentro de um `Pipeline`. No código atual, os SVMs usam `StandardScaler` dentro do pipeline.

## 4. Baseline trivial

O `DummyClassifier(strategy="prior")` é executado em todas as famílias de características. Ele funciona como piso obrigatório e permite demonstrar se os descritores/modelos extraem sinal acima da prevalência da classe.

## 5. Modelos clássicos

A grade inicial contém:

1. SVM linear;
2. SVM RBF;
3. Random Forest;
4. HistGradientBoosting.

SVM e Random Forest usam balanceamento por peso de classe. HistGradientBoosting usa `class_weight="balanced"`.

As grades foram mantidas deliberadamente pequenas para que o experimento seja reproduzível dentro do prazo do TP. Se a amostra final for muito grande, a grade pode ser reduzida **antes** de olhar o desempenho do teste externo.

## 6. Famílias de descritores

O contrato de integração espera, inicialmente:

- `hog_*` — gradiente/bordas;
- `lbp_*` — textura local;
- `glcm_*` — textura estatística;
- `ALL` — concatenação de todas as features numéricas.

A comparação principal será **descritor × modelo** com média e desvio-padrão entre os folds externos.

## 7. Métricas de classificação

Por fold:

- AUC-ROC;
- AUC-PR;
- sensibilidade;
- especificidade;
- F1;
- acurácia balanceada.

Acurácia simples não é usada como métrica principal.

## 8. Detecção/localização

O desafio original é de detecção. Por isso, a classificação por imagem não deve ser apresentada como substituta de localização.

Para avaliar localização, o pipeline precisa receber regiões candidatas com:

```text
patientId,x,y,width,height,<features...>
```

Cada região candidata é pontuada pelo classificador. As coordenadas e o escore formam a saída de detecção.

O avaliador implementado calcula:

- AP em diferentes limiares de IoU;
- mAP em IoU 0.50:0.95;
- FROC em pontos de falso-positivo por imagem.

### Contrato necessário com a frente de features

Se Ruan entregar somente **uma feature vector por radiografia**, será possível medir classificação, mas **não** localização. Para cumprir a avaliação de detecção, a extração deve também produzir candidatos espaciais (por exemplo, janela/grade/ROI proposta por método clássico) com as coordenadas da região.

Os rótulos de treino dos candidatos podem ser derivados das bounding boxes oficiais, deixando explícito o limiar de IoU adotado. Esse limiar deve ser definido antes do experimento final.

## 9. Saídas auditáveis

O script principal gera:

- `results/metrics_folds.csv`;
- `results/summary_numeric.csv`;
- `results/summary_formatted.csv`;
- `results/predictions_oof.csv`;
- `results/best_params.json`;
- curvas ROC e PR;
- matriz de confusão.

As predições são **out-of-fold**, portanto cada linha é prevista por um modelo que não treinou naquela observação.

## 10. Congelamento

Depois que a equipe aprovar:

1. a amostra;
2. a definição de região candidata;
3. as três famílias de descritores;
4. o protocolo de folds;

esses itens serão congelados. Mudanças posteriores devem ser registradas como correção metodológica ou ablação, não como tentativa informal de melhorar o resultado final.
