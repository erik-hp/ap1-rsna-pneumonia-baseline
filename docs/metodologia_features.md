# Metodologia — dados, pré-processamento e características

Frente do Ruan. Código: `scripts/extract_features.py`, `src/rsna_baseline/preprocessing.py` e `src/rsna_baseline/features.py`. Testes: `tests/test_features.py`.

## Fluxo

```
stage_2_train_labels.csv + stage_2_train_images/*.dcm
  -> amostra estratificada por paciente (seed 42)
  -> load_dicom -> preprocess (256x256, percentis 1-99, CLAHE)
  -> 175 regiões candidatas multiescala por imagem
  -> rótulo da região por IoU com as caixas reais
  -> describe: HOG + LBP + GLCM (346 características)
  -> data/processed/*.csv
```

Cada imagem é processada de forma independente. Nada é ajustado ao conjunto inteiro durante a extração. Normalizadores e PCA, quando utilizados, permanecem no `Pipeline` da frente de modelagem.

## Dados e amostragem

- Fonte: RSNA Pneumonia Detection Challenge 2018 (Kaggle), treino do estágio 2.
- Unidade de amostra: paciente, identificado por `patientId`.
- `Target` do paciente = 1 se houver ao menos uma bounding box positiva.
- Seed fixa: 42.
- Modo padrão: `--sampling natural --n-total 2000`, preservando aproximadamente a proporção original das classes.
- `--sampling balanced --n-per-class N` existe para testes controlados e comparações.
- `--sampling all` utiliza todos os pacientes disponíveis.
- `sample_manifest.csv` registra exatamente quais pacientes participaram de cada execução.

## Pré-processamento (`preprocessing.py`)

| Etapa | Parâmetro | Motivo |
|---|---|---|
| Leitura | `pydicom`; inverte MONOCHROME1 | Mantém o mesmo sentido de intensidade |
| Resize | 256 x 256, anti-aliasing | Padroniza escala e reduz custo |
| Normalização | percentis 1 e 99 -> [0, 1] | Reduz influência de pixels extremos |
| Contraste | CLAHE, `clip_limit=0.01` | Realça contraste local |

O manifesto também guarda metadados DICOM úteis para auditoria, como posição de vista, interpretação fotométrica, espaçamento de pixel e informações de janela quando disponíveis. O `PatientID` do DICOM é comparado ao identificador esperado antes do processamento.

## Regiões candidatas

As regiões candidatas são janelas quadradas sobre a imagem pré-processada de 256 x 256:

| Tamanho | Posições por eixo | Regiões |
|---:|---:|---:|
| 48 x 48 | 7 | 49 |
| 64 x 64 | 7 | 49 |
| 96 x 96 | 6 | 36 |
| 128 x 128 | 5 | 25 |
| 160 x 160 | 4 | 16 |
| **Total** |  | **175** |

O passo é **32 px**. As coordenadas `x`, `y`, `width` e `height` são convertidas de volta para a escala original da radiografia.

Uma região recebe `Target = 1` quando seu maior IoU com alguma caixa verdadeira é maior ou igual a **0,30**. O valor `max_iou` é salvo apenas para auditoria e rotulagem; ele **não é uma característica de entrada do modelo**. As caixas verdadeiras não são fornecidas ao classificador durante inferência.

## Descritores (`features.py`)

Cada imagem/região é redimensionada para 64 x 64 antes dos descritores. Total: **346 características**.

| Família | Parâmetros | Nº | Prefixo |
|---|---|---:|---|
| HOG | `orientations=9`, `pixels_per_cell=(16,16)`, `cells_per_block=(2,2)` | 324 | `hog_` |
| LBP | `P=8`, `R=1`, `method="uniform"`, histograma normalizado | 10 | `lbp_` |
| GLCM | 32 níveis; distâncias 1 e 3; 4 ângulos; 6 propriedades | 12 | `glcm_` |

Valores não finitos são convertidos para zero.

### Por que estes descritores

- **HOG:** descreve orientação local de gradientes e estrutura.
- **LBP:** resume padrões locais de textura.
- **GLCM:** resume relações estatísticas de segunda ordem entre pixels.
- **CLAHE:** realça contraste local antes da extração.

Os parâmetros escolhidos devem ser apresentados no artigo como configuração fixa do baseline; parâmetros sem estudo de ablação específico não devem ser descritos como otimizados.

## Saídas (`data/processed/`)

| Arquivo | Conteúdo |
|---|---|
| `sample_manifest.csv` | pacientes e rótulos usados na execução |
| `features_all.csv` | uma linha por radiografia: `patientId, Target, hog_*, lbp_*, glcm_*` |
| `features_regions.csv` | 175 linhas por paciente: coordenadas, `Target`, `max_iou` e descritores |
| `ground_truth_boxes.csv` | bounding boxes verdadeiras dos pacientes positivos amostrados |
| `image_manifest.csv` | identificadores e metadados DICOM das imagens processadas |

Os dados derivados podem ser grandes e **não são versionados no GitHub**. Eles devem ser regenerados pelo script ou armazenados externamente durante a execução.

## Reprodutibilidade

Instalação:

```bash
pip install -r requirements.txt
pip install -e .
```

Extração principal:

```bash
python scripts/extract_features.py \
  --raw data/raw \
  --out data/processed \
  --sampling natural \
  --n-total 2000
```

Validação:

```bash
python scripts/validate_features.py --features data/processed/features_all.csv --auto-ablation
pytest -q
```

O notebook `notebooks/rsna_features_colab.ipynb` é apenas uma interface conveniente para download/execução no Colab; a lógica oficial está nos módulos e scripts versionados.

## Limitações

- As janelas candidatas são fixas, quadradas e não aprendidas.
- IoU 0,30 é utilizado para rotular regiões de treino; a avaliação final deve reportar IoU/mAP/FROC separadamente.
- Não há segmentação explícita do pulmão.
- O redimensionamento para 256 x 256 e posteriormente 64 x 64 reduz detalhe fino.
- `Target = 0` agrega casos sem pneumonia, incluindo diferentes condições radiográficas.
- A matriz regional pode ficar grande; por isso os CSVs regionais não devem ser commitados.

## Referências a verificar antes de citar no artigo

- Dalal e Triggs. Histograms of oriented gradients for human detection. CVPR, 2005.
- Ojala, Pietikäinen e Mäenpää. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI, 2002.
- Haralick, Shanmugam e Dinstein. Textural features for image classification. IEEE Trans. SMC, 1973.
- Zuiderveld. Contrast limited adaptive histogram equalization. Graphics Gems IV, 1994.
- Shih et al. Augmenting the NIH chest radiograph dataset with expert annotations of possible pneumonia. Radiology: Artificial Intelligence, 2019.
- van der Walt et al. scikit-image: image processing in Python. PeerJ, 2014.
