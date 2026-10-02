# Metodologia — dados, pré-processamento e extração de características

**Responsável principal:** Ruan Pablo Saraiva Ripardo  
**Desafio:** RSNA 2018 Pneumonia Detection Challenge  
**Frente:** dados, leitura DICOM, pré-processamento, regiões candidatas e extração de características clássicas.

Código principal desta frente:

- `scripts/extract_features.py`
- `src/rsna_baseline/preprocessing.py`
- `src/rsna_baseline/features.py`
- `tests/test_features.py`

## 1. Objetivo da etapa

A etapa implementada tem como objetivo transformar os dados brutos do conjunto de treino da RSNA em matrizes de características reproduzíveis, adequadas às etapas posteriores de classificação e detecção. A solução utiliza somente técnicas clássicas de Visão Computacional, sem redes neurais profundas ou modelos pré-treinados.

O fluxo implementado é:

```text
stage_2_train_labels.csv + stage_2_train_images/*.dcm
<<<<<<< HEAD
  -> amostra estratificada por paciente (seed 42)
  -> load_dicom -> preprocess (256x256, percentis 1-99, CLAHE)
  -> 175 regiões candidatas multiescala por imagem
  -> rótulo da região por IoU com as caixas reais
  -> describe: HOG + LBP + GLCM (346 características)
  -> data/processed/*.csv
```

Cada imagem é processada de forma independente. Nada é ajustado ao conjunto inteiro durante a extração. Normalizadores e PCA, quando utilizados, permanecem no `Pipeline` da frente de modelagem.
=======
  -> seleção reproduzível de pacientes (seed 42)
  -> leitura do DICOM e coleta de metadados
  -> pré-processamento da radiografia
  -> extração de HOG + LBP + GLCM da imagem completa
  -> geração de regiões candidatas multi-escala
  -> rotulagem das regiões por IoU com as bounding boxes reais
  -> extração de HOG + LBP + GLCM por região
  -> geração dos arquivos CSV para modelagem e avaliação
```

A extração é independente por imagem. Nenhum normalizador, PCA ou seletor de atributos é ajustado sobre o conjunto completo nesta etapa; transformações que dependem dos dados permanecem na etapa de modelagem.
>>>>>>> a5e3e2f (docs: atualiza metodologia, adiciona download automático e guia de execução)

## 2. Organização dos dados

<<<<<<< HEAD
- Fonte: RSNA Pneumonia Detection Challenge 2018 (Kaggle), treino do estágio 2.
- Unidade de amostra: paciente, identificado por `patientId`.
- `Target` do paciente = 1 se houver ao menos uma bounding box positiva.
- Seed fixa: 42.
- Modo padrão: `--sampling natural --n-total 2000`, preservando aproximadamente a proporção original das classes.
- `--sampling balanced --n-per-class N` existe para testes controlados e comparações.
- `--sampling all` utiliza todos os pacientes disponíveis.
- `sample_manifest.csv` registra exatamente quais pacientes participaram de cada execução.
=======
A estrutura esperada é:
>>>>>>> a5e3e2f (docs: atualiza metodologia, adiciona download automático e guia de execução)

```text
data/
├── raw/
│   ├── stage_2_train_images/
│   │   ├── *.dcm
│   │   └── ...
│   └── stage_2_train_labels.csv
└── processed/
```

Os arquivos DICOM são dados brutos e não devem ser versionados no GitHub. O repositório deve manter o código capaz de reproduzir os arquivos processados.

O conjunto de rótulos utilizado apresentou **26.684 pacientes únicos**, distribuídos da seguinte forma:

- `Target = 0`: 20.672 pacientes;
- `Target = 1`: 6.012 pacientes.

Isso corresponde a aproximadamente 77,5% de negativos e 22,5% de positivos no conjunto disponível.

## 3. Amostragem

A seleção dos pacientes utiliza `SEED = 42`, permitindo repetir exatamente a amostra escolhida. O script passou a oferecer três estratégias:

- `balanced`: seleciona a mesma quantidade de pacientes por classe; é usada principalmente para testes rápidos e integração;
- `natural`: seleciona uma quantidade total de pacientes preservando aproximadamente a proporção original das classes;
- `all`: utiliza todos os pacientes disponíveis.

A amostra utilizada em cada execução é registrada em `sample_manifest.csv`, contendo `patientId` e `Target`. Dessa forma, é possível identificar exatamente quais pacientes participaram de cada experimento.

A amostra balanceada não representa a prevalência natural do conjunto e, por isso, deve ser usada como amostra de desenvolvimento/integração, e não como estimativa da prevalência real da doença.

## 4. Leitura dos arquivos DICOM

A leitura é feita com `pydicom`. A função `load_dicom()` recupera o `pixel_array` como `float32` e registra metadados úteis para auditoria e análise exploratória.

Os metadados contemplados são:

- `DICOMPatientID`;
- `rows` e `cols`;
- `Modality`;
- `ViewPosition`;
- `PhotometricInterpretation`;
- `PixelSpacing`;
- `WindowCenter`;
- `WindowWidth`;
- `RescaleSlope`;
- `RescaleIntercept`;
- `PatientSex`;
- `PatientAge`.

Quando `PhotometricInterpretation` é `MONOCHROME1`, a imagem é invertida para manter o mesmo sentido de intensidade das imagens `MONOCHROME2`.

O `DICOMPatientID` também é comparado ao `patientId` proveniente do CSV. Se houver inconsistência entre o rótulo e o arquivo DICOM, a execução é interrompida, evitando associação silenciosa de uma imagem ao paciente errado.

Os campos de janela, pixel spacing e rescale são mantidos no manifesto para documentação e auditoria. Eles não são utilizados diretamente como características do modelo nesta baseline.

## 5. Pré-processamento

Todas as imagens passam pela mesma sequência de pré-processamento:

| Etapa | Configuração | Finalidade |
|---|---|---|
<<<<<<< HEAD
| Leitura | `pydicom`; inverte MONOCHROME1 | Mantém o mesmo sentido de intensidade |
| Resize | 256 x 256, anti-aliasing | Padroniza escala e reduz custo |
| Normalização | percentis 1 e 99 -> [0, 1] | Reduz influência de pixels extremos |
| Contraste | CLAHE, `clip_limit=0.01` | Realça contraste local |

O manifesto também guarda metadados DICOM úteis para auditoria, como posição de vista, interpretação fotométrica, espaçamento de pixel e informações de janela quando disponíveis. O `PatientID` do DICOM é comparado ao identificador esperado antes do processamento.
=======
| Leitura | `pydicom` | Recuperar pixels e metadados do exame |
| Fotometria | inversão de `MONOCHROME1` | Uniformizar o sentido das intensidades |
| Resize | `256 x 256`, com anti-aliasing | Padronizar a dimensão e reduzir o custo computacional |
| Normalização robusta | percentis 1 e 99 | Reduzir a influência de valores extremos |
| Escala | intervalo `[0, 1]` | Padronizar a intensidade para os descritores |
| Contraste | CLAHE, `clip_limit=0.01` | Realçar contraste local |

O resize é realizado antes do CLAHE. Não foi implementado recorte explícito dos pulmões; portanto, a imagem completa e as janelas candidatas podem conter regiões externas ao parênquima pulmonar.
>>>>>>> a5e3e2f (docs: atualiza metodologia, adiciona download automático e guia de execução)

## 6. Bounding boxes e sistema de coordenadas

<<<<<<< HEAD
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
=======
As anotações positivas do arquivo `stage_2_train_labels.csv` possuem `x`, `y`, `width` e `height` na escala original da radiografia.

Como a imagem é redimensionada para `256 x 256`, as bounding boxes são convertidas temporariamente para essa escala durante o cálculo do IoU. São utilizados os fatores:
>>>>>>> a5e3e2f (docs: atualiza metodologia, adiciona download automático e guia de execução)

```text
sx = largura_original / 256
sy = altura_original / 256
```

<<<<<<< HEAD
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
=======
As coordenadas das regiões candidatas salvas no CSV são convertidas novamente para a escala original, permitindo comparação direta com as caixas verdadeiras.

As bounding boxes reais **não são usadas como características de entrada**. Elas servem apenas para rotular regiões durante a preparação do treino e para avaliação posterior da localização.

## 7. Regiões candidatas
>>>>>>> a5e3e2f (docs: atualiza metodologia, adiciona download automático e guia de execução)

Foi utilizada uma estratégia de *sliding windows* multi-escala sobre a imagem `256 x 256`.

Configuração final mantida:

```python
SIZE = 256
WIN_SIZES = (48, 64, 96, 128, 160)
WIN_STRIDE = 32
IOU_POS = 0.3
```

Essa configuração produz **175 regiões candidatas por imagem**:

- 49 janelas de `48 x 48`;
- 49 janelas de `64 x 64`;
- 36 janelas de `96 x 96`;
- 25 janelas de `128 x 128`;
- 16 janelas de `160 x 160`.

Uma região recebe `Target = 1` quando seu maior IoU com qualquer bounding box verdadeira do paciente é maior ou igual a `0,3`. O maior valor obtido também é salvo na coluna `max_iou`, permitindo auditar a qualidade da cobertura das regiões candidatas.

### 7.1 Ajuste empírico das janelas

A configuração inicial utilizava somente janelas de `112` e `160` pixels, com `stride = 48`, totalizando 25 regiões por imagem. Em um teste balanceado com 20 pacientes (10 positivos e 10 negativos), **3 dos 10 pacientes positivos não possuíam nenhuma região candidata com IoU >= 0,3**.

A análise das caixas mostrou casos pequenos, incluindo uma lesão redimensionada para aproximadamente `31,25 x 40` pixels. Com uma janela mínima de `112 x 112`, mesmo um encaixe ideal não atingiria o limiar de IoU definido.

Após a adoção das escalas `(48, 64, 96, 128, 160)` com `stride = 32`, o mesmo teste de 20 pacientes apresentou **0 pacientes positivos sem região candidata positiva**.

Em um teste ampliado com 100 pacientes balanceados (50 positivos e 50 negativos), foram produzidas 17.500 regiões:

- 16.709 regiões negativas;
- 791 regiões positivas;
- 48 dos 50 pacientes positivos apresentaram ao menos uma região com IoU >= 0,3;
- 2 dos 50 positivos não apresentaram região positiva, correspondendo a **96% de cobertura dos pacientes positivos nessa amostra**.

Os dois casos não cobertos apresentavam caixas pequenas após o resize, aproximadamente `32 x 14,5` e `30,75 x 27,25` pixels, com máximos IoU de aproximadamente `0,201` e `0,263`, respectivamente. Optou-se por manter a configuração atual e registrar esses casos como limitação da estratégia de janelas fixas.

Esse percentual de cobertura mede somente a capacidade do gerador de regiões candidatas nessa amostra; **não corresponde ao recall final do modelo de detecção**.

## 8. Extração de características

A função `describe()` recebe a imagem completa ou uma região candidata. Antes dos descritores, cada entrada é redimensionada para `64 x 64` pixels (`CROP = 64`).

São extraídas três famílias de características hand-crafted, totalizando **346 atributos**.

### 8.1 HOG

Configuração:

```python
orientations = 9
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
```

O HOG gera **324 características**, com prefixo `hog_`.

### 8.2 LBP

Configuração:

```python
P = 8
R = 1
method = "uniform"
```

O histograma é normalizado pela quantidade de pixels e produz **10 características**, com prefixo `lbp_`.

### 8.3 GLCM

A imagem é quantizada em 32 níveis de cinza. A GLCM utiliza:

- níveis: 32;
- distâncias: 1 e 3 pixels;
- ângulos: 0°, 45°, 90° e 135°;
- matriz simétrica;
- matriz normalizada.

Para cada distância, os valores são calculados nos quatro ângulos e agregados pela média. São extraídas as propriedades:

- contrast;
- dissimilarity;
- homogeneity;
- energy;
- correlation;
- ASM.

Com 6 propriedades e 2 distâncias, são obtidas **12 características GLCM**, com prefixo `glcm_`.

Qualquer valor `NaN`, `+inf` ou `-inf` retornado pelos descritores é convertido para `0.0` antes da gravação.

### 8.4 Total de características

| Família | Quantidade |
|---|---:|
| HOG | 324 |
| LBP | 10 |
| GLCM | 12 |
| **HOG + LBP + GLCM** | **346** |

As combinações reconhecidas para os estudos de ablação são:

- HOG: 324;
- LBP: 10;
- GLCM: 12;
- HOG + LBP: 334;
- HOG + GLCM: 336;
- LBP + GLCM: 22;
- HOG + LBP + GLCM: 346.

## 9. Arquivos gerados

O `extract_features.py` produz os seguintes arquivos no diretório informado em `--out`:

| Arquivo | Conteúdo |
|---|---|
<<<<<<< HEAD
| `sample_manifest.csv` | pacientes e rótulos usados na execução |
| `features_all.csv` | uma linha por radiografia: `patientId, Target, hog_*, lbp_*, glcm_*` |
| `features_regions.csv` | 175 linhas por paciente: coordenadas, `Target`, `max_iou` e descritores |
| `ground_truth_boxes.csv` | bounding boxes verdadeiras dos pacientes positivos amostrados |
| `image_manifest.csv` | identificadores e metadados DICOM das imagens processadas |

Os dados derivados podem ser grandes e **não são versionados no GitHub**. Eles devem ser regenerados pelo script ou armazenados externamente durante a execução.
=======
| `sample_manifest.csv` | pacientes selecionados e seus `Target`; permite reproduzir/auditar a amostra |
| `image_manifest.csv` | `patientId`, `Target`, nome do DICOM e metadados do exame |
| `features_all.csv` | uma linha por radiografia: `patientId`, `Target`, `hog_*`, `lbp_*`, `glcm_*` |
| `features_regions.csv` | uma linha por região: coordenadas, `Target`, `max_iou` e as 346 características |
| `ground_truth_boxes.csv` | bounding boxes verdadeiras dos pacientes positivos da amostra |

O `features_all.csv` é a principal entrada para os experimentos de classificação. O `features_regions.csv` e o `ground_truth_boxes.csv` dão suporte à etapa de detecção/localização.
>>>>>>> a5e3e2f (docs: atualiza metodologia, adiciona download automático e guia de execução)

Arquivos processados grandes, especialmente `features_regions.csv`, não precisam ser versionados no GitHub, pois são reproduzíveis a partir dos DICOMs e do script de extração. Na amostra balanceada de 100 pacientes, `features_regions.csv` atingiu aproximadamente 54 MB.

<<<<<<< HEAD
Instalação:

```bash
pip install -r requirements.txt
pip install -e .
```

Extração principal:
=======
## 10. Validação e testes realizados

A suíte `tests/test_features.py` verifica, entre outros pontos:

- quantidade e prefixos das características;
- determinismo da extração;
- ausência de valores não finitos em regiões constantes;
- normalização do histograma LBP;
- dimensão e intervalo do pré-processamento;
- limites das janelas candidatas;
- cálculo de IoU.

A suíte possui 17 testes no projeto. Após a alteração da configuração de janelas, o teste que fixava a contagem antiga de 25 regiões precisou ser atualizado para calcular a quantidade esperada a partir de `WIN_SIZES`, `WIN_STRIDE` e `SIZE`, evitando que o teste fique preso a uma configuração anterior.

A validação de `features_all.csv` para a amostra balanceada de 100 pacientes confirmou:

```text
rows: 100
unique_groups: 100
classes: 50 negativos / 50 positivos
positive_prevalence: 0.5
feature_count_total: 346
HOG: 324
LBP: 10
GLCM: 12
duplicate_full_rows: 0
```

O comando utilizado foi:

```bash
python scripts/validate_features.py \
  --features data/processed_test_100/features_all.csv \
  --auto-ablation
```

## 11. Execução paralela

A extração utiliza `joblib.Parallel`. Cada paciente corresponde a uma tarefa independente. Assim:

```text
100 pacientes = 100 tasks
500 pacientes = 500 tasks
```

Os caminhos de entrada e saída são resolvidos para caminhos absolutos em tempo de execução, evitando problemas de localização dos DICOMs nos processos paralelos do backend `Loky`, sem inserir caminhos absolutos específicos de uma máquina no código.

Em uma execução com **500 pacientes e `--jobs 2`**, as 500 tarefas de extração foram concluídas em aproximadamente **21,3 minutos** no ambiente utilizado. Após a conclusão das tasks, o processo ainda deve ser mantido ativo até terminar a consolidação e gravação dos CSVs e o prompt do terminal retornar.

## 12. Exemplos de execução

### 12.1 Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 12.2 Teste balanceado de integração

```bash
python scripts/extract_features.py \
  --raw data/raw \
  --out data/processed_test_100 \
  --sampling balanced \
  --n-per-class 50 \
  --jobs 2
```

### 12.3 Amostra preservando a distribuição natural
>>>>>>> a5e3e2f (docs: atualiza metodologia, adiciona download automático e guia de execução)

```bash
python scripts/extract_features.py \
  --raw data/raw \
  --out data/processed \
  --sampling natural \
<<<<<<< HEAD
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
=======
  --n-total 500 \
  --jobs 2
```

### 12.4 Validação

```bash
python scripts/validate_features.py \
  --features data/processed/features_all.csv \
  --auto-ablation

pytest -q
```

## 13. Reprodutibilidade

As principais medidas adotadas foram:

- `SEED = 42` para seleção dos pacientes;
- manifesto da amostra em `sample_manifest.csv`;
- pipeline único de leitura, pré-processamento e extração;
- parâmetros explícitos para HOG, LBP, GLCM, CLAHE e regiões candidatas;
- validação de `PatientID` entre DICOM e CSV;
- caminhos fornecidos por argumentos de linha de comando;
- execução reproduzível dos arquivos processados a partir dos dados brutos;
- testes automatizados;
- validação da matriz final antes da modelagem.
>>>>>>> a5e3e2f (docs: atualiza metodologia, adiciona download automático e guia de execução)

## 14. Limitações atuais

As principais limitações desta etapa são:

1. **Regiões candidatas fixas:** o sliding window não se adapta ao formato real da opacidade.
2. **Lesões muito pequenas:** na amostra balanceada de 100 pacientes, 2 de 50 positivos não atingiram `IoU >= 0,3` com nenhuma janela.
3. **Desbalanceamento regional:** no teste de 100 pacientes, somente 791 de 17.500 regiões foram positivas (aproximadamente 4,5%).
4. **Sem segmentação pulmonar:** regiões fora dos pulmões também são avaliadas.
5. **Perda de detalhe por resize:** a radiografia é reduzida para `256 x 256` e cada região para `64 x 64` antes dos descritores.
6. **Metadados não usados como features:** os metadados DICOM são preservados para auditoria, mas a baseline utiliza apenas características visuais hand-crafted.
7. **Amostra balanceada de desenvolvimento:** resultados obtidos com amostras 50/50 não devem ser interpretados como desempenho sobre a prevalência natural do conjunto.

## 15. Entrega para a etapa de modelagem

Após a extração e validação, a principal entrega desta frente para a etapa de modelagem é `features_all.csv`. Para localização, também são fornecidos `features_regions.csv` e `ground_truth_boxes.csv`.

O fluxo de integração é:

```text
DICOM + labels
  -> pré-processamento
  -> HOG / LBP / GLCM
  -> features_all.csv / features_regions.csv
  -> validação
  -> modelagem e avaliação
```

## 16. Referências metodológicas a utilizar no artigo

Antes de citar, as referências devem ser verificadas pela equipe na fonte original.

- Dalal, N.; Triggs, B. *Histograms of Oriented Gradients for Human Detection*. CVPR, 2005.
- Ojala, T.; Pietikäinen, M.; Mäenpää, T. *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*. IEEE TPAMI, 2002.
- Haralick, R. M.; Shanmugam, K.; Dinstein, I. *Textural Features for Image Classification*. IEEE Transactions on Systems, Man, and Cybernetics, 1973.
- Zuiderveld, K. *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV, 1994.
- Shih, G. et al. *Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia*. Radiology: Artificial Intelligence, 2019.
- van der Walt, S. et al. *scikit-image: Image Processing in Python*. PeerJ, 2014.
