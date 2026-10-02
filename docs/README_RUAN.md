# Execução da parte do Ruan — Dados, Pré-processamento e Extração de Características

Este documento descreve como executar toda a etapa desenvolvida por **Ruan Pablo Saraiva Ripardo** no projeto **RSNA 2018 Pneumonia Detection Challenge**.

A responsabilidade desta etapa é transformar os dados brutos da RSNA em arquivos de características prontos para serem utilizados pela etapa de modelagem.

Fluxo geral:

```text
Download dos dados
        ↓
Leitura dos DICOMs
        ↓
Validação de metadados
        ↓
Pré-processamento
        ↓
Resize 256 × 256
        ↓
Normalização
        ↓
CLAHE
        ↓
Extração de HOG / LBP / GLCM
        ↓
Geração de regiões candidatas
        ↓
Cálculo de IoU
        ↓
features_all.csv
features_regions.csv
ground_truth_boxes.csv
image_manifest.csv
sample_manifest.csv
```

---

## 1. Estrutura esperada do projeto

A partir da raiz do repositório:

```text
ap1-rsna-pneumonia-baseline/
├── data/
├── docs/
├── notebooks/
├── scripts/
│   ├── download_data.py
│   ├── extract_features.py
│   ├── validate_features.py
│   └── ...
├── src/
│   └── rsna_baseline/
│       ├── preprocessing.py
│       ├── features.py
│       └── ...
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

Os dados brutos e arquivos grandes gerados não devem ser versionados no GitHub.

---

## 2. Criar e ativar o ambiente virtual

No Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Depois da ativação, o terminal deverá mostrar algo semelhante a:

```text
(.venv) usuario@computador:~/ap1-rsna-pneumonia-baseline$
```

---

## 3. Instalar as dependências

Com o ambiente virtual ativado:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

O `requirements.txt` deve incluir também:

```text
kagglehub
```

O comando:

```bash
pip install -e .
```

instala o pacote `rsna_baseline` em modo editável e permite que os scripts encontrem corretamente os módulos localizados em `src/`.

---

## 4. Baixar os dados automaticamente

Antes do primeiro download, a conta do Kaggle deve ter aceitado os termos da competição:

```text
RSNA Pneumonia Detection Challenge
```

Depois disso, execute:

```bash
python scripts/download_data.py
```

Se for necessário autenticar a conta Kaggle:

```bash
python scripts/download_data.py --login
```

O script organiza automaticamente os arquivos necessários em:

```text
data/
└── raw/
    ├── stage_2_train_labels.csv
    └── stage_2_train_images/
        ├── <patientId>.dcm
        ├── <patientId>.dcm
        └── ...
```

Exemplo:

```text
data/raw/stage_2_train_images/051e6e97-fda7-4b5c-8d5f-084bf3607d22.dcm
```

O nome do arquivo DICOM deve corresponder ao `patientId` existente no CSV de rótulos.

---

## 5. Executar os testes automatizados

Antes de iniciar a extração:

```bash
pytest -q
```

O esperado é que todos os testes sejam aprovados.

Na versão atual do projeto:

```text
17 passed
```

Os testes verificam elementos como:

- geração das regiões candidatas;
- limites das janelas;
- cálculo de IoU;
- extração de características;
- dimensões dos vetores;
- ausência de valores inválidos em funções testadas;
- comportamento determinístico de componentes importantes.

---

## 6. Pré-processamento utilizado

Cada DICOM é carregado por `load_dicom()`.

São lidos a imagem e metadados relevantes, incluindo, quando disponíveis:

```text
PatientID
Rows
Columns
Modality
ViewPosition
PhotometricInterpretation
PixelSpacing
WindowCenter
WindowWidth
RescaleSlope
RescaleIntercept
PatientSex
PatientAge
```

Também é verificado se o `PatientID` do DICOM corresponde ao `patientId` do CSV.

O pipeline aplicado à imagem é:

```text
DICOM
  ↓
pixel_array
  ↓
tratamento de MONOCHROME1
  ↓
resize para 256 × 256
  ↓
normalização por percentis 1 e 99
  ↓
valores limitados ao intervalo 0–1
  ↓
CLAHE
```

O resultado final é uma imagem `float32` com tamanho:

```text
256 × 256
```

---

## 7. Características extraídas

São utilizadas três famílias de descritores clássicos:

```text
HOG
LBP
GLCM
```

A matriz final possui:

```text
HOG  = 324 características
LBP  = 10 características
GLCM = 12 características
-----------------------------
Total = 346 características
```

As colunas são identificadas por prefixo:

```text
hog_...
lbp_...
glcm_...
```

---

## 8. Regiões candidatas

Para a tarefa de localização, são utilizadas regiões candidatas multi-escala.

Configuração atual:

```text
Imagem: 256 × 256

Tamanhos de janela:
48 × 48
64 × 64
96 × 96
128 × 128
160 × 160

Stride:
32

IoU mínimo para uma região positiva:
0.30
```

Essa configuração gera:

```text
175 regiões candidatas por imagem
```

As bounding boxes verdadeiras são utilizadas somente para calcular o IoU e rotular as regiões de treinamento.

Elas não são utilizadas como entrada do modelo na inferência.

---

## 9. Executar uma amostra pequena para teste

Antes de gerar uma amostra maior, pode-se testar o pipeline com 100 pacientes balanceados:

```bash
python scripts/extract_features.py \
  --raw data/raw \
  --out data/processed_test_100 \
  --sampling balanced \
  --n-per-class 50 \
  --jobs 2
```

Essa execução seleciona:

```text
50 pacientes negativos
50 pacientes positivos
100 pacientes no total
```

Com 175 regiões por imagem:

```text
100 × 175 = 17.500 regiões candidatas
```

Essa configuração foi utilizada para validar a integração do pipeline.

---

## 10. Executar uma amostra com distribuição natural

Para uma execução mais próxima da distribuição original do conjunto:

```bash
python scripts/extract_features.py \
  --raw data/raw \
  --out data/processed \
  --sampling natural \
  --n-total 500 \
  --jobs 2
```

Neste modo:

```text
--sampling natural
```

a proporção entre positivos e negativos é preservada aproximadamente.

No conjunto utilizado durante o desenvolvimento, a distribuição original observada foi:

```text
Target 0: 20672 pacientes
Target 1:  6012 pacientes
Total:    26684 pacientes
```

---

## 11. Executar com todos os pacientes disponíveis

Também é possível utilizar:

```bash
python scripts/extract_features.py \
  --raw data/raw \
  --out data/processed \
  --sampling all \
  --jobs 2
```

Essa opção processa todos os pacientes disponíveis localmente.

Ela pode demandar bastante tempo e espaço em disco.

---

## 12. Modos de amostragem

O script suporta:

```text
balanced
natural
all
```

### Balanced

Seleciona a mesma quantidade por classe.

Exemplo:

```bash
--sampling balanced --n-per-class 50
```

### Natural

Seleciona uma quantidade total mantendo aproximadamente a proporção original.

Exemplo:

```bash
--sampling natural --n-total 500
```

### All

Processa todos os pacientes disponíveis.

```bash
--sampling all
```

A seed utilizada para tornar a seleção reproduzível é:

```text
42
```

---

## 13. Paralelismo

O parâmetro:

```bash
--jobs
```

controla quantos processos trabalham simultaneamente.

Exemplos:

```bash
--jobs 1
```

usa apenas um worker.

```bash
--jobs 2
```

usa dois workers.

```bash
--jobs -1
```

permite utilizar todos os núcleos disponíveis.

Cada paciente corresponde a uma task.

Portanto:

```text
100 pacientes = 100 tasks
500 pacientes = 500 tasks
```

Em uma execução de desenvolvimento com 500 pacientes e `--jobs 2`, foram processadas 500 tasks.

---

## 14. Arquivos gerados

Ao final da execução, o diretório informado em `--out` recebe:

```text
sample_manifest.csv
image_manifest.csv
features_all.csv
features_regions.csv
ground_truth_boxes.csv
```

### sample_manifest.csv

Registra exatamente quais pacientes foram selecionados.

Estrutura principal:

```text
patientId
Target
```

Serve para auditoria e reprodutibilidade da amostragem.

### image_manifest.csv

Contém:

```text
patientId
Target
dicom_file
metadados DICOM
```

É utilizado para auditoria dos dados e rastreabilidade.

### features_all.csv

É o arquivo principal entregue para a etapa de classificação.

Formato:

```text
patientId,Target,hog_...,lbp_...,glcm_...
```

Cada paciente possui uma linha.

### features_regions.csv

É utilizado para a etapa de detecção/localização.

Formato geral:

```text
patientId
x
y
width
height
Target
max_iou
hog_...
lbp_...
glcm_...
```

As coordenadas são devolvidas na escala original da radiografia.

### ground_truth_boxes.csv

Contém as caixas verdadeiras:

```text
patientId
x
y
width
height
```

Esse arquivo é utilizado posteriormente na avaliação da detecção.

---

## 15. Resultado da validação das regiões candidatas

Durante o desenvolvimento foi realizado um teste ampliado com:

```text
100 pacientes
50 positivos
50 negativos
```

Foram produzidas:

```text
17.500 regiões candidatas
16.709 negativas
791 positivas
```

Dos 50 pacientes positivos:

```text
48 possuíam pelo menos uma região com IoU >= 0.30
2 não possuíam região positiva
```

Isso corresponde a uma cobertura de candidatos de:

```text
48 / 50 = 96%
```

Esse valor representa somente a cobertura das regiões candidatas nesta amostra.

Ele não corresponde ao recall do modelo final de detecção.

Os dois casos sem região positiva apresentaram bounding boxes pequenas após o redimensionamento, sendo considerada essa uma limitação da configuração atual de sliding windows.

---

## 16. Validar o arquivo de características

Depois de gerar o `features_all.csv`:

```bash
python scripts/validate_features.py \
  --features data/processed/features_all.csv \
  --auto-ablation
```

Para o teste de 100 pacientes:

```bash
python scripts/validate_features.py \
  --features data/processed_test_100/features_all.csv \
  --auto-ablation
```

A validação verifica aspectos como:

```text
número de linhas
pacientes únicos
classes
prevalência
quantidade de features
duplicatas
NaN
inf
prefixos das características
combinações para ablação
```

Em uma execução de validação com 100 pacientes, foram encontrados:

```text
rows: 100
unique_groups: 100

classes:
0: 50
1: 50

feature_count_total: 346

HOG: 324
LBP: 10
GLCM: 12

duplicate_full_rows: 0
```

Também foram reconhecidas as combinações:

```text
HOG
LBP
GLCM
HOG + LBP
HOG + GLCM
LBP + GLCM
HOG + LBP + GLCM
```

---

## 17. Entrega para a etapa do Erik

Depois que o arquivo passar pela validação, os principais arquivos entregues para integração são:

```text
features_all.csv
features_regions.csv
ground_truth_boxes.csv
sample_manifest.csv
image_manifest.csv
```

O arquivo principal para a classificação é:

```text
features_all.csv
```

A sequência de integração é:

```text
Ruan
  ↓
DICOM + labels
  ↓
pré-processamento
  ↓
HOG / LBP / GLCM
  ↓
features_all.csv
  ↓
validação
  ↓
Erik
  ↓
modelagem
  ↓
Nested Cross-Validation
  ↓
métricas e ablações
```

---

## 18. Como reproduzir toda a parte do Ruan do zero

Em uma máquina limpa:

```bash
git clone https://github.com/erik-hp/ap1-rsna-pneumonia-baseline.git
cd ap1-rsna-pneumonia-baseline
```

Criar o ambiente:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Baixar os dados:

```bash
python scripts/download_data.py
```

Caso seja necessária autenticação:

```bash
python scripts/download_data.py --login
```

Executar os testes:

```bash
pytest -q
```

Gerar uma amostra natural de 500 pacientes:

```bash
python scripts/extract_features.py \
  --raw data/raw \
  --out data/processed \
  --sampling natural \
  --n-total 500 \
  --jobs 2
```

Validar:

```bash
python scripts/validate_features.py \
  --features data/processed/features_all.csv \
  --auto-ablation
```

Ao final, os arquivos ficam em:

```text
data/processed/
```

---

## 19. Fluxo resumido de execução

```bash
# Ativar ambiente
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
pip install -e .

# Baixar os dados
python scripts/download_data.py

# Testar o código
pytest -q

# Extrair as características
python scripts/extract_features.py \
  --raw data/raw \
  --out data/processed \
  --sampling natural \
  --n-total 500 \
  --jobs 2

# Validar as características
python scripts/validate_features.py \
  --features data/processed/features_all.csv \
  --auto-ablation
```

---

## 20. Saída final esperada

Ao concluir a parte do Ruan:

```text
data/processed/
├── sample_manifest.csv
├── image_manifest.csv
├── features_all.csv
├── features_regions.csv
└── ground_truth_boxes.csv
```

O arquivo que serve como entrada principal para os modelos clássicos é:

```text
features_all.csv
```

---

## 21. Observações importantes

- Não versionar os DICOMs no GitHub.
- Evitar versionar `features_regions.csv`, pois o arquivo pode ficar muito grande.
- Sempre executar `pytest -q` antes da extração definitiva.
- Manter a seed em `42` para reprodutibilidade.
- Não alterar os parâmetros de pré-processamento sem registrar a alteração.
- Para o experimento principal, preferir `sampling natural` caso o objetivo seja preservar o desbalanceamento observado no conjunto.
- A amostra `balanced` é especialmente útil para testes rápidos e integração.
- Os arquivos intermediários podem ser regenerados pelos scripts, portanto o repositório deve priorizar código, documentação e instruções de reprodução.
