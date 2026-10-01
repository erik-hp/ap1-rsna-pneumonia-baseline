# Metodologia — dados, pré-processamento e características

Frente do Ruan. Código: `scripts/extract_features.py`, `src/rsna_baseline/preprocessing.py` e `src/rsna_baseline/features.py`. Testes: `tests/test_features.py`.

## Fluxo

```
stage_2_train_labels.csv + stage_2_train_images/*.dcm
  -> amostra estratificada por paciente (seed 42)
  -> load_dicom -> preprocess (256x256, percentis 1-99, CLAHE)
  -> regiões candidatas (imagem inteira + 25 janelas)
  -> rótulo da região por IoU com as caixas reais
  -> describe: HOG + LBP + GLCM (346 características)
  -> data/processed/*.csv
```

Cada imagem é processada de forma independente. Nada é ajustado ao conjunto inteiro, então a extração não gera vazamento. Normalizadores e PCA ficam no `Pipeline` da frente de modelagem.

## Dados e amostragem

- Fonte: RSNA Pneumonia Detection Challenge 2018 (Kaggle), treino do estágio 2. O teste oficial não tem rótulos públicos.
- Unidade de amostra: o paciente. `Target` do paciente = 1 se houver ao menos uma bounding box.
- Amostra estratificada por classe: `--n-per-class` pacientes por classe (padrão 1000), sorteados com `random_state=42`. A contagem por classe é impressa na execução.
- A amostra padrão é balanceada (50/50), o que pode diferir da proporção natural de positivos. Isso afeta a discussão de desbalanceamento no artigo.

## Pré-processamento (`preprocessing.py`)

| Etapa | Parâmetro | Motivo |
|---|---|---|
| Leitura | `pydicom`; inverte se MONOCHROME1 | Garante o mesmo sentido de intensidade em todas as imagens |
| Resize | 256 x 256, anti-aliasing | Padroniza a escala e reduz o custo |
| Normalização | percentis 1 e 99 -> [0, 1] | Reduz a influência de pixels extremos e marcadores |
| Contraste | CLAHE, `clip_limit=0.01` | Realça o contraste local das opacidades |

O resize vem antes do CLAHE, por velocidade.

## Regiões candidatas

Imagem inteira mais 25 janelas quadradas sobre a grade de 256 x 256: 16 de 112 px e 9 de 160 px, passo de 48 px. As coordenadas (`x`, `y`, `width`, `height`) saem na escala original da imagem, comparáveis às caixas reais.

Uma janela recebe `Target = 1` se o IoU máximo com alguma caixa real for maior ou igual a 0,3 (`IOU_POS`). As caixas reais só rotulam janelas de treino e nunca entram como característica.

## Descritores (`features.py`)

Cada região é redimensionada para 64 x 64. Total: 346 características por região.

| Família | Parâmetros | Nº | Prefixo |
|---|---|---|---|
| HOG | `orientations=9`, `pixels_per_cell=(16, 16)`, `cells_per_block=(2, 2)` | 324 | `hog_` |
| LBP | `P=8`, `R=1`, `method="uniform"`, histograma normalizado | 10 | `lbp_` |
| GLCM | 32 níveis, distâncias 1 e 3, ângulos 0, 45, 90 e 135 graus (média), simétrica e normalizada; contrast, dissimilarity, homogeneity, energy, correlation, ASM | 12 | `glcm_` |

Valores não finitos (por exemplo, correlação em recortes uniformes) viram 0.

### Por que estes descritores

A pneumonia aparece na radiografia como opacidades: regiões com intensidade e textura diferentes do tecido saudável. As três famílias capturam esse sinal de ângulos diferentes, o que também atende ao mínimo de três famílias do enunciado.

- **HOG (gradiente e bordas):** descreve a distribuição local de orientações de borda e a estrutura da região (Dalal e Triggs, 2005). É o mais indireto para este problema, porque as opacidades costumam ter bordas difusas, mas capta estruturas como costelas e vasos, que influenciam o que o modelo vê.
- **LBP (textura local):** codifica padrões locais de textura e, na versão rotação-invariante uniforme usada aqui, é invariante a mudanças monotônicas de intensidade e à rotação (Ojala et al., 2002). Isso ajuda porque o contraste varia entre aparelhos e exames.
- **GLCM (textura estatística):** resume estatísticas de segunda ordem da relação entre pares de pixels (Haralick et al., 1973), ligadas à homogeneidade e à aspereza de uma região, que é o que distingue uma opacidade de pulmão aerado.
- **CLAHE:** realça contraste local para tornar as opacidades mais separáveis do fundo (Zuiderveld, 1994).

### Origem dos parâmetros (o que está justificado e o que não está)

| Parâmetro | Justificativa | Status |
|---|---|---|
| HOG `orientations=9` | Valor usado por Dalal e Triggs (2005) | Justificado |
| HOG célula 16 x 16, bloco 2 x 2 | Gera grade de 4 x 4 células no recorte de 64 x 64 | Escolha prática, sem ablação |
| LBP `P=8`, `R=1`, uniforme | Configuração padrão da literatura; produz 10 bins (P + 2) | Justificado |
| GLCM 32 níveis | Evita matrizes esparsas em recortes pequenos | Escolha prática, sem ablação |
| GLCM distâncias 1 e 3, 4 ângulos | Textura fina e um pouco mais grossa; média dos ângulos reduz dependência de direção | Escolha prática, sem ablação |
| GLCM propriedades | Contraste, energia/ASM, homogeneidade e correlação vêm do conjunto de Haralick; `dissimilarity` é adicional da implementação do scikit-image | Justificado em parte |
| CLAHE `clip_limit=0.01` | Valor padrão do scikit-image | Não ajustado |
| Janelas 112 e 160 px, passo 48, IoU 0,3 | Escolha inicial | Não validada |

Pendências para o artigo: o enunciado exige referência que mostre o desempenho desses descritores em radiografia (a revisão de literatura do Natanael deve fornecê-la), e o tamanho das janelas deve ser comparado com a distribuição das dimensões das caixas reais que a EDA vai produzir.

## Saídas (`data/processed/`)

| Arquivo | Conteúdo |
|---|---|
| `features_all.csv` | `patientId, Target, hog_*, lbp_*, glcm_*`; uma linha por radiografia |
| `features_regions.csv` | `patientId, x, y, width, height, Target, hog_*, lbp_*, glcm_*`; 25 linhas por paciente |
| `ground_truth_boxes.csv` | `patientId, x, y, width, height` das caixas reais dos pacientes amostrados |
| `image_manifest.csv` | `patientId, Target, rows, cols, ViewPosition, PatientSex, PatientAge` de todas as imagens |

`features_regions.csv` ultrapassa 100 MB com a amostra padrão e não deve ir para o GitHub.

## Reprodutibilidade

- Seed 42 e caminhos relativos.
- Dependências: `pydicom`, `scikit-image`, `pandas`, `numpy`, `joblib` (declarar com versões no `requirements.txt`).
- Execução:

```
python scripts/extract_features.py --raw data/raw --out data/processed --n-per-class 1000
python scripts/validate_features.py --features data/processed/features_all.csv --auto-ablation
pytest -q
```

## Limitações

- Janelas de tamanho fixo e não aprendidas.
- Poucos positivos entre as janelas, com desbalanceamento maior que no nível da imagem.
- Limiar de IoU, tamanho das janelas e passo definidos sem ablação.
- Sem recorte de ROI pulmonar: as janelas cobrem também regiões fora do pulmão.
- `Target = 0` mistura pacientes normais e com opacidade não pneumônica (classes detalhadas não usadas).
- O resize para 64 x 64 e o resize antes do CLAHE perdem detalhe fino.
- Sem família de intensidade (histogramas e estatísticas de região).

## Referências a ler antes de citar

- Dalal e Triggs. Histograms of oriented gradients for human detection. CVPR, 2005.
- Ojala, Pietikäinen e Mäenpää. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI, 2002.
- Haralick, Shanmugam e Dinstein. Textural features for image classification. IEEE Trans. SMC, 1973.
- Zuiderveld. Contrast limited adaptive histogram equalization. Graphics Gems IV, 1994.
- Shih et al. Augmenting the NIH chest radiograph dataset with expert annotations of possible pneumonia. Radiology: Artificial Intelligence, 2019.
- van der Walt et al. scikit-image: image processing in Python. PeerJ, 2014.
