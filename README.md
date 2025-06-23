# 📦 Compressão de Imagens com DCT + Huffman

Este projeto implementa um pipeline de compressão de imagens utilizando a **Transformada Discreta do Cosseno (DCT)**, **quantização escalonada** e **codificação de Huffman**, simulando conceitos utilizados no padrão JPEG.

---

## ✨ Visão Geral

O algoritmo executa:

1. **DCT em blocos 8x8** nos canais RGB da imagem.
2. **Quantização** dos coeficientes com 3 níveis diferentes.
3. **Reconstrução** da imagem com IDCT.
4. **Compressão com Huffman** dos coeficientes quantizados.
5. **Cálculo de métricas** de qualidade e compressão.
6. **Visualização dos resultados** em um painel de comparação.

---

## 🧰 Tecnologias Utilizadas

- Python 3
- NumPy
- OpenCV
- Matplotlib
- SciPy
- Scikit-Image
- Pandas

---

## 🖼️ Conjunto de Imagens

As imagens utilizadas devem estar no diretório `./img/` com os seguintes nomes:

- `casa.tiff`
- `aereo.tiff`
- `peppers.tiff`
- `casa_carro.tiff`
- `caca.tiff`
- `doces.tiff`
- `mandril.tiff`


Cada imagem é redimensionada para `512x512` automaticamente durante a execução.

---

## 📋 Tabelas de Quantização

São utilizadas três tabelas diferentes para simular níveis de compressão:

| Nome      | Descrição                          |
|-----------|------------------------------------|
| padrão    | Quantização leve (alta qualidade)  |
| moderado  | Quantização intermediária          |
| agressivo | Quantização forte (alta compressão)|

As tabelas são aplicadas individualmente aos blocos 8x8 de cada canal da imagem.

---

## ⚙️ Como Funciona o Algoritmo

### 1. 🔁 Processamento em Lote

O script percorre automaticamente todas as imagens e aplica a compressão com cada tabela de quantização.

### 2. 📥 Compressão e Reconstrução

Para cada imagem:

- É dividida em blocos `8x8`
- É aplicada DCT e quantização
- Reconstruída com IDCT para avaliação da qualidade

### 3. 🧪 Cálculo de Métricas

Após a reconstrução, são calculadas:

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **% de coeficientes zerados** após quantização

### 4. 📦 Compressão com Huffman

- Os coeficientes DCT quantizados são concatenados
- Gerada uma árvore de Huffman
- É estimado o **tamanho comprimido em bytes**
- Calculada a **taxa de compressão (original/comprimido)**

### 5. 📊 Resultados

Os resultados são apresentados em um `DataFrame` no terminal com todas as métricas para cada tabela.

### 6. 🖼️ Visualização

Para cada imagem processada, é exibida uma visualização comparativa:

- Original
- Reconstruída com `padrão`
- Reconstruída com `moderado`
- Reconstruída com `agressivo`

---

## 📈 Exemplo de Saída

### No Terminal

```bash
🔍 Processando imagem: casa

 Tabela   PSNR (dB)   SSIM   % Coef. Zerados   Tamanho RAW (bytes)   Comprimido (bytes)   Compressão Huffman (x)
 padrão       30.21  0.925             71.23                786432              112345                    7.00
 moderado     27.10  0.840             82.65                786432               78987                    9.96
 agressivo    24.40  0.715             90.23                786432               56321                   13.96
```

### Imagem Gerada na Plotagem

<div style="text-align: center;">
  <img src="./img/caça.png" alt="Resultado do Algoritmo">
</div>
