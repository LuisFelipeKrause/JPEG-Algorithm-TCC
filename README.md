## Luís Felipe Krause de Castro

# 📦 Compressão de Imagens: JPEG e Transformadas

Este projeto implementa dois algoritmos de compressão baseados no JPEG:

1. O arquivo `JPEG_Algorithm_Quantization.py` executa o algoritmo JPEG utilizando três matrizes de quantização diferentes:

   - **padrão**: matriz utilizada no JPEG padrão.
   - **moderado**: compressão intermediária.
   - **agressivo**: compressão forte, maior redução de tamanho.

   Para cada imagem, o script:

   - Divide a imagem em **blocos 8x8**.
   - Aplica **DCT (Discrete Cosine Transform)** nos blocos.
   - Quantiza os coeficientes com a tabela selecionada.
   - Reconstrói a imagem com **IDCT**.
   - Calcula métricas de qualidade e compressão:
     - **PSNR** (Peak Signal-to-Noise Ratio)
     - **SSIM** (Structural Similarity Index)
     - **% de coeficientes zerados**
     - **Tamanho original e comprimido**
     - **Taxa de compressão**
   - Exibe **visualizações comparativas** entre original e imagens reconstruídas.

2. O arquivo `JPEG_Algorithm_Transform.py` realiza compressão utilizando outras transformadas além da DCT, permitindo testar:

   - **Fourier**
   - **Laplace**
   - **Wavelet**

   Para cada transformada, o script:

   - Processa a imagem em blocos.
   - Aplica **quantização**.
   - Reconstrói a imagem.
   - Calcula as mesmas métricas de qualidade e compressão do JPEG.
   - Gera comparativos gráficos entre original e imagens reconstruídas.

## ✨ Visão Geral

O fluxo de processamento inclui:

1. **Aplicação de transformadas**  
   - Pode ser **DCT**, **Fourier**, **Laplace** ou **Wavelet (Haar)**.  
   - Executado em **blocos 8x8**.

2. **Quantização dos coeficientes**  
   - Para JPEG, são usadas três matrizes de quantização: **padrão**, **moderado** e **agressivo**.

3. **Reconstrução da imagem**  
   - A imagem é reconstruída a partir dos coeficientes transformados, usando **IDCT**, **iFFT**, ou **IDWT**, dependendo da transformada utilizada na compressão.

4. **Compressão com Huffman**  
   - Os coeficientes quantizados são codificados com **Huffman** para estimar o tamanho comprimido em bytes.

5. **Cálculo de métricas de qualidade e compressão**  
   - **PSNR (Peak Signal-to-Noise Ratio)**  
   - **SSIM (Structural Similarity Index)**  
   - **% de coeficientes zerados**  
   - **Taxa de compressão (original/comprimido)**

6. **Visualização dos resultados**  
   - Comparativos gráficos entre **original** e imagens reconstruídas.  
   - Boxplots, scatter plots e violin plots para análise estatística das métricas.


## 🧰 Tecnologias Utilizadas

- **Python 3** – Linguagem principal para implementação do algoritmo e scripts de análise.  
- **NumPy** – Operações matriciais, manipulação de arrays e cálculos numéricos.  
- **OpenCV** – Leitura, escrita, redimensionamento e conversão de cores das imagens.  
- **Matplotlib** – Visualização das imagens e criação de gráficos comparativos.  
- **SciPy** – Implementação de DCT/IDCT e transformadas de Fourier.  
- **PyWavelets** – Transformadas Wavelet (Haar) para compressão full-frame.  
- **Scikit-Image** – Cálculo de métricas de qualidade de imagem como PSNR e SSIM.  
- **Pandas** – Armazenamento e manipulação de resultados em DataFrames para análise.  
- **Seaborn** – Criação de gráficos estatísticos, como boxplots e violin plots, de forma estética.


## 🖼️ Conjunto de Imagens

As imagens utilizadas devem estar no diretório `./img/`. Durante a execução, cada imagem é automaticamente redimensionada para `512x512`.

- As bases de imagens incluem:
  - **SIPI Image Database** ([link](http://sipi.usc.edu/database/))
  - **ImageNet Sample Images** ([link](http://www.image-net.org/))

---

## 📋 Tabelas de Quantização

Para o algoritmo JPEG (DCT), são utilizadas três tabelas diferentes para simular níveis de compressão:

| Nome      | Descrição                          |
|-----------|------------------------------------|
| padrão    | Quantização leve (alta qualidade)  |
| moderado  | Quantização intermediária          |
| agressivo | Quantização forte (alta compressão)|

---

## ⚙️ Como Funciona o Algoritmo

### 1. 🔁 Processamento em Lote

O script percorre automaticamente todas as imagens da pasta `./img/` e aplica as transformadas ou tabelas de quantização configuradas.

### 2. 📥 Compressão e Reconstrução

Para cada imagem:

- É dividida em **blocos 8x8**
- É aplicada a transformada escolhida
- Coeficientes são **quantizados**
- A imagem é **reconstruída** para avaliação da qualidade

### 3. 🧪 Cálculo de Métricas

Após a reconstrução, são calculadas:

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **% de coeficientes zerados** após quantização
- **Tamanho original vs. comprimido** (Huffman)
- **Taxa de compressão**

### 4. 📦 Compressão com Huffman

- Os coeficientes quantizados são concatenados
- Gerada a **árvore de Huffman**
- Estimado o **tamanho comprimido em bytes**
- Calculada a **taxa de compressão (original/comprimido)**

### 5. 🖼️ Visualização

Para imagens selecionadas (`img_comparativo`), é exibida uma comparação:

- Original
- Reconstruída com cada tabela ou transformada
- Métricas de qualidade sobrepostas nos títulos das figuras

---

## 📈 Exemplo de Saída

### Plotagem
<div style="text-align: center;"> <img src="./img/exemplo_resultado.png" alt="Resultado do Algoritmo"> </div>
