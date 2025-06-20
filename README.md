# Simulador de Compressão JPEG em Python

Este projeto é um simulador educacional do algoritmo de compressão de imagens **JPEG**, desenvolvido como parte do Trabalho de Conclusão de Curso em Ciência da Computação. A implementação é feita em **Python**, utilizando processamento por blocos, transformada DCT, quantização e codificação de Huffman.

---

## 📖 Descrição

O algoritmo realiza a **compressão e descompressão** de imagens em **RGB**, seguindo os principais passos do JPEG clássico:

- Transformada Discreta do Cosseno (DCT)
- Quantização (com três níveis configuráveis)
- Leitura em zigue-zague
- Codificação de Huffman
- Reconstrução da imagem via IDCT

Além disso, o simulador calcula métricas como **PSNR**, **SSIM**, **taxa de compressão**, e porcentagem de coeficientes zerados.

---

## 🚀 Funcionalidades

- ✅ Suporte a imagens `RGB`
- ✅ Três níveis de quantização: `padrão`, `moderado`, `agressivo`
- ✅ Avaliação de qualidade com PSNR e SSIM
- ✅ Compressão com Huffman simulada
- ✅ Visualização das imagens reconstruídas

---

## 🖼️ Exemplo de saída

| **Tabela** | **PSNR (dB)** | **SSIM** | **% Coef. Zerados** | **Tamanho RAW (bytes)** | **Comprimido (bytes)** | **Compressão Huffman (x)** |
| ---------- | ------------- | -------- | ------------------- | ----------------------- | ---------------------- | -------------------------- |
| padrão     | 37.09         | 0.9264   | 92.36               | 3,145,782               | 528,083                | 5.96                       |
| moderado   | 34.02         | 0.8706   | 96.57               | 3,145,782               | 457,480                | 6.88                       |
| agressivo  | 31.25         | 0.8110   | 97.57               | 3,145,782               | 435,912                | 7.22                       |

---

## ⚙️ Bibliotecas

  - `numpy`
  - `opencv-python`
  - `matplotlib`
  - `scikit-image`
  - `pandas`
  - `scipy`
