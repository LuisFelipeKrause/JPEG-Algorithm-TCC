import tempfile
import os
import numpy as np
import cv2
import heapq
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.fftpack import dct, idct
import seaborn as sns
from PIL import Image

# ---------------------------
# Funções DCT
# ---------------------------

def dct2(block):
    """Aplica a Transformada Discreta de Cosseno 2D em um bloco 8x8."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Aplica a Transformada Inversa de Cosseno 2D em um bloco 8x8."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def split_into_blocks(img, block_size=8):
    """Divide a imagem em blocos quadrados de tamanho block_size x block_size."""
    h, w = img.shape
    return (img.reshape(h // block_size, block_size, -1, block_size)
              .swapaxes(1, 2)
              .reshape(-1, block_size, block_size))

def merge_blocks(blocks, img_shape, block_size=8):
    """Reconstrói a imagem a partir dos blocos processados."""
    h, w = img_shape
    return (np.array(blocks)
              .reshape(h // block_size, w // block_size, block_size, block_size)
              .swapaxes(1, 2)
              .reshape(h, w))

# ---------------------------
# Huffman encoding (simples)
# ---------------------------

class HuffmanNode:
    """Representa um nó interno na árvore de Huffman."""
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class HuffmanLeaf:
    """Representa uma folha (símbolo) na árvore de Huffman."""
    def __init__(self, symbol):
        self.symbol = symbol
    def walk(self, code, acc):
        code[self.symbol] = acc or "0"

def build_huffman_tree(freq):
    """Constrói a árvore de Huffman a partir de um dicionário de frequências."""
    heap = [(f, i, HuffmanLeaf(int(s))) for i, (s, f) in enumerate(freq.items())]
    heapq.heapify(heap)
    count = len(heap)
    while len(heap) > 1:
        f1, c1, l1 = heapq.heappop(heap)
        f2, c2, l2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, count, HuffmanNode(l1, l2)))
        count += 1
    return heap[0][2]

def huffman_encoding(data):
    """Codifica uma lista de símbolos usando Huffman e retorna o binário e o código."""
    freq = {}
    for symbol in data:
        freq[symbol] = freq.get(symbol, 0) + 1
    if not freq:  # robustez
        return "", {}
    root = build_huffman_tree(freq)
    code = {}
    root.walk(code, "")
    encoded_data = "".join(code[symbol] for symbol in data)
    return encoded_data, code

# ---------------------------
# JPEG simplificado
# ---------------------------

def jpeg_compression(img, q_table):
    """Compressão JPEG de uma matriz (Y, Cb ou Cr) usando DCT e quantização."""
    h, w = img.shape
    blocks = split_into_blocks(img)
    dct_blocks = [dct2(b.astype(np.float32)) for b in blocks]
    quantized_blocks = [np.round(b / q_table) for b in dct_blocks]
    return quantized_blocks

def jpeg_decompression(quantized_blocks, q_table, img_shape):
    """Reconstrução de uma matriz a partir dos blocos quantizados."""
    dequantized_blocks = [b * q_table for b in quantized_blocks]
    idct_blocks = [np.round(idct2(b)).astype(np.uint8) for b in dequantized_blocks]
    img_rec = merge_blocks(idct_blocks, img_shape)
    return np.clip(img_rec, 0, 255)

# ---------------------------
# Processamento por imagem
# ---------------------------

def process_image(img, q_table):
    """
    Processa uma imagem RGB:
    - Converte para YCbCr
    - Aplica JPEG nos canais
    - Calcula métricas de compressão (PSNR, SSIM, % de zeros, taxa de compressão)
    """
    img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y, Cb, Cr = cv2.split(img_ycc)

    canais_recon = []
    total_bits = 0
    coef_zerados = 0
    total_coef = 0

    # --- Canal de luminância (Y) ---
    quant_blocks = jpeg_compression(Y, q_table)
    all_coefs = np.hstack([b.flatten() for b in quant_blocks])
    coef_zerados += np.sum(all_coefs == 0)
    total_coef += all_coefs.size
    encoded_data, _ = huffman_encoding(all_coefs)
    total_bits += len(encoded_data)
    rec_Y = jpeg_decompression(quant_blocks, q_table, Y.shape)

    # --- Canais de crominância (Cb, Cr) com subamostragem 4:2:0 ---
    def process_chroma(channel):
        """Processa um canal de crominância com subamostragem 4:2:0."""
        small = cv2.resize(channel, (channel.shape[1] // 2, channel.shape[0] // 2))
        q_blocks = jpeg_compression(small, q_table)
        coefs = np.hstack([b.flatten() for b in q_blocks])
        nonlocal coef_zerados, total_coef, total_bits
        coef_zerados += np.sum(coefs == 0)
        total_coef += coefs.size
        encoded, _ = huffman_encoding(coefs)
        total_bits += len(encoded)
        rec_small = jpeg_decompression(q_blocks, q_table, small.shape)
        # reexpande de volta pro tamanho original
        rec_full = cv2.resize(rec_small, (channel.shape[1], channel.shape[0]))
        return rec_full

    rec_Cb = process_chroma(Cb)
    rec_Cr = process_chroma(Cr)

    # --- Junta de volta Y, Cb, Cr ---
    img_recon_ycc = cv2.merge([rec_Y, rec_Cb, rec_Cr])
    img_final = cv2.cvtColor(img_recon_ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    # --- Medição de tamanho da imagem original ---
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp:
        Image.fromarray(img).save(tmp.name, format="BMP")
        tamanho_raw = os.path.getsize(tmp.name)
    os.remove(tmp.name)

    tamanho_comp = total_bits / 8

    # --- Cálculo de métricas ---
    psnr = peak_signal_noise_ratio(img, img_final, data_range=255)
    try:
        ssim = structural_similarity(img, img_final, channel_axis=2)
    except TypeError:
        ssim = structural_similarity(img, img_final, multichannel=True)

    percentual_zerados = (coef_zerados / total_coef) * 100
    taxa_comp = tamanho_raw / tamanho_comp if tamanho_comp > 0 else 0

    return {
        'PSNR': psnr,
        'SSIM': ssim,
        '% de Zeros': percentual_zerados,
        'Taxa de Compressão (em x)': taxa_comp,
        'Tamanho Original (bytes)': tamanho_raw,
        'Tamanho Comprimido (bytes)': tamanho_comp
    }


# ---------------------------
# Execução principal
# ---------------------------

def main(pasta_imagens, quant_tables):
    """
    Loop principal:
    - Lê imagens da pasta
    - Redimensiona e converte para RGB
    - Aplica compressão JPEG para cada tabela de quantização
    - Armazena resultados em um DataFrame
    """
    resultados = []
    arquivos = [f for f in os.listdir(pasta_imagens) if f.endswith(".tiff") or f.endswith(".JPEG")]

    for arq in arquivos:
        caminho = os.path.join(pasta_imagens, arq)
        img = cv2.imread(caminho)
        if img is None:
            continue
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for nome_tab, q in quant_tables.items():
            met = process_image(img, q)
            met['Imagem'] = arq
            met['Quantização'] = nome_tab
            resultados.append(met)

    df = pd.DataFrame(resultados)
    print(df.head(15))
    print(df.tail(15))
    return df

# ---------------------------
# Exemplo de uso
# ---------------------------

if __name__ == "__main__":
    # Matriz padrão Q50
    padrao = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])

    moderado = np.array([
        [40,43,45,50,60,70,90,100],
        [43,45,50,60,70,90,100,110],
        [45,50,60,70,90,100,110,128],
        [50,60,70,90,100,110,128,128],
        [60,70,90,100,110,128,128,128],
        [70,90,100,110,128,128,128,128],
        [90,100,110,128,128,128,128,128],
        [100,110,128,128,128,128,128,128]
    ])
    
    agressivo = np.array([
        [80,85,90,100,120,140,180,200],
        [85,90,100,120,140,180,200,220],
        [90,100,120,140,180,200,220,255],
        [100,120,140,180,200,220,255,255],
        [120,140,180,200,220,255,255,255],
        [140,180,200,220,255,255,255,255],
        [180,200,220,255,255,255,255,255],
        [200,220,255,255,255,255,255,255]
    ])

    tabelas = {
        "Padrão": padrao,
        "Moderada": moderado,
        "Agressiva": agressivo
    }

    pasta = "./img_testes_rapidos"  # coloque o caminho da pasta
    df = main(pasta, tabelas)

    # Geração de boxplots
    metricas = ['PSNR', 'SSIM', '% de Zeros', 'Taxa de Compressão (em x)']
    for met in metricas:
        plt.figure(figsize=(8, 5))
        df.boxplot(column=met, by="Quantização")
        plt.title(f"Boxplot - {met}")
        plt.suptitle("")
        plt.xlabel("Matriz de Quantização")
        plt.ylabel(met)
        plt.grid(True)
        plt.show()

    metricas = ['PSNR', 'SSIM', '% de Zeros']
    plt.figure(figsize=(10, 6))

    for met in metricas:
        plt.figure(figsize=(8, 5))
        for nome_tab in tabelas.keys():
            df_tab = df[df['Quantização'] == nome_tab]
            plt.scatter(df_tab['Taxa de Compressão (em x)'], 
                        df_tab[met], 
                        label=nome_tab, alpha=0.7)
        plt.title(f"{met} x Taxa de Compressão")
        plt.xlabel("Taxa de Compressão (em x)")
        plt.ylabel(met)
        plt.grid(True)
        plt.legend()
        plt.show()

    # gráfico de violino
    metricas = ['PSNR', 'SSIM', '% de Zeros', 'Taxa de Compressão (em x)']
    for met in metricas:
        plt.figure(figsize=(8,5))
        sns.violinplot(x='Quantização', y=met, data=df)
        plt.title(f"Violin Plot - {met}")
        plt.grid(True)
        plt.show()

    # heatmap
    corr = df[['PSNR','SSIM','% de Zeros','Taxa de Compressão (em x)']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Matriz de Correlação das Métricas")
    plt.show()
