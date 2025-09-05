import os
import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.fftpack import dct, idct

# ---------------------------
# Funções DCT
# ---------------------------

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def split_into_blocks(img, block_size=8):
    h, w = img.shape
    return (img.reshape(h // block_size, block_size, -1, block_size)
              .swapaxes(1, 2)
              .reshape(-1, block_size, block_size))

def merge_blocks(blocks, img_shape, block_size=8):
    h, w = img_shape
    return (np.array(blocks)
              .reshape(h // block_size, w // block_size, block_size, block_size)
              .swapaxes(1, 2)
              .reshape(h, w))

# ---------------------------
# Huffman encoding (simples)
# ---------------------------

class HuffmanNode:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class HuffmanLeaf:
    def __init__(self, symbol):
        self.symbol = symbol
    def walk(self, code, acc):
        code[self.symbol] = acc or "0"

def build_huffman_tree(freq):
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
    freq = {}
    for symbol in data:
        freq[symbol] = freq.get(symbol, 0) + 1
    if not freq:
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
    blocks = split_into_blocks(img)
    dct_blocks = [dct2(b.astype(np.float32)) for b in blocks]
    quantized_blocks = [np.round(b / q_table) for b in dct_blocks]
    return quantized_blocks

def jpeg_decompression(quantized_blocks, q_table, img_shape):
    dequantized_blocks = [b * q_table for b in quantized_blocks]
    idct_blocks = [np.round(idct2(b)).astype(np.uint8) for b in dequantized_blocks]
    img_rec = merge_blocks(idct_blocks, img_shape)
    return np.clip(img_rec, 0, 255)

def process_image(img, q_table):
    img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y, Cb, Cr = cv2.split(img_ycc)

    # --- Canal Y ---
    quant_blocks = jpeg_compression(Y, q_table)
    all_coefs = np.hstack([b.flatten() for b in quant_blocks])
    encoded_data, _ = huffman_encoding(all_coefs)
    rec_Y = jpeg_decompression(quant_blocks, q_table, Y.shape)

    # --- Canais Cb e Cr com subamostragem 4:2:0 ---
    def process_chroma(channel):
        small = cv2.resize(channel, (channel.shape[1] // 2, channel.shape[0] // 2))
        q_blocks = jpeg_compression(small, q_table)
        rec_small = jpeg_decompression(q_blocks, q_table, small.shape)
        rec_full = cv2.resize(rec_small, (channel.shape[1], channel.shape[0]))
        return rec_full

    rec_Cb = process_chroma(Cb)
    rec_Cr = process_chroma(Cr)

    img_recon_ycc = cv2.merge([rec_Y, rec_Cb, rec_Cr])
    img_final = cv2.cvtColor(img_recon_ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    psnr = peak_signal_noise_ratio(img, img_final, data_range=255)
    ssim = structural_similarity(img, img_final, channel_axis=2)

    return img_final, psnr, ssim

# ---------------------------
# Execução principal
# ---------------------------

if __name__ == "__main__":
    # Matrizes
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

    # ---- Escolha da imagem ----
    caminho_img = "./img/aereo.tiff"
    img = cv2.imread(caminho_img)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---- Processa ----
    resultados = {"Original": (img, None, None)}
    for nome, q in tabelas.items():
        rec, psnr, ssim = process_image(img, q)
        resultados[nome] = (rec, psnr, ssim)

    # ---- Plota ----
    plt.figure(figsize=(12, 6))
    for i, (nome, (im, psnr, ssim)) in enumerate(resultados.items(), 1):
        plt.subplot(1, 4, i)
        plt.imshow(im)
        plt.axis("off")
        if psnr is None:
            plt.title(f"{nome}")
        else:
            plt.title(f"{nome}\nPSNR: {psnr:.6f}\nSSIM: {ssim:.6f}")
    plt.tight_layout()
    plt.show()
