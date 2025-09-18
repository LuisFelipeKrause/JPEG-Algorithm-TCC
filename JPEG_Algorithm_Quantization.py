import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from collections import Counter, namedtuple
import heapq
import os
import pandas as pd
import seaborn as sns

# ------------------------
# --- Pasta de imagens ---
# ------------------------
pasta_img = './img_testes_rapidos'
arquivos = [os.path.join(pasta_img, f) for f in os.listdir(pasta_img) 
            if f.lower().endswith(('.tiff', '.jpeg'))]

img_comparativo = [
    '2.1.10.tiff',
    'gray21.512.tiff',
    'n02066245_grey_whale.JPEG',
    'n02096051_Airedale.JPEG',
    ]

# ------------------------------
# --- Tabelas de Quantização ---
# ------------------------------
tabelas_quantizacao = {
    'padrão': np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ]),
    'moderado': np.array([
        [40,43,45,50,60,70,90,100],
        [43,45,50,60,70,90,100,110],
        [45,50,60,70,90,100,110,128],
        [50,60,70,90,100,110,128,128],
        [60,70,90,100,110,128,128,128],
        [70,90,100,110,128,128,128,128],
        [90,100,110,128,128,128,128,128],
        [100,110,128,128,128,128,128,128]
    ]),
    'agressivo': np.array([
        [80,85,90,100,120,140,180,200],
        [85,90,100,120,140,180,200,220],
        [90,100,120,140,180,200,220,255],
        [100,120,140,180,200,220,255,255],
        [120,140,180,200,220,255,255,255],
        [140,180,200,220,255,255,255,255],
        [180,200,220,255,255,255,255,255],
        [200,220,255,255,255,255,255,255]
    ])
}

# --------------------------
# --- Funções Auxiliares ---
# --------------------------
def dividir_blocos(img, size=8):
    h, w = img.shape
    return [img[i:i+size, j:j+size] for i in range(0, h, size) for j in range(0, w, size)]

def juntar_blocos(blocos, height, width, size=8):
    img_reconstruida = np.zeros((height, width), dtype=np.uint8)
    idx = 0
    for i in range(0, height, size):
        for j in range(0, width, size):
            img_reconstruida[i:i+size, j:j+size] = np.clip(blocos[idx], 0, 255)
            idx += 1
    return img_reconstruida

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# ------------------------------
# --- Codificação de Huffman ---
# ------------------------------
class HuffmanNode(namedtuple("HuffmanNode", ["left", "right"])):
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class HuffmanLeaf(namedtuple("HuffmanLeaf", ["symbol"])):
    def walk(self, code, acc):
        code[self.symbol] = acc or "0"

def build_huffman_tree(freq):
    heap = []
    count = 0
    for symbol, f in freq.items():
        heapq.heappush(heap, (f, count, HuffmanLeaf(int(symbol))))
        count += 1
    while len(heap) > 1:
        f1, c1, l1 = heapq.heappop(heap)
        f2, c2, l2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, count, HuffmanNode(l1, l2)))
        count += 1
    return heap[0][2]

# ---------------------------
# --- Subamostragem 4:@:0 ---
# ---------------------------
def subamostragem_420(YCbCr):
    Y = YCbCr[:,:,0]
    Cb = YCbCr[:,:,1][::2, ::2]
    Cr = YCbCr[:,:,2][::2, ::2]
    return Y, Cb, Cr

# --------------------------------
# --- Processamento dos Canais ---
# --------------------------------
def processar_canal(canal, Q):
    blocos = dividir_blocos(canal)
    blocos_dct = []
    zeros_total = 0
    coef_total = 0
    for bloco in blocos:
        bloco_dct = dct2(bloco)
        quantizado = np.round(bloco_dct / Q)
        zeros_total += np.sum(quantizado == 0)
        coef_total += quantizado.size
        blocos_dct.append(quantizado.astype(np.int16))
    blocos_rec = []
    for bloco_q in blocos_dct:
        bloco_idct = idct2(bloco_q * Q)
        blocos_rec.append(np.round(bloco_idct))
    img_rec = juntar_blocos(blocos_rec, canal.shape[0], canal.shape[1])
    return img_rec, blocos_dct, zeros_total, coef_total

resultados_gerais = []

for arquivo in arquivos:
    img = cv2.imread(arquivo)
    # Redimensiona para 512x512
    img = cv2.resize(img, (512, 512))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Tamanho original em bytes (RGB 8 bits por canal)
    h, w, c = img_rgb.shape
    tamanho_original = h * w * c  # bytes
    
    # Subamostragem 4:2:0
    Y, Cb, Cr = subamostragem_420(img_ycbcr)
    
    imgs_reconstruidas = {}
    resultados = {}
    
    for nome, Q in tabelas_quantizacao.items():
        # Processa canais
        Y_rec, blocos_dct_Y, zY, cY = processar_canal(Y, Q)
        Cb_rec_sub, blocos_dct_Cb, zCb, cCb = processar_canal(Cb, Q)
        Cr_rec_sub, blocos_dct_Cr, zCr, cCr = processar_canal(Cr, Q)
        
        # Interpolação para tamanho original
        Cb_rec = cv2.resize(Cb_rec_sub, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)
        Cr_rec = cv2.resize(Cr_rec_sub, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Reconstrói YCbCr e converte para RGB
        img_ycbcr_rec = np.stack([Y_rec, Cb_rec, Cr_rec], axis=2).astype(np.uint8)
        img_final = cv2.cvtColor(img_ycbcr_rec, cv2.COLOR_YCrCb2RGB)
        imgs_reconstruidas[nome] = img_final
        
        # Métricas
        psnr = peak_signal_noise_ratio(img_rgb, img_final, data_range=255)
        ssim = structural_similarity(img_rgb, img_final, channel_axis=2)
        perc_zeros = (zY + zCb + zCr) / (cY + cCb + cCr) * 100
        
        # Huffman
        coef_all = np.concatenate([np.concatenate([b.flatten() for b in blocos_dct_Y]),
                                   np.concatenate([b.flatten() for b in blocos_dct_Cb]),
                                   np.concatenate([b.flatten() for b in blocos_dct_Cr])])
        freq = Counter(coef_all)
        arvore = build_huffman_tree(freq)
        huff_code = {}
        arvore.walk(huff_code, "")
        tamanho_comprimido = len("".join(huff_code[val] for val in coef_all)) / 8  # bytes
        
        # Taxa de compressão
        taxa_comp = tamanho_original / tamanho_comprimido if tamanho_comprimido != 0 else np.nan
        
        resultados[nome] = {
            'Imagem': os.path.basename(arquivo),
            'Tabela': nome,
            'PSNR': round(psnr, 2),
            'SSIM': round(ssim, 4),
            '% Coef. Zerados': round(perc_zeros, 2),
            'Tamanho Original (bytes)': tamanho_original,
            'Tamanho Comprimido (bytes)': round(tamanho_comprimido),
            'Taxa Compressão (x)': round(taxa_comp, 2)
        }
    
    resultados_gerais.extend(resultados.values())
    
    # ------------------------
    # --- Plot Comparativo ---
    # ------------------------
    if os.path.basename(arquivo) in img_comparativo:
        nome_img = os.path.basename(arquivo)

        plt.figure(figsize=(10,10))

        # Original (posição 1)
        plt.subplot(2,2,1)
        plt.title("Original")
        plt.imshow(img_rgb)
        plt.axis("off")

        # Três compressões (posições 2,3,4)
        for i, nome in enumerate(['padrão','moderado','agressivo'], start=2):
            plt.subplot(2,2,i)
            plt.title(f"{nome}\nPSNR:{resultados[nome]['PSNR']:.2f}  SSIM:{resultados[nome]['SSIM']:.4f}")
            plt.imshow(imgs_reconstruidas[nome])
            plt.axis("off")

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"./img_saida/Quantization/Comparativo/{nome_img}.png", dpi=300, bbox_inches="tight")
        plt.close()


# -----------------------
# --- DataFrame Final ---
# -----------------------
df = pd.DataFrame(resultados_gerais)

# ---------------------------
# --- BoxPlot por Métrica ---
# ---------------------------
for metric in ['PSNR','SSIM','% Coef. Zerados','Taxa Compressão (x)']:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Tabela', y=metric, data=df)
    plt.title(f'Boxplot - {metric}')
    # plt.show()
    plt.savefig(f"./img_saida/Quantization/BoxPlot_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------
# --- Scatter plot vs Taxa de Compressão ---
# ------------------------------------------
metricas = ['PSNR', 'SSIM', '% Coef. Zerados']
for met in metricas:
    plt.figure(figsize=(8,5))
    for nome_tab in tabelas_quantizacao.keys():
        df_tab = df[df['Tabela'] == nome_tab]
        plt.scatter(df_tab['Taxa Compressão (x)'], 
                    df_tab[met], 
                    label=nome_tab, alpha=0.7)
    plt.title(f"{met} x Taxa de Compressão")
    plt.xlabel("Taxa de Compressão (em x)")
    plt.ylabel(met)
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(f"./img_saida/Quantization/ScatterPlot_{met}.png", dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------
# --- Gráfico de violino ---
# ---------------------------
metricas = ['PSNR', 'SSIM', '% Coef. Zerados', 'Taxa Compressão (x)']
for met in metricas:
    plt.figure(figsize=(8,5))
    sns.violinplot(x='Tabela', y=met, data=df)
    plt.title(f"Violin Plot - {met}")
    plt.grid(True)
    # plt.show()
    plt.savefig(f"./img_saida/Quantization/ViolinPlot_{met}.png", dpi=300, bbox_inches="tight")
    plt.close()
