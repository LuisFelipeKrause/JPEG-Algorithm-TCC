# JPEG_Algorithm_Transform_fixed.py
import os
import numpy as np
import cv2
import pywt # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter, namedtuple
import heapq
from scipy.fftpack import dct, idct
from numpy.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ----------------------------
# --- Configurações / Pasta ---
# ----------------------------
pasta_img = './img_testes_rapidos'   # ajuste se necessário
arquivos = [os.path.join(pasta_img, f) for f in os.listdir(pasta_img)
            if f.lower().endswith(('.tiff', '.jpeg', '.jpg', '.png'))]

img_comparativo = [
    '1.1.11.tiff',
    'texmos1.p512.tiff',
    'n01614925_bald_eagle.JPEG',
    'n02066245_grey_whale.JPEG',
    '4.2.03.tiff',
    'gray21.512.tiff'
]

# ------------------------------
# --- Matrizes / Transformadas -
# ------------------------------
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float64)

# DCT 2D / IDCT 2D (para blocos 8x8)
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Laplace (cv2) - retorna float64
def laplace_block(block):
    return cv2.Laplacian(block.astype(np.float64), cv2.CV_64F)

# Wavelet helpers (usamos DWT level=1 + montagem 8x8)
def wavelet_to_block(block):
    # block: 8x8 -> cA(4x4), cH(4x4), cV(4x4), cD(4x4) -> montamos 8x8
    cA, (cH, cV, cD) = pywt.dwt2(block, 'haar')
    top = np.hstack((cA, cH))
    bot = np.hstack((cV, cD))
    stacked = np.vstack((top, bot))
    return stacked  # 8x8

def block_to_wavelet_coeffs(stacked):
    # recebe 8x8 empilhado e devolve (cA, (cH, cV, cD)) com shapes 4x4
    h, w = stacked.shape
    assert h % 2 == 0 and w % 2 == 0
    hh, ww = h//2, w//2
    cA = stacked[:hh, :ww]
    cH = stacked[:hh, ww:]
    cV = stacked[hh:, :ww]
    cD = stacked[hh:, ww:]
    return cA, (cH, cV, cD)

# ------------------------------
# --- Huffman (estimativa) -----
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
    if len(heap) == 0:
        raise ValueError("freq vazia")
    while len(heap) > 1:
        f1, c1, l1 = heapq.heappop(heap)
        f2, c2, l2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, count, HuffmanNode(l1, l2)))
        count += 1
    return heap[0][2]

# ---------------------------
# --- Blocos / Subamostragem -
# ---------------------------
def dividir_blocos(img, size=8):
    h, w = img.shape
    blocos = []
    for i in range(0, h, size):
        for j in range(0, w, size):
            bloco = img[i:i+size, j:j+size]
            if bloco.shape == (size, size):
                blocos.append(bloco)
    return blocos

def juntar_blocos(blocos, height, width, size=8):
    img_recon = np.zeros((height, width), dtype=np.float64)
    idx = 0
    for i in range(0, height, size):
        for j in range(0, width, size):
            img_recon[i:i+size, j:j+size] = blocos[idx]
            idx += 1
    return img_recon

def subamostragem_420(img_ycrcb):
    # OpenCV usa YCrCb em que [0]=Y, [1]=Cr, [2]=Cb
    Y = img_ycrcb[:,:,0]
    Cr = img_ycrcb[:,:,1][::2, ::2]
    Cb = img_ycrcb[:,:,2][::2, ::2]
    return Y, Cb, Cr  # retornamos na ordem Y, Cb, Cr para compatibilidade posterior

# ------------------------------
# --- Processar um canal -------
# ------------------------------
def processar_canal(canal, Q, tipo):
    height, width = canal.shape
    blocos = dividir_blocos(canal)
    blocos_q = []
    blocos_rec = []
    zeros_total = 0
    coef_total = 0

    for bloco in blocos:
        bloco = bloco.astype(np.float64)

        if tipo == 'dct':
            coeff = dct2(bloco)
            # quantização
            quant = np.round(coeff / Q)
            dequant = quant * Q
            bloco_rec = idct2(dequant)

        elif tipo == 'fourier':
            coeff_complex = fft2(bloco)
            coeff = coeff_complex.real   # usamos apenas a parte real para quantização
            quant = np.round(coeff / Q)
            dequant = quant * Q
            # reconstrução via ifft (passamos de volta um array real)
            bloco_rec = np.real(ifft2(dequant))

        elif tipo == 'laplace':
            L = laplace_block(bloco)    # pode conter negativos
            # quantizamos a laplaciana
            quant = np.round(L / Q)
            dequant = quant * Q
            # reconstruímos aproximando: bloco - L_dequant
            bloco_rec = bloco - dequant

        elif tipo == 'wavelet':
            stacked = wavelet_to_block(bloco)   # 8x8 formado por cA,cH,cV,cD (cada 4x4)
            # use apenas parte real (pywt dá float real, mas consistente)
            coeff = stacked.astype(np.float64)
            quant = np.round(coeff / Q)
            dequant = quant * Q
            # reconstruir: separar cA,cH,cV,cD e idwt2
            cA, (cH, cV, cD) = block_to_wavelet_coeffs(dequant)
            bloco_rec = pywt.idwt2((cA, (cH, cV, cD)), 'haar')

        else:
            # identidade (fallback)
            coeff = bloco
            quant = np.round(coeff / Q)
            dequant = quant * Q
            bloco_rec = dequant

        # Estatísticas
        zeros_total += np.sum(quant == 0)
        coef_total += quant.size
        blocos_q.append(quant.astype(np.int16))
        blocos_rec.append(np.round(bloco_rec))

    # junta blocos reconstruídos
    img_rec = juntar_blocos(blocos_rec, height, width)
    img_rec = np.clip(img_rec, 0, 255).astype(np.uint8)
    return img_rec, blocos_q, zeros_total, coef_total

# ------------------------------
# --- Rotina Principal -------
# ------------------------------
transformadas = ['dct', 'fourier', 'laplace', 'wavelet']
resultados_gerais = []

for arquivo in arquivos:
    # leitura e preparos
    img_bgr = cv2.imread(arquivo)
    if img_bgr is None:
        print("Não foi possível abrir", arquivo)
        continue
    img_bgr = cv2.resize(img_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    # usado para métricas / exibição
    # convert para YCrCb (OpenCV usa Y,Cr,Cb)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    h, w, _ = img_rgb.shape
    tamanho_original = h * w * 3  # bytes (RGB 8 bits por canal)

    # subamostragem Y 4:2:0 (extraimos Y, Cb e Cr)
    Y_full, Cb_sub, Cr_sub = subamostragem_420(img_ycrcb)

    for tipo in transformadas:
        # Processa cada canal (usamos a matriz padrão Q)
        Y_rec, blocos_Y, zY, cY = processar_canal(Y_full, Q, tipo)
        Cb_rec_sub, blocos_Cb, zCb, cCb = processar_canal(Cb_sub, Q, tipo)
        Cr_rec_sub, blocos_Cr, zCr, cCr = processar_canal(Cr_sub, Q, tipo)

        # Interpolação para o tamanho de Y
        Cb_rec = cv2.resize(Cb_rec_sub, (Y_full.shape[1], Y_full.shape[0]), interpolation=cv2.INTER_LINEAR)
        Cr_rec = cv2.resize(Cr_rec_sub, (Y_full.shape[1], Y_full.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Empilhar corretamente para conversão: OpenCV espera Y, Cr, Cb para COLOR_YCrCb2RGB
        img_ycrcb_rec = np.stack([Y_rec, Cr_rec, Cb_rec], axis=2).astype(np.uint8)
        img_final = cv2.cvtColor(img_ycrcb_rec, cv2.COLOR_YCrCb2RGB)

        # Métricas (garantir uint8 + data_range)
        original_uint8 = img_rgb.astype(np.uint8)
        reconstruida_uint8 = img_final.astype(np.uint8)
        psnr_val = peak_signal_noise_ratio(original_uint8, reconstruida_uint8, data_range=255)
        ssim_val = structural_similarity(original_uint8, reconstruida_uint8, channel_axis=2, data_range=255)

        perc_zeros = (zY + zCb + zCr) / (cY + cCb + cCr) * 100

        # Estimativa de tamanho comprimido por Huffman (proteções)
        try:
            coef_all = np.concatenate([
                np.concatenate([b.flatten() for b in blocos_Y]) if len(blocos_Y) else np.array([], dtype=np.int16),
                np.concatenate([b.flatten() for b in blocos_Cb]) if len(blocos_Cb) else np.array([], dtype=np.int16),
                np.concatenate([b.flatten() for b in blocos_Cr]) if len(blocos_Cr) else np.array([], dtype=np.int16)
            ])
            if coef_all.size == 0:
                tamanho_comprimido = np.nan
            else:
                freq = Counter(coef_all)
                arvore = build_huffman_tree(freq)
                huff_code = {}
                arvore.walk(huff_code, "")
                # se algum coef não estiver no huff_code (improvável), caímos no except
                bits = 0
                for c in coef_all:
                    bits += len(huff_code[int(c)])
                tamanho_comprimido = bits / 8.0  # bytes
        except Exception:
            tamanho_comprimido = np.nan

        taxa_comp = (tamanho_original / tamanho_comprimido) if (tamanho_comprimido and not np.isnan(tamanho_comprimido)) else np.nan

        resultados_gerais.append({
            'Imagem': os.path.basename(arquivo),
            'Transformada': tipo,
            'PSNR': round(psnr_val, 2),
            'SSIM': round(ssim_val, 4),
            '% Coef. Zerados': round(perc_zeros, 2),
            'Tamanho Original (bytes)': int(tamanho_original),
            'Tamanho Comprimido (bytes)': (int(round(tamanho_comprimido)) if not np.isnan(tamanho_comprimido) else np.nan),
            'Taxa Compressão (x)': (round(taxa_comp, 2) if not np.isnan(taxa_comp) else np.nan),
            'Reconstruida': img_final if os.path.basename(arquivo) in img_comparativo else None,
            'Original': img_rgb if os.path.basename(arquivo) in img_comparativo else None
        })

# -----------------------
# --- DataFrame Final ---
# -----------------------
df = pd.DataFrame(resultados_gerais)

# Remover possíveis linhas totalmente vazias e NaNs nas métricas chave
df_clean = df.copy()
# Garantir colunas numéricas onde necessário
for col in ['PSNR','SSIM','% Coef. Zerados','Taxa Compressão (x)']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# ---------------------------
# --- Comparativo Imagens ----
# ---------------------------
for img_nome in img_comparativo:
    subset = df[df['Imagem'] == img_nome]
    if subset.empty:
        print(f"{img_nome} não encontrado nos resultados (ou não existia no diretório).")
        continue
    original = subset.iloc[0]['Original']
    plt.figure(figsize=(4*(len(transformadas)+1), 4))
    plt.subplot(1, len(transformadas)+1, 1)
    plt.imshow(original)
    plt.title('Original')
    plt.axis('off')
    # ordenar por transformadas definidas
    for i, t in enumerate(transformadas, start=2):
        row = subset[subset['Transformada']==t]
        if row.empty:
            # nenhuma reconstrução para essa transformada
            plt.subplot(1, len(transformadas)+1, i)
            plt.text(0.5,0.5,f"No data for {t}", horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            continue
        img_rec = row.iloc[0]['Reconstruida']
        ps = row.iloc[0]['PSNR']
        ss = row.iloc[0]['SSIM']
        plt.subplot(1, len(transformadas)+1, i)
        plt.imshow(img_rec)
        plt.title(f"{t}\nPSNR={ps}, SSIM={ss}")
        plt.axis('off')
    plt.suptitle(f'Comparativo - {img_nome}')
    plt.show()

# ---------------------------
# --- BoxPlot por Métrica ---
# ---------------------------
metrics = ['PSNR','SSIM','% Coef. Zerados','Taxa Compressão (x)']
for metric in metrics:
    if metric not in df_clean.columns:
        continue
    df_plot = df_clean[['Transformada', metric]].dropna()
    # garantir ordem somente com as transformadas que têm dados
    order = [t for t in transformadas if not df_plot[df_plot['Transformada']==t].empty]
    if df_plot.empty or len(order) == 0:
        print(f"Sem dados válidos para boxplot de {metric}. Pulando.")
        continue
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Transformada', y=metric, data=df_plot, order=order)
    plt.title(f'Boxplot - {metric}')
    plt.show()

# ------------------------------------------
# --- Scatter plot vs Taxa de Compressão ---
# ------------------------------------------
for met in ['PSNR','SSIM','% Coef. Zerados']:
    if met not in df_clean.columns:
        continue
    df_plot = df_clean[['Transformada','Taxa Compressão (x)', met]].dropna()
    if df_plot.empty:
        print(f"Sem dados válidos para scatter {met}. Pulando.")
        continue
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='Taxa Compressão (x)', y=met, hue='Transformada', data=df_plot)
    plt.title(f"{met} x Taxa de Compressão")
    plt.xlabel("Taxa de Compressão (em x)")
    plt.ylabel(met)
    plt.grid(True)
    plt.legend()
    plt.show()

# ---------------------------
# --- Gráfico de violino ---
# ---------------------------
for met in metrics:
    if met not in df_clean.columns:
        continue
    df_plot = df_clean[['Transformada', met]].dropna()
    order = [t for t in transformadas if not df_plot[df_plot['Transformada']==t].empty]
    if df_plot.empty or len(order)==0:
        print(f"Sem dados válidos para violin plot de {met}. Pulando.")
        continue
    plt.figure(figsize=(8,5))
    sns.violinplot(x='Transformada', y=met, data=df_plot, order=order, inner='box')
    plt.title(f"Violin Plot - {met}")
    plt.grid(True)
    plt.show()
