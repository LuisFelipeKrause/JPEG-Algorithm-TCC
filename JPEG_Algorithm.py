import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from collections import Counter, namedtuple
import heapq
import os
import pandas as pd


# --- Imagens para Teste --- #
imagens = {
        'textura': './img/textura.tiff',
        'placa_carro': './img/placa_carro.tiff',
        'formas_aleatorias': './img/formas_aleatorias.tiff',
        'letras_formas': './img/letras_formas.tiff',
        'casa': './img/casa.tiff', 
        'aereo': './img/aereo.tiff', 
        'peppers': './img/peppers.tiff',
        'casa_carro': './img/casa_carro.tiff',
        'caca': './img/caca.tiff',
        'doces': './img/doces.tiff',
        'mandril': './img/mandril.tiff',
}

# --- Tabelas de quantização ---
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

# --- Funções auxiliares ---
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

# --- Codificação de Huffman ---
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


# --- Executar para todas as imagens --- #
for nome_img, caminho in imagens.items():
    print(f"\n🔍 Processando imagem: {nome_img}")

    img = cv2.imread(caminho)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape

    bmp_path = "temp_raw.bmp"
    cv2.imwrite(bmp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    tamanho_original_raw = os.path.getsize(bmp_path)
    os.remove(bmp_path)

    resultados = {}
    imgs_reconstruidas = {}

    for nome_tab, Q in tabelas_quantizacao.items():
        canais_rec = []
        coef_all = []
        zeros_total = 0
        coef_total = 0

        for i in range(3):  # R, G, B
            canal = img[:, :, i]
            rec, blocos_dct, z, c = processar_canal(canal, Q)
            canais_rec.append(rec)
            coef_all.extend([b.flatten() for b in blocos_dct])
            zeros_total += z
            coef_total += c

        img_final = np.stack(canais_rec, axis=2).astype(np.uint8)
        imgs_reconstruidas[nome_tab] = img_final

        # --- Métricas ---
        psnr = peak_signal_noise_ratio(img, img_final, data_range=255)
        ssim = structural_similarity(img, img_final, channel_axis=2)
        perc_zeros = (zeros_total / coef_total) * 100

        # --- Huffman ---
        coef_flat = np.concatenate(coef_all)
        freq = Counter(coef_flat)
        arvore = build_huffman_tree(freq)
        huff_code = {}
        arvore.walk(huff_code, "")
        bitstream = "".join(huff_code[val] for val in coef_flat)
        tamanho_bits = len(bitstream)
        tamanho_bytes = tamanho_bits / 8
        taxa_huffman = tamanho_original_raw / tamanho_bytes

        resultados[nome_tab] = {
            'Tabela': nome_tab,
            ' PSNR (dB)': round(psnr, 2),
            '   SSIM': round(ssim, 4),
            '  % Coef. Zerados': round(perc_zeros, 2),
            ' Tamanho RAW (bytes)': tamanho_original_raw,
            ' Comprimido (bytes)': round(tamanho_bytes),
            ' Compressão Huffman (x)': round(taxa_huffman, 2)
        }

    # --- Tabela final ---
    df = pd.DataFrame(resultados.values())
    print(df.to_string(index=False))

    # --- Plotar imagens 2x2 ---
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[0].axis("off")

    for i, nome_tab in enumerate(['padrão', 'moderado', 'agressivo'], start=1):
        axs[i].imshow(imgs_reconstruidas[nome_tab])
        axs[i].set_title(f"Reconstruída - {nome_tab}")
        axs[i].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
