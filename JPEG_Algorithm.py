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
        'Figura 8': './img/aereo.tiff',
        'Figura 9': './img/letras_formas.tiff',
        'Figura 10': './img/caca.tiff',
        'Figura 11': './img/casa.tiff', 
        'Figura 12': './img/formas_aleatorias.tiff',
        'Figura 13': './img/casa_carro.tiff',
        'Figura 14': './img/rabiscos.tiff',
        'Figura 15': './img/doces.tiff',
        'Figura 16': './img/textura.tiff',
        'Figura 17': './img/mandril.tiff',
        'Figura 18': './img/placa_carro.tiff',
        'Figura 19': './img/peppers.tiff',
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


# --- Funções auxiliares --- #
def dividir_blocos(img, size=8):
    return [img[i:i+size, j:j+size] for i in range(0, img.shape[0], size) for j in range(0, img.shape[1], size)]

def juntar_blocos(blocos, height, width, size=8):
    img = np.zeros((height, width), dtype=np.uint8)
    idx = 0
    for i in range(0, height, size):
        for j in range(0, width, size):
            img[i:i+size, j:j+size] = np.clip(blocos[idx], 0, 255)
            idx += 1
    return img

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# --- Huffman --- #
class HuffmanNode(namedtuple("HuffmanNode", ["left", "right"])):
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class HuffmanLeaf(namedtuple("HuffmanLeaf", ["symbol"])):
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

# --- Processamento por canal --- #
def processar_canal(canal, Q):
    blocos = dividir_blocos(canal)
    blocos_dct = []
    zeros_total, coef_total = 0, 0

    for bloco in blocos:
        dct_b = dct2(bloco)
        quant = np.round(dct_b / Q)
        blocos_dct.append(quant.astype(np.int16))
        zeros_total += np.sum(quant == 0)
        coef_total += quant.size

    blocos_rec = [np.round(idct2(bloco * Q)) for bloco in blocos_dct]
    img_rec = juntar_blocos(blocos_rec, *canal.shape)
    return img_rec, blocos_dct, zeros_total, coef_total

# --- Executar compressão e avaliação --- #
def processar_imagem(img, tabelas_quantizacao, nome_img):
    bmp_path = "temp_raw.bmp"
    cv2.imwrite(bmp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    tamanho_raw = os.path.getsize(bmp_path)
    os.remove(bmp_path)

    resultados, imgs_reconstruidas = {}, {}
    for nome_tab, Q in tabelas_quantizacao.items():
        canais_rec, coef_all = [], []
        zeros_total, coef_total = 0, 0

        for i in range(3):  # R, G, B
            rec, blocos_dct, z, c = processar_canal(img[:, :, i], Q)
            canais_rec.append(rec)
            coef_all.extend([b.flatten() for b in blocos_dct])
            zeros_total += z
            coef_total += c

        img_final = np.stack(canais_rec, axis=2).astype(np.uint8)
        imgs_reconstruidas[nome_tab] = img_final

        psnr = peak_signal_noise_ratio(img, img_final, data_range=255)
        ssim = structural_similarity(img, img_final, channel_axis=2)
        perc_zeros = (zeros_total / coef_total) * 100

        coef_flat = np.concatenate(coef_all)
        freq = Counter(coef_flat)
        huff = build_huffman_tree(freq)
        huff_code = {}
        huff.walk(huff_code, "")
        bitstream = "".join(huff_code[val] for val in coef_flat)
        tamanho_bits = len(bitstream)
        tamanho_bytes = tamanho_bits / 8
        taxa_huffman = tamanho_raw / tamanho_bytes

        resultados[nome_tab] = {
            'Tabela': nome_tab,
            'PSNR (dB)': round(psnr, 2),
            'SSIM': round(ssim, 4),
            '% Coef. Zerados': round(perc_zeros, 2),
            'Tamanho RAW (bytes)': tamanho_raw,
            'Comprimido (bytes)': round(tamanho_bytes),
            'Compressão Huffman (x)': round(taxa_huffman, 2)
        }

    df = pd.DataFrame(resultados.values())
    df.to_csv(f"resultados_{nome_img}.csv", index=False)
    print(f"\n🔍 Resultados - {nome_img}:\n{df.to_string(index=False)}")
    return resultados, imgs_reconstruidas

# --- Visualização --- #
def plotar_imagens_com_metricas(img_original, imagens_rec, resultados, titulo_img):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    titulos = ['Original', *[f"Reconstruída - {k}" for k in imagens_rec.keys()]]
    imagens = [img_original, *[imagens_rec[k] for k in imagens_rec]]

    for i, ax in enumerate(axs):
        ax.imshow(imagens[i])
        ax.set_title(titulos[i], fontsize=11, fontweight='bold')
        ax.axis('off')
        psnr_str = f"PSNR: {resultados.get(list(imagens_rec.keys())[i-1], {}).get('PSNR (dB)', 'inf')} dB" if i else "PSNR: inf"
        ssim_str = f"SSIM: {resultados.get(list(imagens_rec.keys())[i-1], {}).get('SSIM', 1):.2f}" if i else "SSIM: 1.00"
        ax.text(0.5, -0.08, psnr_str, ha='center', transform=ax.transAxes, fontsize=10)
        ax.text(0.5, -0.14, ssim_str, ha='center', transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3)
    plt.show()

# --- Gráficos comparativos --- #
def plotar_comparativo(df, ylabel, filename):
    x = np.arange(len(df['Imagem']))
    largura = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    b1 = ax.bar(x - largura, df['padrão'], width=largura, label='Padrão', color='#4B9CD3')
    b2 = ax.bar(x, df['moderado'], width=largura, label='Moderado', color='#F4B400')
    b3 = ax.bar(x + largura, df['agressivo'], width=largura, label='Agressivo', color='#DB4437')

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Imagem'], rotation=45, ha='right')
    ax.set_ylim(0, df.drop(columns='Imagem').values.max() + 5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=10)

    for bars in [b1, b2, b3]:
        for bar in bars:
            altura = bar.get_height()
            ax.annotate(f'{altura:.2f}', xy=(bar.get_x() + bar.get_width() / 2, altura),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def gerar_graficos_comparativos(pasta_csvs):
    arquivos = [f for f in os.listdir(pasta_csvs) if f.endswith(".csv")]
    imagens = []
    psnr_vals, ssim_vals, taxa_comp_vals = [], [], []

    for arq in arquivos:
        df = pd.read_csv(os.path.join(pasta_csvs, arq))
        nome_img = arq.replace("resultados_", "").replace(".csv", "")
        imagens.append(nome_img)

        psnr_vals.append([df[df['Tabela'] == q]['PSNR (dB)'].values[0] for q in ['padrão', 'moderado', 'agressivo']])
        ssim_vals.append([df[df['Tabela'] == q]['SSIM'].values[0] for q in ['padrão', 'moderado', 'agressivo']])
        taxa_comp_vals.append([df[df['Tabela'] == q]['Compressão Huffman (x)'].values[0] for q in ['padrão', 'moderado', 'agressivo']])

    def montar_df(matriz):
        return pd.DataFrame(matriz, columns=['padrão', 'moderado', 'agressivo']).assign(Imagem=imagens)

    plotar_comparativo(montar_df(psnr_vals), "PSNR (dB)", "comparativo_psnr.png")
    plotar_comparativo(montar_df(ssim_vals), "SSIM", "comparativo_ssim.png")


for nome_img, caminho in imagens.items():
    img = cv2.imread(caminho)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados, imgs_recon = processar_imagem(img, tabelas_quantizacao, nome_img)
    plotar_imagens_com_metricas(img, imgs_recon, resultados, nome_img)
gerar_graficos_comparativos('./')
