import os
import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter, namedtuple
import heapq
from scipy.fftpack import dct, idct
from numpy.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -----------------------------
# --- Configurations / Folder ---
# -----------------------------
img_folder = './img'
files = [os.path.join(img_folder, f) for f in os.listdir(img_folder)
         if f.lower().endswith(('.tiff', '.jpeg'))]

comparison_images = [
    '2.1.10.tiff',
    'gray21.512.tiff',
    'n02066245_grey_whale.JPEG',
    'n02096051_Airedale.JPEG',
]

# ------------------------------
# --- Matrices / Transforms ---
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

# DCT 2D / IDCT 2D (for 8x8 blocks)
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Laplace (cv2)
def laplace_block(block):
    return cv2.Laplacian(block.astype(np.float64), cv2.CV_64F)

# Wavelet
def wavelet_to_block(block):
    cA, (cH, cV, cD) = pywt.dwt2(block, 'haar')
    top = np.hstack((cA, cH))
    bottom = np.hstack((cV, cD))
    stacked = np.vstack((top, bottom))
    return stacked  # 8x8

def block_to_wavelet_coeffs(stacked):
    h, w = stacked.shape
    assert h % 2 == 0 and w % 2 == 0
    hh, ww = h//2, w//2
    cA = stacked[:hh, :ww]
    cH = stacked[:hh, ww:]
    cV = stacked[hh:, :ww]
    cD = stacked[hh:, ww:]
    return cA, (cH, cV, cD)

# ------------------------------
# --- Huffman Encoding ---
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
        raise ValueError("Empty frequency dictionary")
    while len(heap) > 1:
        f1, c1, l1 = heapq.heappop(heap)
        f2, c2, l2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, count, HuffmanNode(l1, l2)))
        count += 1
    return heap[0][2]

# ----------------------------
# --- Blocks / Subsampling ---
# ----------------------------
def split_blocks(img, size=8):
    h, w = img.shape
    blocks = []
    for i in range(0, h, size):
        for j in range(0, w, size):
            block = img[i:i+size, j:j+size]
            if block.shape == (size, size):
                blocks.append(block)
    return blocks

def join_blocks(blocks, height, width, size=8):
    reconstructed_img = np.zeros((height, width), dtype=np.float64)
    idx = 0
    for i in range(0, height, size):
        for j in range(0, width, size):
            reconstructed_img[i:i+size, j:j+size] = blocks[idx]
            idx += 1
    return reconstructed_img

def subsampling_420(img_ycrcb):
    Y = img_ycrcb[:,:,0]
    Cr = img_ycrcb[:,:,1][::2, ::2]
    Cb = img_ycrcb[:,:,2][::2, ::2]
    return Y, Cb, Cr

# ------------------------------
# --- Process a Channel -------
# ------------------------------
def process_channel(channel, Q, transform_type):
    height, width = channel.shape
    blocks = split_blocks(channel)
    quantized_blocks = []
    reconstructed_blocks = []
    total_zeros = 0
    total_coeffs = 0

    for block in blocks:
        block = block.astype(np.float64)

        if transform_type == 'dct':
            coeff = dct2(block)
            quant = np.round(coeff / Q)
            dequant = quant * Q
            block_rec = idct2(dequant)

        elif transform_type == 'fourier':
            coeff_complex = fft2(block)
            coeff = coeff_complex.real
            quant = np.round(coeff / Q)
            dequant = quant * Q
            block_rec = np.real(ifft2(dequant))

        elif transform_type == 'laplace':
            L = laplace_block(block)
            quant = np.round(L / Q)
            dequant = quant * Q
            block_rec = block - dequant

        elif transform_type == 'wavelet':
            stacked = wavelet_to_block(block)
            coeff = stacked.astype(np.float64)
            quant = np.round(coeff / Q)
            dequant = quant * Q
            cA, (cH, cV, cD) = block_to_wavelet_coeffs(dequant)
            block_rec = pywt.idwt2((cA, (cH, cV, cD)), 'haar')

        else:
            coeff = block
            quant = np.round(coeff / Q)
            dequant = quant * Q
            block_rec = dequant

        total_zeros += np.sum(quant == 0)
        total_coeffs += quant.size
        quantized_blocks.append(quant.astype(np.int16))
        reconstructed_blocks.append(np.round(block_rec))

    img_rec = join_blocks(reconstructed_blocks, height, width)
    img_rec = np.clip(img_rec, 0, 255).astype(np.uint8)
    return img_rec, quantized_blocks, total_zeros, total_coeffs

# ------------------------------
# --- Main Program ------------
# ------------------------------
transforms = ['dct', 'fourier', 'laplace', 'wavelet']
overall_results = []

for file in files:
    img_bgr = cv2.imread(file)
    if img_bgr is None:
        print("Could not open", file)
        continue
    img_bgr = cv2.resize(img_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    h, w, _ = img_rgb.shape
    original_size = h * w * 3

    Y_full, Cb_sub, Cr_sub = subsampling_420(img_ycrcb)

    for transform_type in transforms:
        Y_rec, blocks_Y, zY, cY = process_channel(Y_full, Q, transform_type)
        Cb_rec_sub, blocks_Cb, zCb, cCb = process_channel(Cb_sub, Q, transform_type)
        Cr_rec_sub, blocks_Cr, zCr, cCr = process_channel(Cr_sub, Q, transform_type)

        Cb_rec = cv2.resize(Cb_rec_sub, (Y_full.shape[1], Y_full.shape[0]), interpolation=cv2.INTER_LINEAR)
        Cr_rec = cv2.resize(Cr_rec_sub, (Y_full.shape[1], Y_full.shape[0]), interpolation=cv2.INTER_LINEAR)

        img_ycrcb_rec = np.stack([Y_rec, Cr_rec, Cb_rec], axis=2).astype(np.uint8)
        img_final = cv2.cvtColor(img_ycrcb_rec, cv2.COLOR_YCrCb2RGB)

        original_uint8 = img_rgb.astype(np.uint8)
        reconstructed_uint8 = img_final.astype(np.uint8)
        mse = np.mean((original_uint8.astype(np.float64) - reconstructed_uint8.astype(np.float64)) ** 2)
        psnr_val = float('inf') if mse == 0 else 10 * np.log10((255 ** 2) / mse)

        ssim_val = structural_similarity(original_uint8, reconstructed_uint8, channel_axis=2, data_range=255)

        zero_percent = (zY + zCb + zCr) / (cY + cCb + cCr) * 100

        try:
            all_coeffs = np.concatenate([
                np.concatenate([b.flatten() for b in blocks_Y]) if len(blocks_Y) else np.array([], dtype=np.int16),
                np.concatenate([b.flatten() for b in blocks_Cb]) if len(blocks_Cb) else np.array([], dtype=np.int16),
                np.concatenate([b.flatten() for b in blocks_Cr]) if len(blocks_Cr) else np.array([], dtype=np.int16)
            ])
            if all_coeffs.size == 0:
                compressed_size = np.nan
            else:
                freq = Counter(all_coeffs)
                tree = build_huffman_tree(freq)
                huff_code = {}
                tree.walk(huff_code, "")
                bits = sum(len(huff_code[int(c)]) for c in all_coeffs)
                compressed_size = bits / 8.0
        except Exception:
            compressed_size = np.nan

        compression_rate = (original_size / compressed_size) if (compressed_size and not np.isnan(compressed_size)) else np.nan

        overall_results.append({
            'Image': os.path.basename(file),
            'Transform': transform_type,
            'PSNR': round(psnr_val, 2),
            'SSIM': round(ssim_val, 4),
            '% Zero Coefficients': round(zero_percent, 2),
            'Original Size (bytes)': int(original_size),
            'Compressed Size (bytes)': (int(round(compressed_size)) if not np.isnan(compressed_size) else np.nan),
            'Compression Rate (x)': (round(compression_rate, 2) if not np.isnan(compression_rate) else np.nan),
            'Reconstructed': img_final if os.path.basename(file) in comparison_images else None,
            'Original': img_rgb if os.path.basename(file) in comparison_images else None
        })

# -----------------------
# --- Final DataFrame ---
# -----------------------
df = pd.DataFrame(overall_results)

df_clean = df.copy()
for col in ['PSNR','SSIM','% Zero Coefficients','Compression Rate (x)']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# ---------------------------
# --- Image Comparison ------
# ---------------------------
for img_name in comparison_images:
    subset = df[df['Image'] == img_name]
    if subset.empty:
        print(f"{img_name} not found in results (or missing in directory).")
        continue

    original = subset.iloc[0]['Original']
    total_imgs = len(transforms) + 1
    cols = 3
    rows = 2

    plt.figure(figsize=(16, 10))

    plt.subplot(rows, cols, 1)
    plt.imshow(original)
    plt.title('Original')
    plt.axis('off')

    for i, t in enumerate(transforms, start=2):
        row = subset[subset['Transform'] == t]
        plt.subplot(rows, cols, i)
        if row.empty:
            plt.text(0.5, 0.5, f"No data for {t}",
                     horizontalalignment='center',
                     verticalalignment='center')
            plt.axis('off')
            continue

        img_rec = row.iloc[0]['Reconstructed']
        ps = row.iloc[0]['PSNR']
        ss = row.iloc[0]['SSIM']
        plt.imshow(img_rec)
        plt.title(f"{t}\nPSNR={ps:.2f}, SSIM={ss:.4f}")
        plt.axis('off')

    plt.tight_layout()

    name_no_ext, _ = os.path.splitext(img_name)
    plt.savefig(f"./img_output/Transform/Comparison/{name_no_ext}.png", dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------
# --- BoxPlot per Metric ----
# ---------------------------
for metric in ['PSNR', 'SSIM', '% Zero Coefficients', 'Compression Rate (x)']:
    if metric not in df_clean.columns:
        continue

    df_plot = df_clean[['Transform', metric]].dropna()
    order = [t for t in transforms if not df_plot[df_plot['Transform'] == t].empty]

    if df_plot.empty or len(order) == 0:
        print(f"No valid data for boxplot of {metric}. Skipping.")
        continue

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x='Transform',
        y=metric,
        data=df_plot,
        order=order,
        color='white',
        showcaps=True,
        linewidth=1.5,
        boxprops={'edgecolor': 'black'},
        medianprops={'color': 'black'},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'},
        flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markeredgecolor': 'black'}
    )

    transform_map = {t: i for i, t in enumerate(order)}
    x_positions = np.array([transform_map[t] for t in df_plot['Transform']]) + np.random.uniform(-0.15, 0.15, size=len(df_plot))

    plt.scatter(
        x=x_positions,
        y=df_plot[metric],
        facecolors='none',
        edgecolors='black',
        s=15,
        linewidths=0.8
    )

    plt.title(f'Boxplot - {metric}', fontsize=12)
    plt.xlabel('Transform', fontsize=11)
    plt.ylabel(metric, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"./img_output/Transform/BoxPlot_{metric}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------
# --- Scatter plot vs Compression Rate -----
# ------------------------------------------
for met in ['PSNR','SSIM','% Zero Coefficients']:
    if met not in df_clean.columns:
        continue
    df_plot = df_clean[['Transform','Compression Rate (x)', met]].dropna()
    if df_plot.empty:
        print(f"No valid data for scatter {met}. Skipping.")
        continue
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='Compression Rate (x)', y=met, hue='Transform', data=df_plot)
    plt.title(f"{met} x Compression Rate")
    plt.xlabel("Compression Rate (x)")
    plt.ylabel(met)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./img_output/Transform/ScatterPlot_{met}.png", dpi=300, bbox_inches="tight")
    plt.close()
