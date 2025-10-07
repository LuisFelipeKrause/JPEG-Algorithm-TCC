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
# --- Image Folder ---
# ------------------------
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
# --- Quantization Tables ---
# ------------------------------
quantization_tables = {
    'standard': np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ]),
    'moderate': np.array([
        [40,43,45,50,60,70,90,100],
        [43,45,50,60,70,90,100,110],
        [45,50,60,70,90,100,110,128],
        [50,60,70,90,100,110,128,128],
        [60,70,90,100,110,128,128,128],
        [70,90,100,110,128,128,128,128],
        [90,100,110,128,128,128,128,128],
        [100,110,128,128,128,128,128,128]
    ]),
    'aggressive': np.array([
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
# --- Helper Functions ---
# --------------------------
def split_blocks(img, size=8):
    h, w = img.shape
    return [img[i:i+size, j:j+size] for i in range(0, h, size) for j in range(0, w, size)]

def join_blocks(blocks, height, width, size=8):
    reconstructed_img = np.zeros((height, width), dtype=np.uint8)
    idx = 0
    for i in range(0, height, size):
        for j in range(0, width, size):
            reconstructed_img[i:i+size, j:j+size] = np.clip(blocks[idx], 0, 255)
            idx += 1
    return reconstructed_img

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

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
    while len(heap) > 1:
        f1, c1, l1 = heapq.heappop(heap)
        f2, c2, l2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, count, HuffmanNode(l1, l2)))
        count += 1
    return heap[0][2]

# ---------------------------
# --- 4:2:0 Subsampling ---
# ---------------------------
def subsample_420(YCbCr):
    Y = YCbCr[:,:,0]
    Cb = YCbCr[:,:,1][::2, ::2]
    Cr = YCbCr[:,:,2][::2, ::2]
    return Y, Cb, Cr

# --------------------------------
# --- Channel Processing ---
# --------------------------------
def process_channel(channel, Q):
    blocks = split_blocks(channel)
    dct_blocks = []
    total_zeros = 0
    total_coeffs = 0
    for block in blocks:
        block_dct = dct2(block)
        quantized = np.round(block_dct / Q)
        total_zeros += np.sum(quantized == 0)
        total_coeffs += quantized.size
        dct_blocks.append(quantized.astype(np.int16))
    rec_blocks = []
    for block_q in dct_blocks:
        block_idct = idct2(block_q * Q)
        rec_blocks.append(np.round(block_idct))
    rec_img = join_blocks(rec_blocks, channel.shape[0], channel.shape[1])
    return rec_img, dct_blocks, total_zeros, total_coeffs

all_results = []

for file in files:
    img = cv2.imread(file)
    # Resize to 512x512
    img = cv2.resize(img, (512, 512))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Original size in bytes (RGB 8 bits per channel)
    h, w, c = img_rgb.shape
    original_size = h * w * c  # bytes
    
    # Subsampling 4:2:0
    Y, Cb, Cr = subsample_420(img_ycbcr)
    
    reconstructed_imgs = {}
    results = {}
    
    for name, Q in quantization_tables.items():
        # Process channels
        Y_rec, dct_blocks_Y, zY, cY = process_channel(Y, Q)
        Cb_rec_sub, dct_blocks_Cb, zCb, cCb = process_channel(Cb, Q)
        Cr_rec_sub, dct_blocks_Cr, zCr, cCr = process_channel(Cr, Q)
        
        # Upsample to original size
        Cb_rec = cv2.resize(Cb_rec_sub, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)
        Cr_rec = cv2.resize(Cr_rec_sub, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Reconstruct YCbCr and convert to RGB
        img_ycbcr_rec = np.stack([Y_rec, Cb_rec, Cr_rec], axis=2).astype(np.uint8)
        final_img = cv2.cvtColor(img_ycbcr_rec, cv2.COLOR_YCrCb2RGB)
        reconstructed_imgs[name] = final_img
        
        # Metrics
        psnr = peak_signal_noise_ratio(img_rgb, final_img, data_range=255)
        ssim = structural_similarity(img_rgb, final_img, channel_axis=2)
        perc_zeros = (zY + zCb + zCr) / (cY + cCb + cCr) * 100
        
        # Huffman
        all_coeffs = np.concatenate([np.concatenate([b.flatten() for b in dct_blocks_Y]),
                                     np.concatenate([b.flatten() for b in dct_blocks_Cb]),
                                     np.concatenate([b.flatten() for b in dct_blocks_Cr])])
        freq = Counter(all_coeffs)
        tree = build_huffman_tree(freq)
        huff_code = {}
        tree.walk(huff_code, "")
        compressed_size = len("".join(huff_code[val] for val in all_coeffs)) / 8  # bytes
        
        # Compression ratio
        compression_ratio = original_size / compressed_size if compressed_size != 0 else np.nan
        
        results[name] = {
            'Image': os.path.basename(file),
            'Table': name,
            'PSNR': round(psnr, 2),
            'SSIM': round(ssim, 4),
            '% Zeroed Coefficients': round(perc_zeros, 2),
            'Original Size (bytes)': original_size,
            'Compressed Size (bytes)': round(compressed_size),
            'Compression Ratio (x)': round(compression_ratio, 2)
        }
    
    all_results.extend(results.values())
    
    # ------------------------
    # --- Comparative Plot ---
    # ------------------------
    if os.path.basename(file) in comparison_images:
        img_name = os.path.basename(file)

        plt.figure(figsize=(10,10))

        # Original (position 1)
        plt.subplot(2,2,1)
        plt.title("Original")
        plt.imshow(img_rgb)
        plt.axis("off")

        # Three compressions (positions 2,3,4)
        for i, name in enumerate(['standard','moderate','aggressive'], start=2):
            plt.subplot(2,2,i)
            plt.title(f"{name.capitalize()}\nPSNR:{results[name]['PSNR']:.2f}  SSIM:{results[name]['SSIM']:.4f}")
            plt.imshow(reconstructed_imgs[name])
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"./img_output/Quantization/Comparison/{img_name}.png", dpi=300, bbox_inches="tight")
        plt.close()


# -----------------------
# --- Final DataFrame ---
# -----------------------
df = pd.DataFrame(all_results)

# ---------------------------
# --- BoxPlot by Metric ---
# ---------------------------
for metric in ['PSNR', 'SSIM', '% Zeroed Coefficients', 'Compression Ratio (x)']:
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x='Table',
        y=metric,
        data=df,
        color='white',
        showcaps=True,
        linewidth=1.5,
        boxprops={'edgecolor': 'black'},
        medianprops={'color': 'black'},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'},
        flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markeredgecolor': 'black'}
    )

    table_map = {'standard': 0, 'moderate': 1, 'aggressive': 2}

    x_positions = np.array([table_map[t] for t in df['Table']]) + np.random.uniform(-0.15, 0.15, size=len(df))

    plt.scatter(
        x=x_positions,
        y=df[metric],
        facecolors='none',
        edgecolors='black',
        s=15,
        linewidths=0.8
    )

    plt.title(f'Boxplot - {metric}', fontsize=12)
    plt.xlabel('Table', fontsize=11)
    plt.ylabel(metric, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"./img_output/Quantization/BoxPlot_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------------------
# --- Scatter Plot vs Compression Ratio ---
# ------------------------------------------
metrics = ['PSNR', 'SSIM', '% Zeroed Coefficients']
for met in metrics:
    plt.figure(figsize=(8,5))
    for table_name in quantization_tables.keys():
        df_tab = df[df['Table'] == table_name]
        plt.scatter(df_tab['Compression Ratio (x)'], 
                    df_tab[met], 
                    label=table_name, alpha=0.7)
    plt.title(f"{met} vs Compression Ratio")
    plt.xlabel("Compression Ratio (x)")
    plt.ylabel(met)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./img_output/Quantization/ScatterPlot_{met}.png", dpi=300, bbox_inches="tight")
    plt.close()
