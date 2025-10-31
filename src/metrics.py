import sys
import numpy as np
import pydicom
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse 


def load_dicom_as_float(path):
    """Lê um arquivo DICOM e normaliza o pixel array para [0,1]."""
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float64)

    # Normalização automática (para evitar overflow)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    return img

def compare_dicom(img1_path, img2_path):
    # Carregar imagens DICOM
    img1 = load_dicom_as_float(img1_path)
    img2 = load_dicom_as_float(img2_path)

    # Garantir dimensões iguais
    if img1.shape != img2.shape:
        raise ValueError(f"As imagens têm tamanhos diferentes: {img1.shape} vs {img2.shape}")

    # Calcular métricas
    mse_val = mse(img1, img2)
    psnr_val = psnr(img1, img2, data_range=1.0)
    ssim_val = ssim(img1, img2, data_range=1.0)

    print(f"\n📊 Comparação entre '{img1_path}' e '{img2_path}':\n")
    print(f"🧮 MSE:  {mse_val:.6f}")
    print(f"📈 PSNR: {psnr_val:.4f} dB")
    print(f"🔍 SSIM: {ssim_val:.6f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python src/metrics.py path/imagem1.dcm path/imagem2.dcm")
        sys.exit(1)
    compare_dicom(sys.argv[1], sys.argv[2])
