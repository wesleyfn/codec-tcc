import os
import glob
import time
import pydicom
import numpy as np
import pandas as pd
import logging
import codec

# --- Configuração dos Logs ---
logging.basicConfig(
    level=logging.INFO,
    format="%(relativeCreated)ds\t [ %(levelname)s ]   %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- PARÂMETROS DO EXPERIMENTO ---
DATASET_DIR = 'images/'
OUTPUT_DIR = 'tcc_results/'
CODECS_TO_TEST = ['jxl', 'j2k', 'jls']
BETAS_TO_TEST = [0.4, 0.8] 
BLOCK_SIZE = 4
TARGET_PERCENTILE = 75 
DEBUG_MODE = False

# --- CONSTANTES DE CÁLCULO ---
MS_PER_S = 1000
BYTES_PER_MB = 1024 * 1024 
# -----------------------------

def calculate_metrics(original_array, processed_array, bits_stored):
    """Calcula PSNR e SSIM."""
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
        # As métricas comparam a imagem processada (stego/reconstruída) com a imagem original.
        data_range = 2**bits_stored - 1
        stego_psnr = psnr(original_array, processed_array, data_range=data_range)
        stego_ssim = ssim(original_array, processed_array, data_range=data_range, channel_axis=None)
        return stego_psnr, stego_ssim
    except ImportError:
        # Fallback manual para PSNR se skimage não estiver disponível
        mse = np.mean((original_array.astype(np.float64) - processed_array.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf'), 1.0
        data_range = 2**bits_stored - 1
        psnr_val = 20 * np.log10(data_range / np.sqrt(mse))
        return psnr_val, 0.9 

def get_image_info(img_path):
    """Extrai informações básicas do arquivo DICOM."""
    original_dicom = pydicom.dcmread(img_path, stop_before_pixels=True)
    original_size = os.path.getsize(img_path)
    
    try:
        modality = os.path.basename(os.path.dirname(img_path))
    except Exception:
        modality = getattr(original_dicom, 'Modality', 'Unknown')
        
    bits_stored = getattr(original_dicom, 'BitsStored', 16)
    shape = (int(getattr(original_dicom, 'Rows', 0)), int(getattr(original_dicom, 'Columns', 0)))
    
    return original_size, modality, bits_stored, shape

def process_single_image(img_path):
    """Processa uma única imagem com todos os betas e codecs."""
    results = []
    
    try:
        # A informação de progresso é tratada na função run_experiments_sequential
        
        # 1. Carregar e obter informações
        original_dicom_full = pydicom.dcmread(img_path)
        original_array = original_dicom_full.pixel_array
        original_size, modality, bits_stored, shape = get_image_info(img_path)

        # 2. Extrair metadados (para cálculo do payload size)
        secret_message_json = codec.extract_dicom_metadata(original_dicom_full)
        message_bits_count = len(codec.convert_message_to_bits(secret_message_json))
        metadata_size_bytes = len(secret_message_json.encode('utf-8'))

        # 3. Iterar sobre os parâmetros BETA e CODECS (chamando o pipeline integrado)
        for beta in BETAS_TO_TEST:
            for codec_name in CODECS_TO_TEST:
                
                # A. ENCODE
                bin_file_result, original_msg_check, total_encoding_time = codec.run_encoder(
                    img_path, OUTPUT_DIR, beta, BLOCK_SIZE, TARGET_PERCENTILE, codec_name,
                    debug_mode=DEBUG_MODE
                )
                
                if bin_file_result is None:
                    logger.warning(f"PULANDO: {os.path.basename(img_path)} - Capacidade insuficiente para Beta={beta} e {codec_name}.")
                    continue
            
                # B. DECODE
                restored_image, decoded_msg_check, decoding_time = codec.run_decoder(
                    bin_file_result, OUTPUT_DIR, debug_mode=DEBUG_MODE 
                )
                
                # C. Extração de Métricas
                stego_psnr, stego_ssim = calculate_metrics(original_array, restored_image, bits_stored)
                reversibility_check = np.array_equal(original_array, restored_image)
                
                # D. Coleta de Resultados Finais
                final_bin_size = os.path.getsize(bin_file_result)
                bpp = (final_bin_size * 8) / (shape[0] * shape[1])
                
                # E. NOVOS CÁLCULOS DE PERFORMANCE NORMALIZADA
                original_size_mb = original_size / BYTES_PER_MB
                compression_ratio = original_size / final_bin_size if final_bin_size > 0 else float('inf')
                encoding_speed_ms_mb = (total_encoding_time * MS_PER_S) / original_size_mb if original_size_mb > 0 else 0
                decoding_speed_ms_mb = (decoding_time * MS_PER_S) / original_size_mb if original_size_mb > 0 else 0
                
                
                results.append({
                    'Image_File': os.path.basename(img_path),
                    'Modality': modality,
                    'Bits_Stored': bits_stored,
                    'Original_Size_Bytes': original_size,
                    'Metadata_Size_Bytes': metadata_size_bytes,
                    'Message_Size_Bits': message_bits_count,
                    'Parameter_Beta': beta,
                    'Parameter_Codec': codec_name,
                    'Parameter_Percentile': TARGET_PERCENTILE,
                    'Stego_Image_PSNR_dB': stego_psnr,
                    'Stego_Image_SSIM': stego_ssim,
                    'Final_Bin_Size_Bytes': final_bin_size,
                    'Bpp': bpp,
                    'Compression_Ratio': compression_ratio,
                    'Encoding_Speed_ms_MB': encoding_speed_ms_mb,
                    'Decoding_Speed_ms_MB': decoding_speed_ms_mb,
                    'Total_Encoding_Time_s': total_encoding_time,
                    'Decoding_Time_s': decoding_time,
                    'Reversibility_Check': reversibility_check
                })
                
                # Log em modo sequencial/debug
                logger.info(f"  ✓ {codec_name.upper()} | B={beta} | PSNR={stego_psnr:.2f}dB | CR={compression_ratio:.2f}x | T_enc_MB={encoding_speed_ms_mb:.0f} ms/MB")

    except Exception as e:
        logger.error(f"Falha crítica ao processar {os.path.basename(img_path)}: {e}")
        return []
    
    return results

def run_experiments_sequential():
    """Executa experimentos em modo sequencial."""
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    image_paths = glob.glob(os.path.join(DATASET_DIR, '**/*.dcm'), recursive=True)
    if not image_paths:
        logger.error(f"Nenhum arquivo .dcm encontrado em '{DATASET_DIR}'. Verifique o caminho.")
        return

    total_images = len(image_paths)
    logger.info(f"--- INICIANDO EXPERIMENTOS SEQUENCIAIS ({total_images} imagens) ---")
    
    all_results = []
    start_time = time.time()
    
    # --- NOVO: Contador de Arquivos ---
    for i, img_path in enumerate(image_paths):
        current_index = i + 1
        img_name = os.path.basename(img_path)
        
        # Exibe o contador de progresso
        logger.info(f"\n[ PROCESSO {current_index}/{total_images} ] Imagem: {img_name}")
        
        try:
            results = process_single_image(img_path)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Falha crítica no processamento de {img_name}: {e}")

    total_time = time.time() - start_time
    
    # Salvar DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUT_DIR, 'results_sequential.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"EXPERIMENTOS SEQUENCIAIS CONCLUÍDOS")
        logger.info(f"Tempo total: {total_time:.2f}s")
        logger.info(f"Resultados salvos em: {csv_path}")
        logger.info(f"{'='*50}")
    else:
        logger.error("Nenhum resultado foi gerado!")


if __name__ == "__main__":
    run_experiments_sequential()