# main_tcc.py
# (Salve na mesma pasta que codec.py)

import os
import glob
import time
import pydicom
import numpy as np
import pandas as pd
import logging
import codec  # Importa seu script como uma biblioteca
import zlib
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# --- Configuração dos Logs ---
logging.basicConfig(
    level=logging.INFO,
    format="%(relativeCreated)dms\t [ %(levelname)s ]   %(message)s",
    handlers=[
        logging.StreamHandler()  # Imprime logs no console
    ]
)
logger = logging.getLogger(__name__)

# --- PARÂMETROS DO EXPERIMENTO ---
DATASET_DIR = 'images/'
OUTPUT_DIR = 'tcc_results/'
CODECS_TO_TEST = ['jxl', 'j2k', 'jls', 'png']  # Codecs para comparar
BETAS_TO_TEST = [0.5, 0.8]                     # Valores de Beta para variar
BLOCK_SIZE = 4
THRESHOLD_FACTOR = 0.8
# -----------------------------------

def get_image_info(img_path):
    """Extrai informações básicas do arquivo DICOM."""
    original_dicom = pydicom.dcmread(img_path, stop_before_pixels=True)
    original_size = os.path.getsize(img_path)
    
    # Tenta extrair a modalidade do nome da pasta pai
    try:
        modality = os.path.basename(os.path.dirname(img_path))
    except Exception:
        modality = getattr(original_dicom, 'Modality', 'Unknown')
        
    bits_stored = getattr(original_dicom, 'BitsStored', 16)
    shape = (int(getattr(original_dicom, 'Rows', 0)), int(getattr(original_dicom, 'Columns', 0)))
    
    return original_size, modality, bits_stored, shape

def run_experiments():
    """Itera sobre imagens, betas e codecs para coletar dados."""
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    image_paths = glob.glob(os.path.join(DATASET_DIR, '**/*.dcm'), recursive=True)
    if not image_paths:
        logger.error(f"Nenhum arquivo .dcm encontrado em '{DATASET_DIR}'. Verifique o caminho.")
        return

    logger.info(f"--- INICIANDO EXPERIMENTOS ---")
    logger.info(f"Imagens encontradas: {len(image_paths)}")
    logger.info(f"Parâmetros Beta: {BETAS_TO_TEST}")
    logger.info(f"Codecs: {CODECS_TO_TEST}")
    logger.info(f"Saída em: {OUTPUT_DIR}")
    
    results_list = []

    for img_path in image_paths:
        try:
            logger.info(f"\n{'='*50}\nProcessando Imagem: {img_path}\n{'='*50}")
            
            # 1. Carregar dados da imagem
            original_dicom_full = pydicom.dcmread(img_path)
            original_array = original_dicom_full.pixel_array
            original_size, modality, bits_stored, shape = get_image_info(img_path)
            
            # 2. Extrair metadados (a mensagem secreta)
            secret_message_json = codec.extract_dicom_metadata(original_dicom_full)
            message_bits = codec.convert_message_to_bits(secret_message_json)
            metadata_size_bytes = len(secret_message_json.encode('utf-8'))

            # 3. Iterar sobre os parâmetros BETA
            for beta in BETAS_TO_TEST:
                logger.info(f"\n--- Testando Beta = {beta} ---")
                
                # 4. Análise de Capacidade e Decomposição (depende de beta e dos parâmetros fixos)
                global_planes, local_planes = codec.decompose_image_adaptively(original_array, beta=beta)
                capacity_map, allowed_indices = codec.create_embedding_capacity_map(
                    original_array, block_size=BLOCK_SIZE, threshold_factor=THRESHOLD_FACTOR
                )
                stego_capacity_bits = len(allowed_indices)

                if len(message_bits) > stego_capacity_bits:
                    logger.warning(f"Metadados ({len(message_bits)} bits) excedem a capacidade da imagem ({stego_capacity_bits} bits). Pulando...")
                    continue
                    
                # 5. Embutimento da Esteganografia (imagem intermediária)
                stego_planes, embedding_map, _, segments_lengths, segment_indices = codec.embed_message_in_planes(
                    local_planes, message_bits, allowed_indices, original_array.shape, 
                    start_offset=0, align_across_planes=False
                )
                stego_image_array = codec.merge_bit_planes(global_planes, stego_planes)
                
                # 6. Calcular métricas da imagem intermediária (PSNR/SSIM)
                data_range = original_array.max() - original_array.min()
                stego_psnr = psnr(original_array, stego_image_array, data_range=data_range)
                stego_ssim = ssim(original_array, stego_image_array, data_range=data_range)

                # 7. Iterar sobre os CODECS
                for codec_name in CODECS_TO_TEST:
                    logger.info(f"--- Testando Codec = {codec_name.upper()} ---")
                    
                    base_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_beta{beta}_codec{codec_name}"
                    bin_path = os.path.join(OUTPUT_DIR, f"{base_filename}.bin")
                    
                    # 8. Medir Codificação (Compressão + Empacotamento)
                    start_enc = time.time()
                    
                    compressed_bytes = codec.compress_image_data(stego_image_array, codec_name)
                    bitmaps_blob = zlib.compress(embedding_map.tobytes(), level=9)
                    
                    header = codec.create_steganography_header(
                        codec=codec_name, s=len(local_planes), segments_lengths=segments_lengths,
                        segments_indices=segment_indices, bitmaps_blob_size=len(bitmaps_blob),
                        width=original_array.shape[1], height=original_array.shape[0], start_offset=0,
                        align_across_planes=False, block_size=BLOCK_SIZE, threshold_factor=THRESHOLD_FACTOR
                    )
                    
                    final_bin_size = codec.create_steganography_container(bin_path, header, bitmaps_blob, compressed_bytes)
                    
                    end_enc = time.time()
                    encoding_time = end_enc - start_enc

                    # 9. Medir Decodificação (Desempacotamento + Descompressão + Extração)
                    start_dec = time.time()
                    
                    restored_dicom, _, restored_image = codec.decode_steganography_container(
                        bin_path, 
                        output_prefix=os.path.join(OUTPUT_DIR, f"{base_filename}_decoded")
                    )
                    
                    end_dec = time.time()
                    decoding_time = end_dec - start_dec
                    
                    # 10. Verificação Final (Reversibilidade)
                    reversibility_check = np.array_equal(original_array, restored_image)
                    
                    # 11. Calcular Métricas Finais
                    cr = original_size / final_bin_size
                    bpp = (final_bin_size * 8) / (original_array.shape[0] * original_array.shape[1])

                    # 12. Salvar resultados
                    results_list.append({
                        'Image_File': os.path.basename(img_path),
                        'Modality': modality,
                        'Bits_Stored': bits_stored,
                        'Shape': f"{shape[0]}x{shape[1]}",
                        'Original_Size_Bytes': original_size,
                        'Metadata_Size_Bytes': metadata_size_bytes,
                        'Parameter_Beta': beta,
                        'Parameter_Codec': codec_name,
                        'Stego_Capacity_Bits': stego_capacity_bits,
                        'Stego_Image_PSNR_dB': stego_psnr,
                        'Stego_Image_SSIM': stego_ssim,
                        'Final_Bin_Size_Bytes': final_bin_size,
                        'CR': cr,
                        'Bpp': bpp,
                        'Encoding_Time_s': encoding_time,
                        'Decoding_Time_s': decoding_time,
                        'Reversibility_Check': reversibility_check
                    })
                    
                    logger.info(f"Resultados (Codec: {codec_name.upper()}, Beta: {beta}): CR={cr:.2f}, Bpp={bpp:.3f}, EncTime={encoding_time:.3f}s, DecTime={decoding_time:.3f}s, Reversível={reversibility_check}")

        except Exception as e:
            logger.error(f"Falha ao processar {img_path}: {e}", exc_info=True)
            
    # --- Fim dos Loops ---
    
    # 13. Salvar DataFrame
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(OUTPUT_DIR, 'results.csv')
    df.to_csv(csv_path, index=False)
    
    logger.info(f"\n{'='*50}\nEXPERIMENTOS CONCLUÍDOS\nResultados salvos em: {csv_path}\n{'='*50}")

if __name__ == "__main__":
    run_experiments()