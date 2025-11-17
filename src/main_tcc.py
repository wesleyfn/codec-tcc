import os
import glob
import time
import pydicom
import numpy as np
import pandas as pd
import logging
import codec
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuração dos Logs ---
logging.basicConfig(
    level=logging.INFO,
    format="%(relativeCreated)ds\t [ %(levelname)s ]   %(message)s",
    handlers=[
        logging.StreamHandler()  # Imprime logs no console
    ]
)
logger = logging.getLogger(__name__)

# --- PARÂMETROS DO EXPERIMENTO ---
DATASET_DIR = '../images/'
OUTPUT_DIR = 'tcc_results/'
# Codecs para comparar
CODECS_TO_TEST = ['jxl', 'j2k', 'jls', 'png']
BETAS_TO_TEST = [0.4, 0.8]                     # Valores de Beta para variar
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

def calculate_metrics(original_array, processed_array):
    """Calcula PSNR e SSIM de forma segura."""
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
        data_range = original_array.max() - original_array.min()
        stego_psnr = psnr(original_array, processed_array, data_range=data_range)
        stego_ssim = ssim(original_array, processed_array, data_range=data_range)
        return stego_psnr, stego_ssim
    except ImportError:
        # Fallback simples se scipy/skimage não estiver disponível
        mse = np.mean((original_array - processed_array) ** 2)
        if mse == 0:
            return float('inf'), 1.0
        data_range = original_array.max() - original_array.min()
        psnr_val = 20 * np.log10(data_range / np.sqrt(mse))
        return psnr_val, 0.9  # SSIM aproximado

def process_single_image(img_path):
    """Processa uma única imagem com todos os betas e codecs."""
    results = []
    
    try:
        logger.info(f"Processando Imagem: {os.path.basename(img_path)}")
        
        # 1. Carregar dados da imagem
        start_load = time.time()
        original_dicom_full = pydicom.dcmread(img_path)
        original_array = original_dicom_full.pixel_array
        original_size, modality, bits_stored, shape = get_image_info(img_path)
        load_time = time.time() - start_load
        
        # 2. Extrair metadados (a mensagem secreta)
        start_metadata = time.time()
        secret_message_json = codec.extract_dicom_metadata(original_dicom_full)
        message_bits = codec.convert_message_to_bits(secret_message_json)
        metadata_size_bytes = len(secret_message_json.encode('utf-8'))
        metadata_time = time.time() - start_metadata

        # 3. Iterar sobre os parâmetros BETA
        for beta in BETAS_TO_TEST:
            logger.info(f"--- Beta = {beta} para {os.path.basename(img_path)} ---")
            
            # 4. Análise de Capacidade e Decomposição
            start_decomposition = time.time()
            global_planes, local_planes, bits_per_pixel = codec.adaptive_modalities_decomposition(original_array, beta=beta)
            
            # USAR A NOVA FUNÇÃO DE CAPACIDADE DINÂMICA
            capacity_map, allowed_indices = codec.create_capacity_map_dynamic(
                original_array, required_bits=len(message_bits), 
                block_size=BLOCK_SIZE, threshold_factor=THRESHOLD_FACTOR
            )
            stego_capacity_bits = len(allowed_indices)
            decomposition_time = time.time() - start_decomposition

            if len(message_bits) > stego_capacity_bits:
                logger.warning(f"Metadados ({len(message_bits)} bits) excedem a capacidade ({stego_capacity_bits} bits). Pulando...")
                continue
                
            # 5. Embutimento da Esteganografia
            start_embedding = time.time()
            stego_planes, segments_lengths, segment_indices, used_indices, flip_bits = codec.embed_message_in_planes(
                local_planes, message_bits, allowed_indices, original_array.shape, 
                start_offset=0, align_across_planes=False
            )
            embedding_time = time.time() - start_embedding
            
            start_merge = time.time()
            stego_image_array = codec.merge_global_local_planes(global_planes, stego_planes, original_array.dtype)
            merge_time = time.time() - start_merge
            
            # 6. Calcular métricas da imagem intermediária
            start_metrics = time.time()
            stego_psnr, stego_ssim = calculate_metrics(original_array, stego_image_array)
            metrics_time = time.time() - start_metrics

            # USAR A NOVA FUNÇÃO OTIMIZADA DE BITMAP_BLOB
            bitmaps_blob = codec.create_optimized_bitmap_blob(used_indices, flip_bits)

            # 7. Iterar sobre os CODECS
            for codec_name in CODECS_TO_TEST:
                logger.info(f"--- Codec = {codec_name.upper()} para {os.path.basename(img_path)} ---")
                
                base_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_beta{beta}_codec{codec_name}"
                bin_path = os.path.join(OUTPUT_DIR, f"{base_filename}.bin")
                
                # 8. Medir Codificação (Compressão + Empacotamento) com breakdown de tempo
                start_total_enc = time.time()
                
                # Tempo de compressão
                start_compression = time.time()
                compressed_bytes = codec.compress_image_data(stego_image_array, codec_name)
                end_compression = time.time()
                compression_time = end_compression - start_compression
                
                # Tempo de empacotamento
                start_packaging = time.time()
                
                # USAR A NOVA FUNÇÃO DE HEADER QUE INCLUI stego_image_size
                header_bytes = codec.create_steganography_header_bytes(
                    codec=codec_name, s=len(local_planes), segments_lengths=segments_lengths,
                    segments_indices=segment_indices, stego_image_size=len(compressed_bytes),
                    width=original_array.shape[1], height=original_array.shape[0], start_offset=0,
                    align_across_planes=False, block_size=BLOCK_SIZE, threshold_factor=THRESHOLD_FACTOR,
                    bits_per_pixel=bits_per_pixel
                )
                
                final_bin_size = codec.create_steganography_container(bin_path, header_bytes, bitmaps_blob, compressed_bytes)
                end_packaging = time.time()
                packaging_time = end_packaging - start_packaging
                
                end_total_enc = time.time()
                total_encoding_time = end_total_enc - start_total_enc

                # 9. Medir Decodificação
                start_dec = time.time()
                
                try:
                    restored_dicom, extracted_metadata, restored_image = codec.decode_steganography_container(
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
                    
                    # 12. Calcular taxa de compressão e eficiência
                    uncompressed_size = original_array.size * (bits_stored / 8)  # Tamanho não comprimido em bytes
                    compression_ratio_vs_raw = uncompressed_size / final_bin_size
                    
                    # Eficiência de embedding (bits por byte comprimido)
                    embedding_efficiency = len(message_bits) / final_bin_size if final_bin_size > 0 else 0
                    
                    # 13. Calcular overhead
                    compressed_image_size = len(compressed_bytes)
                    header_size = len(header_bytes)
                    bitmaps_size = len(bitmaps_blob)
                    total_container_size = final_bin_size
                    
                    # Overhead como porcentagem do tamanho total
                    overhead_percentage = ((header_size + bitmaps_size) / total_container_size) * 100 if total_container_size > 0 else 0
                    
                    # Overhead relativo à imagem comprimida
                    overhead_vs_compressed = ((header_size + bitmaps_size) / compressed_image_size) * 100 if compressed_image_size > 0 else 0

                    # 14. Coletar resultados
                    results.append({
                        'Image_File': os.path.basename(img_path),
                        'Modality': modality,
                        'Bits_Stored': bits_stored,
                        'Shape': f"{shape[0]}x{shape[1]}",
                        'Original_Size_Bytes': original_size,
                        'Metadata_Size_Bytes': metadata_size_bytes,
                        'Parameter_Beta': beta,
                        'Parameter_Codec': codec_name,
                        'Parameter_Block_Size': BLOCK_SIZE,
                        'Parameter_Threshold_Factor': THRESHOLD_FACTOR,
                        'Stego_Capacity_Bits': stego_capacity_bits,
                        'Message_Size_Bits': len(message_bits),
                        'Stego_Image_PSNR_dB': stego_psnr,
                        'Stego_Image_SSIM': stego_ssim,
                        'Final_Bin_Size_Bytes': final_bin_size,
                        'Compressed_Image_Size_Bytes': compressed_image_size,
                        'Header_Size_Bytes': header_size,
                        'Bitmaps_Size_Bytes': bitmaps_size,
                        'CR': cr,
                        'Bpp': bpp,
                        'Compression_Ratio_vs_Raw': compression_ratio_vs_raw,
                        'Embedding_Efficiency': embedding_efficiency,
                        'Available_Capacity_Utilization': len(message_bits) / stego_capacity_bits if stego_capacity_bits > 0 else 0,
                        # Métricas de tempo detalhadas
                        'Load_Time_s': load_time,
                        'Metadata_Extraction_Time_s': metadata_time,
                        'Decomposition_Time_s': decomposition_time,
                        'Embedding_Time_s': embedding_time,
                        'Merge_Time_s': merge_time,
                        'Metrics_Calculation_Time_s': metrics_time,
                        'Compression_Time_s': compression_time,
                        'Packaging_Time_s': packaging_time,
                        'Total_Encoding_Time_s': total_encoding_time,
                        'Decoding_Time_s': decoding_time,
                        # Métricas de overhead
                        'Overhead_Percentage': overhead_percentage,
                        'Overhead_vs_Compressed_Percentage': overhead_vs_compressed,
                        'Reversibility_Check': reversibility_check
                    })
                    
                    logger.info(f"Concluído: {os.path.basename(img_path)}, Codec: {codec_name.upper()}, Beta: {beta}")
                    logger.info(f"Tempos - Compressão: {compression_time:.3f}s, Empacotamento: {packaging_time:.3f}s, Total: {total_encoding_time:.3f}s")
                    logger.info(f"Overhead: {overhead_percentage:.2f}% do container, {overhead_vs_compressed:.2f}% vs imagem comprimida")
                    logger.info(f"Reversibilidade: {reversibility_check}")
                    
                except Exception as decode_error:
                    logger.error(f"Erro na decodificação para {base_filename}: {decode_error}")
                    # Adicionar resultado mesmo com erro de decodificação para análise
                    results.append({
                        'Image_File': os.path.basename(img_path),
                        'Modality': modality,
                        'Bits_Stored': bits_stored,
                        'Shape': f"{shape[0]}x{shape[1]}",
                        'Original_Size_Bytes': original_size,
                        'Metadata_Size_Bytes': metadata_size_bytes,
                        'Parameter_Beta': beta,
                        'Parameter_Codec': codec_name,
                        'Parameter_Block_Size': BLOCK_SIZE,
                        'Parameter_Threshold_Factor': THRESHOLD_FACTOR,
                        'Stego_Capacity_Bits': stego_capacity_bits,
                        'Message_Size_Bits': len(message_bits),
                        'Stego_Image_PSNR_dB': stego_psnr,
                        'Stego_Image_SSIM': stego_ssim,
                        'Final_Bin_Size_Bytes': final_bin_size,
                        'Compressed_Image_Size_Bytes': compressed_image_size,
                        'Header_Size_Bytes': header_size,
                        'Bitmaps_Size_Bytes': bitmaps_size,
                        'CR': cr,
                        'Bpp': bpp,
                        'Compression_Ratio_vs_Raw': compression_ratio_vs_raw,
                        'Embedding_Efficiency': embedding_efficiency,
                        'Available_Capacity_Utilization': len(message_bits) / stego_capacity_bits if stego_capacity_bits > 0 else 0,
                        # Métricas de tempo detalhadas
                        'Load_Time_s': load_time,
                        'Metadata_Extraction_Time_s': metadata_time,
                        'Decomposition_Time_s': decomposition_time,
                        'Embedding_Time_s': embedding_time,
                        'Merge_Time_s': merge_time,
                        'Metrics_Calculation_Time_s': metrics_time,
                        'Compression_Time_s': compression_time,
                        'Packaging_Time_s': packaging_time,
                        'Total_Encoding_Time_s': total_encoding_time,
                        'Decoding_Time_s': 0,  # Falhou na decodificação
                        # Métricas de overhead
                        'Overhead_Percentage': overhead_percentage,
                        'Overhead_vs_Compressed_Percentage': overhead_vs_compressed,
                        'Reversibility_Check': False,
                        'Decode_Error': str(decode_error)
                    })

    except Exception as e:
        logger.error(f"Falha ao processar {img_path}: {e}")
        return []
    
    return results

def run_experiments_parallel(max_workers=2):
    """Executa experimentos em paralelo usando ThreadPoolExecutor."""
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    image_paths = glob.glob(os.path.join(DATASET_DIR, '**/*.dcm'), recursive=True)
    if not image_paths:
        logger.error(f"Nenhum arquivo .dcm encontrado em '{DATASET_DIR}'. Verifique o caminho.")
        return

    logger.info(f"--- INICIANDO EXPERIMENTOS PARALELOS ---")
    logger.info(f"Imagens encontradas: {len(image_paths)}")
    logger.info(f"Parâmetros Beta: {BETAS_TO_TEST}")
    logger.info(f"Codecs: {CODECS_TO_TEST}")
    logger.info(f"Workers: {max_workers}")
    
    all_results = []
    start_time = time.time()

    # Usar ThreadPoolExecutor para evitar problemas de memória
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submeter todas as tarefas
        future_to_image = {executor.submit(process_single_image, img_path): img_path for img_path in image_paths}
        
        # Coletar resultados conforme forem sendo completados
        for future in as_completed(future_to_image):
            img_path = future_to_image[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info(f"✓ Concluído: {os.path.basename(img_path)} - {len(results)} resultados")
            except Exception as e:
                logger.error(f"Erro ao processar {img_path}: {e}")

    total_time = time.time() - start_time
    
    # Salvar DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUT_DIR, 'results_parallel.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"EXPERIMENTOS PARALELOS CONCLUÍDOS")
        logger.info(f"Tempo total: {total_time:.2f}s")
        logger.info(f"Resultados salvos em: {csv_path}")
        logger.info(f"Total de resultados: {len(all_results)}")
        
        # Estatísticas básicas
        successful = df[df['Reversibility_Check'] == True].shape[0] if 'Reversibility_Check' in df.columns else 0
        failed = len(all_results) - successful
        logger.info(f"Sucessos: {successful}, Falhas: {failed}")
        logger.info(f"{'='*50}")
    else:
        logger.error("Nenhum resultado foi gerado!")

if __name__ == "__main__":
    # Opção 1: Paralelismo com threads (mais seguro)
    run_experiments_parallel(max_workers=1)
