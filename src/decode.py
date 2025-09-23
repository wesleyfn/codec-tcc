import numpy as np
import os
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import tempfile
from datetime import datetime
from tools import (
    load_compressed_stego_bitstream_multi as load_stego_data,
    decompress_image,
    save_image,
    build_image_from_modality,
    bits_to_message
)

def _create_basic_dicom_metadata(image_array, original_metadata=None, patient_info=None):
    """
    Cria metadados DICOM consistentes para uma imagem recuperada.
    - Se original_metadata for fornecido, herda e preserva campos importantes.
    - Caso contrário, cria um Secondary Capture mínimo.

    Args:
        image_array: ndarray da imagem
        original_metadata: pydicom Dataset original (opcional)
        patient_info: dict opcional {'name','id','birth_date','sex','institution'}
    """
    if original_metadata is not None:
        # Copia o dataset original
        ds = original_metadata.copy()
        
        # Atualiza campos obrigatórios para imagem derivada
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.StudyInstanceUID = ds.get("StudyInstanceUID", pydicom.uid.generate_uid())

        ds.ImageType = ['DERIVED', 'SECONDARY']
        ds.StudyDescription = "Recovered from Steganography"
        ds.SeriesDescription = "Recovered Series"
        
        # Atualiza tamanho da imagem
        ds.Rows, ds.Columns = image_array.shape[:2]

        # Bits de acordo com o dtype
        ds.BitsAllocated = image_array.dtype.itemsize * 8
        ds.BitsStored = ds.get("BitsStored", ds.BitsAllocated)
        ds.HighBit = ds.BitsStored - 1
        ds.PixelRepresentation = 0  # assume unsigned
    else:
        # Criar dataset DICOM básico do zero
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian
        
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        # Criar dataset principal
        ds = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Metadados básicos obrigatórios
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.InstanceNumber = 1
        
        # Informações do paciente (conforme solicitado) - usar valores fornecidos ou padrão
        if patient_info:
            ds.PatientName = patient_info.get('name', 'ANONYMOUS^PATIENT')
            ds.PatientID = patient_info.get('id', 'ANON_001')
            ds.PatientBirthDate = patient_info.get('birth_date', '19800101')
            ds.PatientSex = patient_info.get('sex', 'O')
            ds.InstitutionName = patient_info.get('institution', 'STEGANOGRAPHY_LAB')
        else:
            ds.PatientName = 'ANONYMOUS^PATIENT'
            ds.PatientID = 'ANON_001'
            ds.PatientBirthDate = '19800101'
            ds.PatientSex = 'O'  # Other/Unknown
            ds.InstitutionName = 'STEGANOGRAPHY_LAB'
        
        # Informações do estudo
        ds.StudyDate = '20250917'
        ds.StudyTime = '120000'
        ds.StudyDescription = 'RECOVERED_FROM_STEGANOGRAPHY'
        ds.SeriesDescription = 'RECOVERED_SERIES'
        ds.Modality = 'CR'  # Computed Radiography
        
        # Informações da imagem - ajustar baseado na imagem real
        ds.Rows = image_array.shape[0]
        ds.Columns = image_array.shape[1]
        
        # Determinar bits baseado no tipo de dados da imagem
        if image_array.dtype == np.uint8:
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
        elif image_array.dtype == np.uint16:
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
        else:
            # Fallback para 16 bits
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
        
        ds.PixelRepresentation = 0  # unsigned
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        
        # Informações adicionais importantes
        ds.ImageType = ['DERIVED', 'SECONDARY']
        ds.AcquisitionNumber = 1
        ds.SeriesNumber = 1
        ds.PixelSpacing = [1.0, 1.0]  # mm por pixel
        ds.WindowCenter = int(image_array.max() // 2)
        ds.WindowWidth = int(image_array.max())

    return ds


def lsb_extract_multi_plane(local_stego, nbits, bitmaps, segments_lengths, n_lsb, bits_used, segment_indices=None):
    """
    LSB multi-plane extraction - reconstruct message from multiple bit planes
    
    Args:
        local_stego: Imagem local esteganográfica
        nbits: Número de bits por pixel
        bitmaps: Lista de bitmaps (um para cada plano LSB)
        segments_lengths: Lista de comprimentos de cada segmento
        n_lsb: Número de planos LSB (igual ao valor s)
        bits_used: Número total de bits da mensagem (sem EOF MARKER)
        segment_indices: Lista de indices de embaralhamento (opcional)
    
    Returns:
        tuple: (imagem_recuperada, bits_extraidos, estatisticas)
    """
    recovered = local_stego.copy().astype(np.int32)
    all_segments = []
    total_extracted = 0
    
    # Máscara para limpar todos os LSBs
    clear_mask = ~((1 << n_lsb) - 1)
    
    # Extrair de cada plano LSB
    for plane_idx in range(n_lsb):
        # Localizar posições marcadas neste bitmap (row-major order)
        # Para multi-plane, bitmap tem valores 0/1 (não 0/255 como PEE)
        bitmap = bitmaps[plane_idx]
        positions = np.argwhere(bitmap == 1)
        if positions.size > 0:
            positions = positions[np.lexsort((positions[:,1], positions[:,0]))]  # row-major sort
        
        # Bits esperados para este segmento
        expected_bits = segments_lengths[plane_idx]
        segment_bits = []
        
        # Máscara para extrair apenas o bit deste plano
        bit_mask = 1 << plane_idx
        
        # Extrair bits deste plano usando operações vetorizadas
        if positions.size > 0:
            # Limitar posições ao número esperado de bits
            num_positions = min(len(positions), expected_bits)
            valid_positions = positions[:num_positions]
            
            # Extrair pixels das posições válidas
            y_coords = valid_positions[:, 0]
            x_coords = valid_positions[:, 1]
            pixel_values = local_stego[y_coords, x_coords]
            
            # Extrair bits específicos deste plano usando operações vetorizadas
            segment_bits = ((pixel_values >> plane_idx) & 1).tolist()
            
            # Garantir tamanho exato
            if len(segment_bits) < expected_bits:
                print(f"⚠️  Plano {plane_idx}: Extraído {len(segment_bits)} bits, esperado {expected_bits}")
            else:
                segment_bits = segment_bits[:expected_bits]
        else:
            segment_bits = []
        
        all_segments.append(segment_bits)
        total_extracted += len(segment_bits)
    
    # Limpar todos os LSBs da imagem recuperada usando operação vetorizada
    recovered &= clear_mask
    
    # Reconstruir mensagem na ordem original
    if segment_indices is not None:
        # Reverter o embaralhamento: reordenar segmentos pela ordem original
        original_segments = [None] * n_lsb
        for i in range(n_lsb):
            original_idx = segment_indices[i]  # onde o segmento i foi colocado
            original_segments[original_idx] = all_segments[i]
        
        # Concatenar segmentos na ordem original
        reconstructed_bits = []
        for segment in original_segments:
            reconstructed_bits.extend(segment)
    else:
        # Sem informação de embaralhamento - concatenar na ordem atual
        reconstructed_bits = []
        for segment in all_segments:
            reconstructed_bits.extend(segment)
    
    # Usar bits_used para obter exatamente a mensagem original (sem EOF MARKER)
    final_bits = reconstructed_bits[:bits_used] if len(reconstructed_bits) >= bits_used else reconstructed_bits
    
    # Estatísticas
    stats = {
        'planes_used': n_lsb,
        'segments_lengths': segments_lengths,
        'total_positions': sum(len(np.argwhere(bmp == 1)) for bmp in bitmaps),
        'bits_extracted': len(final_bits),
        'total_extracted': total_extracted,
        'method': 'LSB_MULTI_PLANE',
        'n_lsb': n_lsb
    }
    
    return recovered.astype(local_stego.dtype), final_bits, stats



def decode_from_bitstream(bitstream_path, save_recovered=True, patient_info=None):
    """
    Pipeline completo de decodificação esteganográfica LSB multi-plano (STG3)
    
    Args:
        bitstream_path: Caminho para o arquivo de bitstream STG3
        save_recovered: Se deve salvar a imagem recuperada (padrão: True)
        patient_info: Dict opcional com informações do paciente para o DICOM
                     {'name': str, 'id': str, 'birth_date': str, 'sex': str, 'institution': str}
    
    Returns:
        dict: Resultados da decodificação
    """
    # Carregar dados do formato STG3
    stego_data = load_stego_data(bitstream_path)
    
    # Recuperar componentes comprimidos
    local_recovered_comp = decompress_image(stego_data['local_component'])
    global_recovered_comp = decompress_image(stego_data['global_component'])
    
    # Usar parâmetros salvos
    params = stego_data['stego_params']
    
    # EXTRAÇÃO LSB multi-plano (STG3)
    print(f"🔄 Extraindo usando LSB multi-plano (STG3)")
    local_recovered, bits, stats = lsb_extract_multi_plane(
        local_recovered_comp,
        params['s'],
        stego_data['bitmaps'],
        params['segments_lengths'],
        params['s'],  # n_lsb = s
        params['bits_used'],  # usar bits_used em vez de EOF MARKER
        params.get('segment_indices')
    )

    # Reconstruir imagem original
    image_original = build_image_from_modality(local_recovered, global_recovered_comp)
    
    # Salvar imagem recuperada se solicitado
    recovered_paths = {}
    if save_recovered:
        # Determinar dados de saída baseados no caminho do bitstream
        output_folder = os.path.dirname(bitstream_path)
        bitstream_name = os.path.basename(bitstream_path)
        image_base_name = bitstream_name.replace('_stego_data.bin', '')
        
        # Salvar como PNG para debug rápido
        png_data = {
            'name': f'{image_base_name}_recovered.png',
            'is_dicom': False,
            'bits_stored': stego_data['local_component']['original_dtype']().itemsize * 8,
            'metadata': None
        }
        save_image(image_original, png_data, output_folder)
        recovered_paths['png'] = f"{output_folder}/{image_base_name}_recovered.png"
        
        # Salvar como DCM (precisamos criar metadados básicos)
        dcm_data = {
            'name': f'{image_base_name}_recovered.dcm',
            'is_dicom': True,
            'bits_stored': stego_data['local_component']['original_dtype']().itemsize * 8,
            'metadata': _create_basic_dicom_metadata(image_original, patient_info)
        }
        save_image(image_original, dcm_data, output_folder)
        recovered_paths['dcm'] = f"{output_folder}/{image_base_name}_recovered.dcm"
    
    # Converter bits para mensagem
    recovered_message = bits_to_message(bits)
    
    # Decodificação concluída
    
    return {
        'message': recovered_message,
        'recovered_image': image_original,
        'bits': bits,
        'stats': stats,
        'recovered_paths': recovered_paths,
        'stego_params': params,
        'metadata': stego_data['metadata']
    }

def decode_message_only(bitstream_path):
    """
    Extrai apenas a mensagem sem salvar a imagem recuperada
    
    Args:
        bitstream_path: Caminho para o arquivo de bitstream
    
    Returns:
        str: Mensagem extraída
    """
    result = decode_from_bitstream(bitstream_path, save_recovered=False)
    return result['message']

def verify_extraction(original_message, extracted_message):
    """
    Verifica a integridade da extração da mensagem
    
    Args:
        original_message: Mensagem original
        extracted_message: Mensagem extraída
    
    Returns:
        dict: Resultados da verificação
    """
    # Verificar se as mensagens são idênticas
    match = original_message == extracted_message
    
    if not match:
        min_len = min(len(original_message), len(extracted_message))
        diff_positions = []
        for i in range(min_len):
            if original_message[i] != extracted_message[i]:
                diff_positions.append(i)
    
    return {
        'match': match,
        'original_length': len(original_message),
        'extracted_length': len(extracted_message),
        'accuracy': len(extracted_message) / len(original_message) if original_message else 0
    }

def main():
    """Exemplo de uso do decoder"""
    name = "torax"
    bitstream_path = f'output/{name}/{name}_stego_data.bin'
    
    if not os.path.exists(bitstream_path):
        print(f"❌ Arquivo de bitstream não encontrado: {bitstream_path}")
        print(f"   Execute primeiro o encode.py para gerar o bitstream.")
        return
    
    # Decodificar mensagem da imagem
    result = decode_from_bitstream(bitstream_path, save_recovered=True)
    
    # Exemplo de verificação (se você souber a mensagem original)
    original_message = (
        " \"É uma mensagem longa para testar a implementação de esteganografia\n"
        "  com funcionalidade de inserção e extração LSB multi-plano e deve ser\n"
        "  suficientemente extensa para cobrir múltiplos bits e garantir que o\n"
        "  processo de inserção seja devidamente exercitado. Lorem ipsum dolor\n"
        "  sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt\n"
        "  ut labore et dolore magna aliqua. Lorem ipsum dolor sit amet, consectetur\n"
        "  adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore\n"
        "  magna aliqua.\""
    )

    print(f"📝 Mensagem extraída:\n{result['message']}")
    
    verification = verify_extraction(original_message, result['message'])
    print(f"✅ Decodificação finalizada - Precisão: {verification['accuracy']*100:.1f}%")

if __name__ == '__main__':
    main()