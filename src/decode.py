import numpy as np
import os
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import tempfile
from datetime import datetime
from tools import (
    load_compressed_stego_bitstream,
    decompress_image,
    save_image,
    build_image_from_modality,
    bits_to_message,
    convert_bitmap_for_processing,
    EOF_MARKER,
    message_to_bits
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

def _predictor_left_top(img, y, x):
    """Preditor usado no PEE - média entre pixel à esquerda e acima"""
    left = int(img[y, x-1]) if x > 0 else 0
    top = int(img[y-1, x]) if y > 0 else 0
    return (left + top) // 2

def pee_extract(local_stego, nbits, bitmap, threshold=2):
    """
    PEE (Prediction Error Expansion) extraction
    
    Args:
        local_stego: Imagem local esteganográfica
        nbits: Número de bits por pixel
        bitmap: Bitmap de posições modificadas
        threshold: Limiar para expansão do erro de predição
    
    Returns:
        tuple: (imagem_recuperada, bits_extraidos, estatisticas)
    """
    maxval = (1 << nbits) - 1

    # Localizar posições marcadas (row-major order)
    positions = np.argwhere(bitmap == 255)
    if positions.size:
        positions = positions[np.lexsort((positions[:,1], positions[:,0]))]  # row-major sort

    n_positions = len(positions)
    
    recovered = local_stego.copy().astype(np.int32)
    bits = []
    mismatches = []
    extracted = 0

    # Formulário do intervalo expandido: |e'| <= 2*T + 1
    max_exp = 2 * threshold + 1
    
    # Bits necessários para o marcador EOF
    eof_bits = message_to_bits(EOF_MARKER)
    eof_length = len(eof_bits)
    
    for idx, (y, x) in enumerate(positions):
        # Computar preditor usando imagem progressivamente recuperada (causal)
        pred = _predictor_left_top(recovered, int(y), int(x))
        e_prime = int(recovered[y, x]) - pred

        # Verificação com threshold: e' deve estar no intervalo de erros expandidos possíveis
        if abs(e_prime) <= max_exp:
            b = e_prime & 1  # bit inserido
            e = (e_prime - b) // 2  # erro original
            orig_pixel = pred + e
            
            # Verificação: pixel original dentro da faixa válida
            if 0 <= orig_pixel <= maxval:
                recovered[y, x] = orig_pixel
                bits.append(int(b))
                extracted += 1
                
                # Verificação automática de EOF APÓS adicionar o bit
                if len(bits) >= eof_length:
                    # Verifica se os últimos bits formam o marcador EOF
                    if bits[-eof_length:] == eof_bits:
                        bits = bits[:-eof_length]  # Remove o marcador dos bits finais
                        pass  # EOF detectado
                        break
                
            else:
                # Marcado no bitmap mas reconstrução dá valor fora de faixa
                mismatches.append({
                    'pos': (int(y), int(x)), 
                    'reason': 'orig_out_of_range', 
                    'pred': pred, 
                    'e_prime': int(e_prime), 
                    'reconstructed': int(orig_pixel)
                })
        else:
            # Marcado no bitmap mas e' não compatível com threshold (possível corrupção)
            mismatches.append({
                'pos': (int(y), int(x)), 
                'reason': 'e_prime_out_of_range', 
                'pred': pred, 
                'e_prime': int(e_prime)
            })

    stats = {
        'positions': n_positions,
        'extracted': extracted,
        'mismatches': len(mismatches),
        'mismatch_positions': mismatches
    }

    return recovered.astype(local_stego.dtype), bits, stats

def decode_from_bitstream(bitstream_path, save_recovered=True, patient_info=None):
    """
    Pipeline completo de decodificação esteganográfica a partir de bitstream
    
    Args:
        bitstream_path: Caminho para o arquivo de bitstream
        save_recovered: Se deve salvar a imagem recuperada (padrão: True)
        patient_info: Dict opcional com informações do paciente para o DICOM
                     {'name': str, 'id': str, 'birth_date': str, 'sex': str, 'institution': str}
    
    Returns:
        dict: Resultados da decodificação
    """
    # Carregar dados completos do bitstream
    stego_data = load_compressed_stego_bitstream(bitstream_path)
    
    # Recuperar componentes comprimidos
    local_recovered_comp = decompress_image(stego_data['local_component'])
    global_recovered_comp = decompress_image(stego_data['global_component'])
    
    # Usar bitmap binário salvo (converter para formato PEE se necessário)
    bitmap_binary = stego_data['bitmap']
    bitmap_pee = convert_bitmap_for_processing(bitmap_binary, 'pee')
    
    # Usar parâmetros salvos
    params = stego_data['stego_params']
    
    # EXTRAÇÃO AUTOMÁTICA usando EOF marker
    local_recovered, bits, stats = pee_extract(
        local_recovered_comp, 
        params['s'], 
        bitmap_pee, 
        threshold=params['threshold']
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
            'name': f'recovered_{image_base_name}.png',
            'is_dicom': False,
            'bits_stored': stego_data['local_component']['original_dtype']().itemsize * 8,
            'metadata': None
        }
        save_image(image_original, png_data, output_folder)
        recovered_paths['png'] = f"{output_folder}/recovered_{image_base_name}.png"
        recovered_paths['png'] = f"{output_folder}/recovered_{image_base_name}.png"
        
        # Salvar como DCM (precisamos criar metadados básicos)
        dcm_data = {
            'name': f'recovered_{image_base_name}.dcm',
            'is_dicom': True,
            'bits_stored': stego_data['local_component']['original_dtype']().itemsize * 8,
            'metadata': _create_basic_dicom_metadata(image_original, patient_info)
        }
        save_image(image_original, dcm_data, output_folder)
        recovered_paths['dcm'] = f"{output_folder}/recovered_{image_base_name}.dcm"
    
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
    dir = "output"
    name = "peito"
    bitstream_path = f'{dir}/{name}/{name}_stego_data.bin'
    
    if not os.path.exists(bitstream_path):
        print(f"❌ Arquivo de bitstream não encontrado: {bitstream_path}")
        print(f"   Execute primeiro o encode.py para gerar o bitstream.")
        return
    
    # Decodificar mensagem da imagem
    result = decode_from_bitstream(bitstream_path)
    
    # Exemplo de verificação (se você souber a mensagem original)
    original_message = (
        " \"É uma mensagem longa para testar a implementação de esteganografia\n"
        "  com funcionalidade de inserção e extração PEE e deve ser suficientemente\n"
        "  extensa para cobrir múltiplos bits e garantir que o processo de inserção\n"
        "  seja devidamente exercitado. Lorem ipsum dolor sit amet, consectetur\n"
        "  adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore\n"
        "  magna aliqua. Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n"
        "  Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\""
    )
    
    verification = verify_extraction(original_message, result['message'])
    print(f"✅ Decodificação finalizada - Precisão: {verification['accuracy']*100:.1f}%")

if __name__ == '__main__':
    main()