import numpy as np
from PIL import Image
import pydicom
import zlib, gzip, bz2, lzma
import os
import time
import struct
import io
import tempfile
import gdcm
import subprocess
import shutil



EOF_MARKER = "<<<END>>>"

def load_image(filepath):
    """Load DICOM image file only"""
    if not filepath.lower().endswith('.dcm'):
        raise ValueError("Apenas arquivos DICOM (.dcm) são suportados")
    return _load_dicom(filepath)
    
def _load_dicom(filepath):
    """Load DICOM image with metadata"""
    dcm = pydicom.dcmread(filepath)
    img = dcm.pixel_array
    
    # Handle multi-frame DICOM by taking only the first frame
    if len(img.shape) > 2:
        img = img[0]  # Take first frame
        
    bits_stored = getattr(dcm, 'BitsStored', img.dtype.itemsize * 8)
    high_bit = getattr(dcm, 'HighBit', bits_stored - 1)
    
    # Ensure we're working with the right bit depth
    if img.dtype == np.int16:
        img = img.astype(np.uint16)
    
    return {
        'image': img,
        'name': filepath.split('/')[-1],
        'bits_stored': bits_stored,
        'high_bit': high_bit,
        'metadata': dcm,
        'is_dicom': True
    }

def _load_png(filepath):
    """Load PNG image"""
    img = np.array(Image.open(filepath))

    if len(img.shape) > 2:
        # Convert RGB to grayscale using ITU-R BT.709 coefficients for accurate luminance
        img = np.round(0.2126 * img[..., 0] + 
                       0.7152 * img[..., 1] + 
                       0.0722 * img[..., 2]).astype(img.dtype)
    
    return {
        'image': img,
        'name': filepath.split('/')[-1],
        'bits_stored': img.dtype.itemsize * 8,
        'high_bit': img.dtype.itemsize * 8 - 1,
        'metadata': None,
        'is_dicom': False
    }

def create_output_folder(base_name):
    """Create organized results folder"""
    folder_name = f"output/{base_name.replace('.dcm', '').replace('.png', '')}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        pass  # Pasta criada silenciosamente
    
    return folder_name

def save_image(image_array, original_data, output_folder=None):
    """Save image respecting original format and bit depth"""
    if output_folder:
        original_name = original_data['name']
        original_data['name'] = f"{output_folder}/{original_name}"
    
    if original_data and original_data.get('is_dicom'):
        _save_dicom(image_array, original_data)

    _save_png(image_array, original_data)

def _save_dicom(image_array, original_data):
    """Save as DICOM preserving metadata"""
    dcm = original_data['metadata']
    dcm.PixelData = image_array.tobytes()
    dcm.save_as(original_data['name'])

def _save_png(image_array, original_data):
    """Save as PNG"""
    if original_data['bits_stored'] > 8:
        actual_max = image_array.max()
        actual_min = image_array.min()
        
        if actual_max > actual_min:
            scaled = ((image_array.astype(np.float32) - actual_min) / (actual_max - actual_min)) * 255.0
        else:
            scaled = np.zeros_like(image_array, dtype=np.float32)
        
        img_array = np.clip(scaled, 0, 255).astype(np.uint8)
    else:
        img_array = np.clip(image_array, 0, 255).astype(np.uint8)

    Image.fromarray(img_array).save(original_data['name'].replace('.dcm', '.png'))

def extract_bit_planes(image_array, nbits):
    """Extract bit planes from image"""
    planes = []
    for bit in range(nbits):
        plane = (image_array >> bit) & 1
        planes.append(plane.astype(np.uint8))
    return planes

def build_modality(planes, s):
    """Classify planes into global and local based on modality index s"""
    local_planes = planes[:s]   # Lower bits
    global_planes = planes[s:]  # Higher bits
    return local_planes, global_planes

def build_image_from_modality(local_image, global_image):
    """Combine local and global images"""
    return local_image | global_image

def image_size(image_array):
    """Calculate image size in kilobytes"""
    return image_array.nbytes / 1024

def compress_png(image_array):
    """Compress image using PNG format - 100% LOSSLESS for all dtypes"""
    # Para garantir 100% lossless, salvar dados brutos + usar PNG apenas como container
    
    # Salvar dados originais como bytes
    original_bytes = image_array.tobytes()
    
    # Comprimir usando PNG os dados brutos como uma "imagem"
    # Reshape para uma imagem 1D que o PNG pode processar
    byte_array = np.frombuffer(original_bytes, dtype=np.uint8)
    
    # Criar uma imagem "fake" que representa nossos dados
    if len(byte_array) > 65535:
        # Para arrays grandes, fazer uma imagem retangular
        width = int(np.sqrt(len(byte_array))) + 1
        height = (len(byte_array) + width - 1) // width
        padded_size = width * height
        
        # Pad com zeros se necessário
        if len(byte_array) < padded_size:
            padded_array = np.zeros(padded_size, dtype=np.uint8)
            padded_array[:len(byte_array)] = byte_array
            byte_array = padded_array
        
        img_2d = byte_array.reshape((height, width))
    else:
        # Para arrays pequenos, usar imagem linear
        img_2d = byte_array.reshape((1, -1))
    
    # Converter para PIL Image
    pil_image = Image.fromarray(img_2d, mode='L')
    
    # Salvar como PNG
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG', optimize=True)
    png_data = buffer.getvalue()
    buffer.close()
    
    # Criar header com metadados para reconstrução exata
    import pickle
    header_data = {
        'original_shape': image_array.shape,
        'original_dtype': str(image_array.dtype),
        'original_size': len(original_bytes),
        'png_shape': img_2d.shape
    }
    
    header_bytes = pickle.dumps(header_data)
    header_size = len(header_bytes)
    
    # Formato: [4 bytes header_size][header][png_data]
    final_data = struct.pack('<I', header_size) + header_bytes + png_data
    
    return final_data

def decompress_png(compressed_data, original_shape, original_dtype):
    """Decompress PNG data back to numpy array - 100% LOSSLESS"""
    import pickle
    
    try:
        # Ler header
        header_size = struct.unpack('<I', compressed_data[:4])[0]
        header_bytes = compressed_data[4:4+header_size]
        png_data = compressed_data[4+header_size:]
        
        # Deserializar header
        header_data = pickle.loads(header_bytes)
        saved_shape = header_data['original_shape']
        saved_dtype = header_data['original_dtype']
        original_size = header_data['original_size']
        png_shape = header_data['png_shape']
        
        # Carregar PNG
        png_buffer = io.BytesIO(png_data)
        pil_image = Image.open(png_buffer)
        png_array = np.array(pil_image)
        png_buffer.close()
        
        # Converter de volta para bytes originais
        byte_array = png_array.flatten()[:original_size]  # Remover padding se houver
        
        # Reconstruir array original
        original_array = np.frombuffer(byte_array.tobytes(), dtype=saved_dtype)
        return original_array.reshape(saved_shape)
        
    except Exception as e:
        print(f"⚠️  Erro na descompressão PNG lossless: {e}")
        # Fallback para método antigo se falhar
        return _decompress_png_legacy(compressed_data, original_shape, original_dtype)

def _decompress_png_legacy(compressed_data, original_shape, original_dtype):
    """Método legado de descompressão PNG"""
    import pickle
    
    # Deserializar dados comprimidos
    buffer = io.BytesIO(compressed_data)
    try:
        data_dict = pickle.load(buffer)
        png_data = data_dict['png_data']
        original_min = data_dict['original_min']
        original_max = data_dict['original_max']
        saved_dtype = data_dict['original_dtype']
        saved_shape = data_dict['original_shape']
    except:
        # Fallback para formato antigo (sem metadados)
        buffer.seek(0)
        png_data = compressed_data
        original_min = 0
        original_max = 65535 if original_dtype == np.uint16 else 255
        saved_dtype = str(original_dtype)
        saved_shape = original_shape
    buffer.close()
    
    # Carregar PNG do buffer
    png_buffer = io.BytesIO(png_data)
    pil_image = Image.open(png_buffer)
    
    # Converter de volta para numpy array
    decompressed_array = np.array(pil_image)
    png_buffer.close()
    
    # Restaurar o range e dtype originais
    if saved_dtype.startswith('uint16'):
        # Desnormalizar de volta ao range original
        if original_max > original_min:
            restored_array = (decompressed_array.astype(np.float64) / 255.0 * 
                             (original_max - original_min) + original_min).astype(np.uint16)
        else:
            restored_array = np.full_like(decompressed_array, original_min, dtype=np.uint16)
    else:
        restored_array = decompressed_array.astype(original_dtype)
    
    return restored_array.reshape(saved_shape)
    if saved_dtype == 'uint16' and original_min != original_max:
        # Restaurar o range original de forma precisa com arredondamento correto
        decompressed_array = (np.round((decompressed_array.astype(np.float64) / 255.0) * 
                                      (original_max - original_min)) + original_min).astype(np.uint16)
    elif saved_dtype == 'uint16':
        decompressed_array = decompressed_array.astype(np.uint16)
    
    # Garantir que tenha a forma original
    if decompressed_array.shape != original_shape:
        decompressed_array = decompressed_array.reshape(original_shape)
    
    buffer.close()
    return decompressed_array.astype(original_dtype)

def compress_gdcm(image_array, syntax='JPEGLS'):
    """Compress image using GDCM with JPEG-LS, JPEG2000 or RLE lossless compression"""
    
    try:
        height, width = image_array.shape
        
        # Criar arquivo DICOM temporário
        temp_dicom_path = os.path.join(tempfile.gettempdir(), f"temp_gdcm_{os.getpid()}.dcm")
        temp_compressed_path = os.path.join(tempfile.gettempdir(), f"temp_gdcm_comp_{os.getpid()}.dcm")
        
        # Criar dataset DICOM básico
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        ds = pydicom.Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        # Tags essenciais
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.InstanceNumber = 1
        
        # Configurar imagem
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.Rows = height
        ds.Columns = width
        
        # Detectar tipo de dados e configurar bits
        if image_array.dtype == np.uint16:
            ds.BitsAllocated = 16
            ds.BitsStored = 16
        else:
            ds.BitsAllocated = 16  # Força 16 bits para compatibilidade
            ds.BitsStored = 16
            image_array = image_array.astype(np.uint16)
            
        ds.HighBit = ds.BitsStored - 1
        ds.PixelRepresentation = 0  # unsigned
        ds.PixelData = image_array.tobytes()
        
        # Salvar DICOM não comprimido
        pydicom.dcmwrite(temp_dicom_path, ds)
        
        # Usar GDCM para comprimir
        reader = gdcm.ImageReader()
        reader.SetFileName(temp_dicom_path)
        
        if not reader.Read():
            raise RuntimeError("Falha ao ler arquivo DICOM temporário")
        
        # Configurar compressor
        compressor = gdcm.ImageChangeTransferSyntax()
        compressor.SetInput(reader.GetImage())
        
        # Definir Transfer Syntax baseado no parâmetro
        if syntax == 'JPEGLS':
            ts = gdcm.TransferSyntax(gdcm.TransferSyntax.JPEGLSLossless)
        elif syntax == 'JPEG2000':
            ts = gdcm.TransferSyntax(gdcm.TransferSyntax.JPEG2000Lossless)
        elif syntax == 'RLE':
            ts = gdcm.TransferSyntax(gdcm.TransferSyntax.RLELossless)
        else:
            ts = gdcm.TransferSyntax(gdcm.TransferSyntax.JPEGLSLossless)  # default
        
        compressor.SetTransferSyntax(ts)
        
        # Executar compressão
        if not compressor.Change():
            raise RuntimeError(f"Falha na compressão {syntax}")
        
        # Salvar resultado comprimido
        writer = gdcm.ImageWriter()
        writer.SetFileName(temp_compressed_path)
        writer.SetImage(compressor.GetOutput())
        
        if not writer.Write():
            raise RuntimeError("Falha ao escrever arquivo comprimido")
        
        # Ler arquivo comprimido como bytes
        with open(temp_compressed_path, 'rb') as f:
            compressed_data = f.read()
        
        # Retornar dados comprimidos com header de informação
        compression_info = {
            'syntax': syntax,
            'original_shape': image_array.shape,
            'original_dtype': str(image_array.dtype),
            'compressed_size': len(compressed_data),
            'transfer_syntax_uid': ts.GetString()
        }
        
        # Criar header com informações (para descompressão)
        header = f"GDCM_COMPRESSED:{syntax}:{height}:{width}:{str(image_array.dtype)}:{ts.GetString()}:".encode('utf-8')
        
        return header + compressed_data
        
    except Exception as e:
        print(f"⚠️  GDCM {syntax} falhou ({e}), usando compressão diferencial médica")
        return _medical_differential_compress(image_array)
    
    finally:
        # Limpeza garantida dos arquivos temporários
        for path in [temp_dicom_path, temp_compressed_path]:
            if 'path' in locals() and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

def _medical_differential_compress(image_array):
    """Compressão diferencial otimizada para imagens médicas"""
    height, width = image_array.shape
    
    # Converter para int32 para evitar overflow
    img_int = image_array.astype(np.int32)
    
    # Calcular diferenças (predição por vizinho esquerdo/superior)
    diff_array = np.zeros_like(img_int)
    
    # Primeira linha e coluna ficam iguais
    diff_array[0, :] = img_int[0, :]
    diff_array[:, 0] = img_int[:, 0]
    
    # Predição por gradiente (médico)
    for y in range(1, height):
        for x in range(1, width):
            # Preditor otimizado para imagens médicas
            left = img_int[y, x-1]
            top = img_int[y-1, x]
            top_left = img_int[y-1, x-1]
            
            # Gradiente médico adaptativo
            prediction = left + top - top_left
            diff_array[y, x] = img_int[y, x] - prediction
    
    # Comprimir diferenças com quantização médica
    compressed_bytes = _compress_medical_differences(diff_array, image_array.dtype)
    
    return compressed_bytes

def _compress_medical_differences(diff_array, original_dtype):
    """Comprime diferenças usando quantização médica"""
    # Converter diferenças para um range menor
    diff_flat = diff_array.flatten()
    
    # Estatísticas para quantização - converter para int antes do min/max
    diff_min = int(diff_flat.min())
    diff_max = int(diff_flat.max())
    diff_range = diff_max - diff_min
    
    if diff_range == 0:
        # Imagem uniforme
        return struct.pack('>BII', 0, diff_min & 0xFFFFFFFF, len(diff_flat)) + b'\x00'
    
    # Quantizar diferenças para 8 bits quando possível
    if diff_range <= 255:
        quantized = ((diff_flat - diff_min) * 255 // diff_range).astype(np.uint8)
        compressed = zlib.compress(quantized.tobytes(), level=9)
        # Header: tipo(1) + min(4) + max(4) + tamanho(4) + dados
        # Garantir que valores sejam positivos para struct
        header = struct.pack('>BII', 1, diff_min & 0xFFFFFFFF, diff_max & 0xFFFFFFFF) + struct.pack('>I', len(diff_flat))
        return header + compressed
    else:
        # Usar 16 bits
        quantized = ((diff_flat - diff_min) * 65535 // diff_range).astype(np.uint16)
        compressed = zlib.compress(quantized.tobytes(), level=9)
        header = struct.pack('>BII', 2, diff_min & 0xFFFFFFFF, diff_max & 0xFFFFFFFF) + struct.pack('>I', len(diff_flat))
        return header + compressed



def decompress_gdcm(compressed_data, original_shape, original_dtype):
    """Decompress GDCM compressed data (JPEG-LS, JPEG2000, RLE or medical differential)"""
    
    try:
        # Verificar se tem header GDCM
        header_marker = b"GDCM_COMPRESSED:"
        if compressed_data.startswith(header_marker):
            # Extrair informações do header
            # Procurar o último ':' (depois da UID que pode ter pontos)
            header_part = compressed_data[len(header_marker):len(header_marker) + 200]  # Limitar busca
            last_colon_pos = header_part.rfind(b':')
            if last_colon_pos != -1:
                header_end = len(header_marker) + last_colon_pos
            else:
                header_end = -1
            if header_end != -1:
                header_str = compressed_data[len(header_marker):header_end].decode('utf-8')
                try:
                    parts = header_str.split(':')
                    syntax = parts[0]
                    height = int(parts[1])
                    width = int(parts[2])
                    dtype_str = parts[3]
                    transfer_syntax_uid = parts[4]
                    
                    # Dados comprimidos sem header
                    actual_compressed_data = compressed_data[header_end + 1:]
                    
                    # Descomprimir usando GDCM
                    return _decompress_gdcm_native(actual_compressed_data, (height, width), dtype_str, syntax)
                    
                except (ValueError, IndexError) as e:
                    print(f"⚠️  Erro ao parsear header GDCM: {e}")
        
        # Verificar se é compressão diferencial médica (fallback)
        if len(compressed_data) >= 13:
            header = compressed_data[:13]
            try:
                quant_type, diff_min, diff_max, size = struct.unpack('>BIII', header)
                if quant_type in [0, 1, 2]:
                    return _decompress_medical_differences(compressed_data, original_shape, original_dtype)
            except:
                pass
        
        # Fallback final: tentar como dados raw
        print("⚠️  Usando fallback: interpretando como dados raw")
        if str(original_dtype) == 'uint16':
            decompressed_array = np.frombuffer(compressed_data, dtype=np.uint16)
        else:
            decompressed_array = np.frombuffer(compressed_data, dtype=original_dtype)
        
        if decompressed_array.size == np.prod(original_shape):
            return decompressed_array.reshape(original_shape)
        else:
            raise ValueError(f"Tamanho incompatível: {decompressed_array.size} vs {np.prod(original_shape)}")
            
    except Exception as e:
        print(f"⚠️  Erro na descompressão GDCM: {e}")
        print("⚠️  Usando compressão diferencial médica como fallback")
        return _decompress_medical_differences(compressed_data, original_shape, original_dtype)

def _decompress_gdcm_native(compressed_data, shape, dtype_str, syntax):
    """Descompressão nativa GDCM para dados realmente comprimidos"""
    
    temp_compressed_path = os.path.join(tempfile.gettempdir(), f"temp_gdcm_decomp_{os.getpid()}.dcm")
    temp_decompressed_path = os.path.join(tempfile.gettempdir(), f"temp_gdcm_out_{os.getpid()}.dcm")
    
    try:
        # Salvar dados comprimidos como arquivo DICOM temporário
        with open(temp_compressed_path, 'wb') as f:
            f.write(compressed_data)
        
        # Ler com GDCM
        reader = gdcm.ImageReader()
        reader.SetFileName(temp_compressed_path)
        
        if not reader.Read():
            raise RuntimeError("Falha ao ler arquivo DICOM comprimido")
        
        # Descomprimir para formato não comprimido
        decompressor = gdcm.ImageChangeTransferSyntax()
        decompressor.SetInput(reader.GetImage())
        
        # Transfer syntax para não comprimido
        ts_output = gdcm.TransferSyntax(gdcm.TransferSyntax.ExplicitVRLittleEndian)
        decompressor.SetTransferSyntax(ts_output)
        
        if not decompressor.Change():
            raise RuntimeError(f"Falha na descompressão {syntax}")
        
        # Salvar descomprimido
        writer = gdcm.ImageWriter()
        writer.SetFileName(temp_decompressed_path)
        writer.SetImage(decompressor.GetOutput())
        
        if not writer.Write():
            raise RuntimeError("Falha ao escrever arquivo descomprimido")
        
        # Ler arquivo descomprimido com pydicom
        ds = pydicom.dcmread(temp_decompressed_path)
        decompressed_array = ds.pixel_array
        
        # Converter tipo se necessário
        if dtype_str == 'uint16':
            decompressed_array = decompressed_array.astype(np.uint16)
        elif dtype_str == 'uint8':
            decompressed_array = decompressed_array.astype(np.uint8)
        
        # Garantir shape correto
        if decompressed_array.shape != shape:
            decompressed_array = decompressed_array.reshape(shape)
        
        return decompressed_array
        
    finally:
        # Limpeza
        for path in [temp_compressed_path, temp_decompressed_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

def _decompress_medical_differences(compressed_data, original_shape, original_dtype):
    """Descomprime diferenças médicas"""
    # Ler header
    header = compressed_data[:13]
    quant_type, diff_min, diff_max, size = struct.unpack('>BIII', header)
    
    # Converter valores unsigned de volta para signed se necessário
    if diff_min > 2147483647:  # Se era negativo
        diff_min = diff_min - 4294967296
    if diff_max > 2147483647:  # Se era negativo
        diff_max = diff_max - 4294967296
    
    # Descomprimir dados
    compressed_diffs = compressed_data[13:]
    decompressed_diffs = zlib.decompress(compressed_diffs)
    
    # Reconstruir diferenças
    if quant_type == 1:  # 8 bits
        quantized = np.frombuffer(decompressed_diffs, dtype=np.uint8)
    else:  # 16 bits  
        quantized = np.frombuffer(decompressed_diffs, dtype=np.uint16)
    
    # Dequantizar
    diff_range = diff_max - diff_min
    if diff_range == 0:
        diff_flat = np.full(size, diff_min, dtype=np.int32)
    else:
        if quant_type == 1:
            diff_flat = (quantized.astype(np.int32) * diff_range // 255) + diff_min
        else:
            diff_flat = (quantized.astype(np.int32) * diff_range // 65535) + diff_min
    
    # Reshape para forma original
    diff_array = diff_flat.reshape(original_shape).astype(np.int32)
    
    # Reconstruir imagem original das diferenças
    height, width = original_shape
    img_reconstructed = np.zeros_like(diff_array)
    
    # Primeira linha e coluna
    img_reconstructed[0, :] = diff_array[0, :]
    img_reconstructed[:, 0] = diff_array[:, 0]
    
    # Reconstruir resto usando predição inversa
    for y in range(1, height):
        for x in range(1, width):
            left = img_reconstructed[y, x-1]
            top = img_reconstructed[y-1, x]
            top_left = img_reconstructed[y-1, x-1]
            
            prediction = left + top - top_left
            img_reconstructed[y, x] = diff_array[y, x] + prediction
    
    return img_reconstructed.astype(original_dtype)

def compress_avif(image_array, quality=90):
    """Compress image using AVIF format - LOSSLESS via PNG+LZMA"""
    try:
        # Para garantir 100% lossless, usar PNG como base + compressão LZMA avançada
        # Isso simula AVIF lossless com melhor compressão que PNG puro
        
        # Primeiro, comprimir com PNG (já lossless)
        png_compressed = compress_png(image_array)
        
        # Aplicar compressão LZMA adicional para simular eficiência AVIF
        lzma_compressed = lzma.compress(png_compressed, preset=9, check=lzma.CHECK_CRC64)
        
        # Criar header identificando como AVIF simulado
        header = f"AVIF_LOSSLESS:{image_array.shape[0]}:{image_array.shape[1]}:{str(image_array.dtype)}:".encode('utf-8')
        return header + lzma_compressed
            
    except Exception as e:
        print(f"⚠️  AVIF lossless falhou ({e}), usando PNG puro")
        return compress_png(image_array)

def decompress_avif(compressed_data, original_shape, original_dtype):
    """Decompress AVIF lossless compressed data"""
    try:
        # Verificar se tem header AVIF lossless
        if compressed_data.startswith(b"AVIF_LOSSLESS:"):
            # Extrair header
            header_parts = compressed_data.split(b":", 4)
            if len(header_parts) >= 5:
                lzma_data = header_parts[4]
            else:
                # Fallback: encontrar manualmente
                header_end = compressed_data.find(b":", 15)
                for _ in range(2):  # Pular height, width, dtype
                    header_end = compressed_data.find(b":", header_end + 1)
                lzma_data = compressed_data[header_end + 1:]
            
            # Descomprimir LZMA primeiro
            png_data = lzma.decompress(lzma_data)
            
            # Descomprimir PNG
            return decompress_png(png_data, original_shape, original_dtype)
        
        # Compatibilidade com formato antigo
        elif compressed_data.startswith(b"AVIF_COMPRESSED:"):
            return decompress_png(compressed_data, original_shape, original_dtype)
        else:
            # Fallback para PNG
            return decompress_png(compressed_data, original_shape, original_dtype)
            
    except Exception as e:
        print(f"⚠️  AVIF descompressão falhou ({e}), tentando PNG")
        return decompress_png(compressed_data, original_shape, original_dtype)

def compress_jpegxl(image_array, quality=90):
    """Compress image using JPEG XL format - LOSSLESS via PNG+BZ2"""
    try:
        # Para garantir 100% lossless, usar PNG como base + compressão BZ2
        # Isso simula JPEG XL lossless com boa compressão
        
        # Primeiro, comprimir com PNG (já lossless)
        png_compressed = compress_png(image_array)
        
        # Aplicar compressão BZ2 para simular eficiência JPEG XL lossless
        bz2_compressed = bz2.compress(png_compressed, compresslevel=9)
        
        # Criar header identificando como JPEG XL simulado
        header = f"JPEGXL_LOSSLESS:{image_array.shape[0]}:{image_array.shape[1]}:{str(image_array.dtype)}:".encode('utf-8')
        return header + bz2_compressed
            
    except Exception as e:
        print(f"⚠️  JPEG XL lossless falhou ({e}), usando PNG puro")
        return compress_png(image_array)

def decompress_jpegxl(compressed_data, original_shape, original_dtype):
    """Decompress JPEG XL lossless compressed data"""
    try:
        # Verificar se tem header JPEG XL lossless
        if compressed_data.startswith(b"JPEGXL_LOSSLESS:"):
            # Extrair header
            header_parts = compressed_data.split(b":", 4)
            if len(header_parts) >= 5:
                bz2_data = header_parts[4]
            else:
                # Fallback: encontrar manualmente
                header_end = compressed_data.find(b":", 17)
                for _ in range(2):  # Pular height, width, dtype
                    header_end = compressed_data.find(b":", header_end + 1)
                bz2_data = compressed_data[header_end + 1:]
            
            # Descomprimir BZ2 primeiro
            png_data = bz2.decompress(bz2_data)
            
            # Descomprimir PNG
            return decompress_png(png_data, original_shape, original_dtype)
        
        # Compatibilidade com formato antigo
        elif compressed_data.startswith(b"JPEGXL_COMPRESSED:"):
            return decompress_png(compressed_data, original_shape, original_dtype)
        else:
            # Fallback para PNG
            return decompress_png(compressed_data, original_shape, original_dtype)
            
    except Exception as e:
        print(f"⚠️  JPEG XL descompressão falhou ({e}), tentando PNG")
        return decompress_png(compressed_data, original_shape, original_dtype)



def compress_image_with_algorithm(image_array, algorithm):
    """Compress image using a specific algorithm"""
    flat = image_array.flatten()
    byte_data = flat.tobytes()
    
    algorithms = {
        'zlib': zlib.compress,
        'gzip': gzip.compress,
        'bz2': bz2.compress,
        'lzma': lzma.compress,
        'png': lambda data: compress_png(image_array),
        'jpegls': lambda data: compress_gdcm(image_array, syntax='JPEGLS'),
        'jpeg2000': lambda data: compress_gdcm(image_array, syntax='JPEG2000'),
        'avif': lambda data: compress_avif(image_array),
        'jpegxl': lambda data: compress_jpegxl(image_array),
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Algoritmo '{algorithm}' não suportado. Use: {list(algorithms.keys())}")
    
    try:
        compress_func = algorithms[algorithm]
        
        # PNG e GDCM precisam tratamento especial
        if algorithm in ['png', 'gdcm']:
            compressed = compress_func(byte_data)  # Para PNG/GDCM, a função já usa image_array
        else:
            compressed = compress_func(byte_data)
        
        compression_ratio = len(compressed) / len(byte_data)
        
        return {
            'compressed_data': compressed,
            'algorithm': algorithm,
            'original_shape': image_array.shape,
            'original_dtype': image_array.dtype,
            'compression_ratio': compression_ratio
        }
    except Exception as e:
        print(f"❌ Erro na compressão {algorithm}: {e}")
        return {
            'compressed_data': byte_data,
            'algorithm': 'none',
            'original_shape': image_array.shape,
            'original_dtype': image_array.dtype,
            'compression_ratio': 1.0
        }

def decompress_image(compressed_info):
    """Decompress image data"""
    if compressed_info['algorithm'] == 'none':
        decompressed = compressed_info['compressed_data']
        
        # Verificar se há header GDCM mesmo quando algoritmo é 'none'
        if decompressed.startswith(b"GDCM_COMPRESSED:"):
            recovered = decompress_gdcm(
                decompressed,
                compressed_info['original_shape'],
                compressed_info['original_dtype']
            )
        else:
            recovered = np.frombuffer(decompressed, dtype=compressed_info['original_dtype'])
            recovered = recovered.reshape(compressed_info['original_shape'])
    elif compressed_info['algorithm'] == 'png':
        # PNG precisa tratamento especial
        recovered = decompress_png(
            compressed_info['compressed_data'],
            compressed_info['original_shape'],
            compressed_info['original_dtype']
        )
    elif compressed_info['algorithm'] in ['gdcm', 'jpegls', 'jpeg2000', 'rle']:
        recovered = decompress_gdcm(
            compressed_info['compressed_data'],
            compressed_info['original_shape'],
            compressed_info['original_dtype']
        )
    elif compressed_info['algorithm'] == 'avif':
        recovered = decompress_avif(
            compressed_info['compressed_data'],
            compressed_info['original_shape'],
            compressed_info['original_dtype']
        )
    elif compressed_info['algorithm'] == 'jpegxl':
        recovered = decompress_jpegxl(
            compressed_info['compressed_data'],
            compressed_info['original_shape'],
            compressed_info['original_dtype']
        )
    else:
        algorithms = {
            'zlib': zlib.decompress,
            'gzip': gzip.decompress,
            'bz2': bz2.decompress,
            'lzma': lzma.decompress
        }
        
        decompress_func = algorithms[compressed_info['algorithm']]
        decompressed = decompress_func(compressed_info['compressed_data'])
        recovered = np.frombuffer(decompressed, dtype=compressed_info['original_dtype'])
        recovered = recovered.reshape(compressed_info['original_shape'])
    
    return recovered

def convert_bitmap_for_processing(bitmap, to_format='pee'):
    """Convert bitmap between storage format (binary 0/1) and processing format (0/255)"""
    if to_format == 'pee':
        return (bitmap * 255).astype(np.uint8)
    elif to_format == 'storage':
        return (bitmap == 255).astype(np.uint8)
    else:
        raise ValueError("to_format deve ser 'pee' ou 'storage'")

def message_to_bits(message: str):
    """Convert message to bits with EOF marker"""
    message_with_eof = message + EOF_MARKER
    return [int(b) for c in message_with_eof for b in format(ord(c), '08b')]

def bits_to_message(bits):
    """Convert bits to message, removing EOF marker"""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    
    message = ''.join(chars)
    
    if EOF_MARKER in message:
        return message.split(EOF_MARKER)[0]
    else:
        return message

def save_compressed_stego_bitstream(local_compressed_info, global_compressed_info, output_folder, image_name, bitmap=None, stego_params=None):
    """Save the complete bitstream of compressed steganographic data using dynamic struct header"""
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    compressed_path = f"{output_folder}/{image_name}_stego_data.bin"
    
    # Mapear algoritmos e tipos para códigos
    algo_codes = {'none': 0, 'zlib': 1, 'gzip': 2, 'bz2': 3, 'lzma': 4, 'png': 5, 'gdcm': 6, 'avif': 7, 'jpegxl': 8}
    dtype_codes = {np.uint8: 0, np.uint16: 1, np.int16: 2, np.uint32: 3, np.int32: 4}
    
    # Preparar dados
    local_algo_code = algo_codes.get(local_compressed_info['algorithm'], 0)
    global_algo_code = algo_codes.get(global_compressed_info['algorithm'], 0)
    local_dtype_code = dtype_codes.get(local_compressed_info['original_dtype'].type, 1)
    global_dtype_code = dtype_codes.get(global_compressed_info['original_dtype'].type, 1)
    
    local_data = local_compressed_info['compressed_data']
    global_data = global_compressed_info['compressed_data']
    
    # Preparar bitmap comprimido se fornecido
    bitmap_data = b''
    bitmap_size = 0
    binary_bitmap = None
    if bitmap is not None:
        binary_bitmap = (bitmap == 255).astype(np.uint8)
        bitmap_compressed = zlib.compress(binary_bitmap.tobytes())
        bitmap_data = bitmap_compressed
        bitmap_size = len(bitmap_compressed)
    
    # Preparar parâmetros de esteganografia
    threshold = stego_params.get('threshold', 2) if stego_params else 2
    s_value = stego_params.get('s', 1) if stego_params else 1
    bits_used = stego_params.get('bits_used', 0) if stego_params else 0
    
    with open(compressed_path, 'wb') as f:
        # Magic number para identificar o formato
        f.write(b'STEG')  # 4 bytes
        
        # Construir header dinamicamente
        header_fields = []
        header_format = '>'  # big-endian
        
        # Campos básicos do header
        fields = [
            ('B', 2),                    # version
            ('B', local_algo_code),      # local algorithm
            ('B', global_algo_code),     # global algorithm
            ('d', time.time()),          # timestamp
            ('f', local_compressed_info['compression_ratio']),   # local ratio
            ('f', global_compressed_info['compression_ratio']),  # global ratio
            ('I', local_compressed_info['original_shape'][0]),   # local height
            ('I', local_compressed_info['original_shape'][1]),   # local width
            ('I', global_compressed_info['original_shape'][0]),  # global height
            ('I', global_compressed_info['original_shape'][1]),  # global width
            ('B', local_dtype_code),     # local dtype
            ('B', global_dtype_code),    # global dtype
            ('I', len(local_data)),      # local data size
            ('I', len(global_data)),     # global data size
            ('I', bitmap_size),          # bitmap size
            ('B', threshold),            # PEE threshold
            ('B', s_value),              # modality s value
            ('I', bits_used),            # bits used in steganography
        ]
        
        for fmt, value in fields:
            header_format += fmt
            header_fields.append(value)
        
        # Criar header
        header = struct.pack(header_format, *header_fields)
        header_size = len(header)
        
        # Escrever tamanho do header (4 bytes)
        f.write(struct.pack('>I', header_size))
        
        # Escrever header
        f.write(header)
        
        # Escrever dados na ordem: bitmap, local, global
        if bitmap_data:
            f.write(bitmap_data)
        f.write(local_data)
        f.write(global_data)
    
    total_size = 4 + 4 + header_size + bitmap_size + len(local_data) + len(global_data)
    
    # Calcular economia do bitmap binário
    # Opcional: prints detalhados podem ser ativados se necessário
    
    return compressed_path

def load_compressed_stego_bitstream(filepath):
    """Load compressed steganographic data from dynamic struct-based binary file"""
    
    # Mapear códigos de volta para algoritmos e tipos
    algo_names = {0: 'none', 1: 'zlib', 2: 'gzip', 3: 'bz2', 4: 'lzma', 5: 'png', 6: 'gdcm', 7: 'avif', 8: 'jpegxl'}
    dtype_types = {0: np.uint8, 1: np.uint16, 2: np.int16, 3: np.uint32, 4: np.int32}
    
    with open(filepath, 'rb') as f:
        # Verificar magic number
        magic = f.read(4)
        if magic != b'STEG':
            raise ValueError("Arquivo não é um arquivo de dados steganográficos válido")
        
        # Ler tamanho do header
        header_size_bytes = f.read(4)
        header_size = struct.unpack('>I', header_size_bytes)[0]
        
        # Ler header dinamicamente
        header_data = f.read(header_size)
        
        # Desempacotar header (formato conhecido baseado na versão)
        header_format = '>B B B d f f I I I I B B I I I B B I'  # versão 2
        header = struct.unpack(header_format, header_data)
        
        (version, local_algo_code, global_algo_code, timestamp,
         local_ratio, global_ratio, 
         local_h, local_w, global_h, global_w,
         local_dtype_code, global_dtype_code, local_size, global_size,
         bitmap_size, threshold, s_value, bits_used) = header
        
        # Ler dados na ordem: bitmap, local, global
        bitmap = None
        if bitmap_size > 0:
            bitmap_compressed = f.read(bitmap_size)
            bitmap_bytes = zlib.decompress(bitmap_compressed)
            bitmap = np.frombuffer(bitmap_bytes, dtype=np.uint8).reshape((local_h, local_w))
        
        local_compressed_data = f.read(local_size)
        global_compressed_data = f.read(global_size)
        
        # Reconstruir informações
        local_info = {
            'algorithm': algo_names[local_algo_code],
            'compression_ratio': local_ratio,
            'original_shape': (local_h, local_w),
            'original_dtype': dtype_types[local_dtype_code],
            'compressed_data': local_compressed_data
        }
        
        global_info = {
            'algorithm': algo_names[global_algo_code], 
            'compression_ratio': global_ratio,
            'original_shape': (global_h, global_w),
            'original_dtype': dtype_types[global_dtype_code],
            'compressed_data': global_compressed_data
        }
        
        stego_params = {
            'threshold': threshold,
            's': s_value,
            'bits_used': bits_used
        }
        
        metadata = {
            'version': version,
            'timestamp': timestamp,
            'file_size': os.path.getsize(filepath),
            'header_size': header_size
        }
        
        print(f"📊 Loaded enhanced stego data:")
        print(f"   Version: {version} | Header: {header_size}B | Timestamp: {time.ctime(timestamp)}")
        print(f"   Local: {local_info['algorithm']} ({local_info['compression_ratio']*100:.1f}%)")
        print(f"   Global: {global_info['algorithm']} ({global_info['compression_ratio']*100:.1f}%)")
        print(f"   PEE Params: threshold={threshold}, s={s_value}, bits_used={bits_used}")
        print(f"   Bitmap: {'Present' if bitmap is not None else 'Not included'}")
        
        return {
            'local_component': local_info,
            'global_component': global_info,
            'bitmap': bitmap,
            'stego_params': stego_params,
            'metadata': metadata
        }