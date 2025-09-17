import numpy as np
from PIL import Image
import pydicom
import zlib, gzip, bz2, lzma
import os
import time
import struct

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
        print(f"📁 Pasta criada: {folder_name}")
    
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

def compress_image_with_algorithm(image_array, algorithm):
    """Compress image using a specific algorithm"""
    original_size = image_size(image_array)
    print(f"📊 Compressão {algorithm} - Original: {original_size:.1f} KB")
    
    flat = image_array.flatten()
    byte_data = flat.tobytes()
    
    algorithms = {
        'zlib': zlib.compress,
        'gzip': gzip.compress,
        'bz2': bz2.compress,
        'lzma': lzma.compress
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"Algoritmo '{algorithm}' não suportado. Use: {list(algorithms.keys())}")
    
    try:
        compress_func = algorithms[algorithm]
        compressed = compress_func(byte_data)
        compression_ratio = len(compressed) / len(byte_data)
        
        print(f"✅ {algorithm}: {len(compressed)} bytes ({compression_ratio*100:.1f}%) - {len(compressed)/1024:.1f} KB")
        
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
    algo_codes = {'none': 0, 'zlib': 1, 'gzip': 2, 'bz2': 3, 'lzma': 4}
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
    if bitmap is not None:
        original_bitmap_size = bitmap.nbytes
        binary_bitmap_size = len(binary_bitmap.tobytes())
        compression_savings = (1 - bitmap_size / original_bitmap_size) * 100
        
        print(f"✅ Enhanced struct-based stego data saved: {compressed_path}")
        print(f"   📊 Magic: 4B | Header Size: 4B | Header: {header_size}B")
        print(f"   📊 Bitmap: {original_bitmap_size}B → {binary_bitmap_size}B → {bitmap_size}B compressed ({compression_savings:.1f}% savings)")
        print(f"   📊 Local: {len(local_data)}B | Global: {len(global_data)}B | Total: {total_size}B")
        print(f"   🔧 PEE Params: threshold={threshold}, s={s_value}, bits_used={bits_used}")
    else:
        print(f"✅ Enhanced struct-based stego data saved: {compressed_path}")
        print(f"   📊 Magic: 4B | Header Size: 4B | Header: {header_size}B | Bitmap: None")
        print(f"   📊 Local: {len(local_data)}B | Global: {len(global_data)}B | Total: {total_size}B")
        print(f"   🔧 PEE Params: threshold={threshold}, s={s_value}, bits_used={bits_used}")
    
    return compressed_path

def load_compressed_stego_bitstream(filepath):
    """Load compressed steganographic data from dynamic struct-based binary file"""
    
    # Mapear códigos de volta para algoritmos e tipos
    algo_names = {0: 'none', 1: 'zlib', 2: 'gzip', 3: 'bz2', 4: 'lzma'}
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