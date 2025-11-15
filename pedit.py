from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import rawpy
import numpy as np
import cv2
import io
import os
from pathlib import Path
import uuid
from werkzeug.utils import secure_filename
import imageio
import colour

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '/tmp/photo_uploads'
PROCESSED_FOLDER = '/tmp/photo_processed'
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Allowed extensions
RAW_EXTENSIONS = {'cr2', 'cr3', 'nef', 'arw', 'dng', 'raf', 'orf', 'rw2', 'heic'}
IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tiff', 'tif', 'webp', 'bmp'}
ALLOWED_EXTENSIONS = RAW_EXTENSIONS | IMAGE_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_raw_image(filepath):
    """Load RAW image file and convert to linear RGB in wide color space"""
    with rawpy.imread(filepath) as raw:
        # Use high quality demosaicing with linear output
        rgb = raw.postprocess(
            gamma=(1, 1),  # Linear gamma
            no_auto_bright=True,
            output_bps=16,  # 16-bit output
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.ProPhoto  # Wide gamut
        )
    return rgb

def load_image(filepath):
    """Load standard image formats"""
    ext = filepath.rsplit('.', 1)[1].lower()
    
    if ext in RAW_EXTENSIONS:
        return load_raw_image(filepath)
    elif ext == 'heic':
        # HEIC support via pillow-heif
        img = Image.open(filepath)
        return np.array(img)
    else:
        img = Image.open(filepath)
        if img.mode == 'RGBA':
            # Preserve alpha channel
            return np.array(img)
        else:
            img = img.convert('RGB')
            return np.array(img)

def numpy_to_tensor(image, device):
    """Convert numpy array to PyTorch tensor on GPU"""
    # Normalize to 0-1 range
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    
    # Convert to torch tensor: HWC -> CHW
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

def tensor_to_numpy(tensor, bit_depth=16):
    """Convert PyTorch tensor back to numpy array"""
    # CHW -> HWC
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Clip and convert to appropriate bit depth
    image = np.clip(image, 0, 1)
    if bit_depth == 16:
        image = (image * 65535).astype(np.uint16)
    else:
        image = (image * 255).astype(np.uint8)
    
    return image

@app.route('/')
def index():
    """Serve the main application page"""
    return send_file('static/index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'cuda_available': torch.cuda.is_available(),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        ext = filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")
        file.save(filepath)
        
        # Load and get metadata
        try:
            image = load_image(filepath)
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            
            return jsonify({
                'file_id': file_id,
                'filename': filename,
                'width': int(width),
                'height': int(height),
                'channels': int(channels),
                'dtype': str(image.dtype),
                'is_raw': ext in RAW_EXTENSIONS
            })
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': f'Failed to load image: {str(e)}'}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/process', methods=['POST'])
def process_image():
    data = request.json
    file_id = data.get('file_id')
    operations = data.get('operations', [])
    
    # Find the uploaded file
    uploaded_files = list(Path(app.config['UPLOAD_FOLDER']).glob(f"{file_id}.*"))
    if not uploaded_files:
        return jsonify({'error': 'File not found'}), 404
    
    filepath = str(uploaded_files[0])
    
    try:
        # Load image
        image = load_image(filepath)
        
        # Convert to tensor and move to GPU
        tensor = numpy_to_tensor(image, device)
        
        # Apply operations on GPU
        for op in operations:
            op_type = op.get('type')
            
            if op_type == 'brightness':
                factor = op.get('value', 1.0)
                tensor = torch.clamp(tensor * factor, 0, 1)
            
            elif op_type == 'contrast':
                factor = op.get('value', 1.0)
                mean = tensor.mean()
                tensor = torch.clamp((tensor - mean) * factor + mean, 0, 1)
            
            elif op_type == 'saturation':
                factor = op.get('value', 1.0)
                # Convert to grayscale weights
                gray = 0.299 * tensor[:, 0:1, :, :] + 0.587 * tensor[:, 1:2, :, :] + 0.114 * tensor[:, 2:3, :, :]
                tensor = torch.clamp(gray + (tensor - gray) * factor, 0, 1)
            
            elif op_type == 'exposure':
                stops = op.get('value', 0.0)
                factor = 2 ** stops
                tensor = torch.clamp(tensor * factor, 0, 1)
            
            elif op_type == 'highlights':
                amount = op.get('value', 0.0)
                mask = (tensor > 0.5).float()
                adjustment = amount * mask * (tensor - 0.5)
                tensor = torch.clamp(tensor + adjustment, 0, 1)
            
            elif op_type == 'shadows':
                amount = op.get('value', 0.0)
                mask = (tensor <= 0.5).float()
                adjustment = amount * mask * (0.5 - tensor)
                tensor = torch.clamp(tensor + adjustment, 0, 1)
            
            elif op_type == 'temperature':
                # Color temperature adjustment
                temp = op.get('value', 0.0)
                if temp > 0:  # Warmer
                    tensor[:, 0, :, :] = torch.clamp(tensor[:, 0, :, :] * (1 + temp * 0.1), 0, 1)
                    tensor[:, 2, :, :] = torch.clamp(tensor[:, 2, :, :] * (1 - temp * 0.1), 0, 1)
                else:  # Cooler
                    tensor[:, 0, :, :] = torch.clamp(tensor[:, 0, :, :] * (1 + temp * 0.1), 0, 1)
                    tensor[:, 2, :, :] = torch.clamp(tensor[:, 2, :, :] * (1 - temp * 0.1), 0, 1)
            
            elif op_type == 'tint':
                tint = op.get('value', 0.0)
                tensor[:, 1, :, :] = torch.clamp(tensor[:, 1, :, :] * (1 + tint * 0.1), 0, 1)
            
            elif op_type == 'sharpen':
                strength = op.get('value', 0.5)
                # Simple unsharp mask on GPU
                kernel = torch.tensor([
                    [[-1, -1, -1],
                     [-1,  9, -1],
                     [-1, -1, -1]]
                ]).float().to(device) / 9.0
                kernel = kernel.unsqueeze(0).repeat(3, 1, 1, 1)
                
                pad = torch.nn.ReflectionPad2d(1)
                tensor_padded = pad(tensor)
                sharpened = torch.nn.functional.conv2d(tensor_padded, kernel, groups=3)
                tensor = torch.clamp(tensor + (sharpened - tensor) * strength, 0, 1)
            
            elif op_type == 'blur':
                radius = int(op.get('value', 5))
                if radius > 0:
                    # Gaussian blur
                    kernel_size = radius * 2 + 1
                    sigma = radius / 3.0
                    
                    # Create 1D Gaussian kernel
                    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
                    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
                    kernel_1d = kernel_1d / kernel_1d.sum()
                    
                    # Apply separable convolution
                    kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(3, 1, 1, 1)
                    kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(3, 1, 1, 1)
                    
                    pad_h = torch.nn.ReflectionPad2d((radius, radius, 0, 0))
                    pad_v = torch.nn.ReflectionPad2d((0, 0, radius, radius))
                    
                    tensor = pad_h(tensor)
                    tensor = torch.nn.functional.conv2d(tensor, kernel_h, groups=3)
                    tensor = pad_v(tensor)
                    tensor = torch.nn.functional.conv2d(tensor, kernel_v, groups=3)
            
            elif op_type == 'rotate':
                angle = op.get('value', 0)
                if angle != 0:
                    # Convert to numpy for rotation
                    img_np = tensor_to_numpy(tensor, bit_depth=8)
                    center = (img_np.shape[1] // 2, img_np.shape[0] // 2)
                    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img_np = cv2.warpAffine(img_np, matrix, (img_np.shape[1], img_np.shape[0]))
                    tensor = numpy_to_tensor(img_np, device)
            
            elif op_type == 'flip':
                direction = op.get('value', 'horizontal')
                if direction == 'horizontal':
                    tensor = torch.flip(tensor, [3])
                elif direction == 'vertical':
                    tensor = torch.flip(tensor, [2])
            
            elif op_type == 'crop':
                x = int(op.get('x', 0))
                y = int(op.get('y', 0))
                width = int(op.get('width', tensor.shape[3]))
                height = int(op.get('height', tensor.shape[2]))
                tensor = tensor[:, :, y:y+height, x:x+width]
        
        # Convert back to numpy
        bit_depth = 16 if data.get('high_quality', True) else 8
        result_image = tensor_to_numpy(tensor, bit_depth=bit_depth)
        
        # Save processed image temporarily
        output_id = str(uuid.uuid4())
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{output_id}.npy")
        np.save(output_path, result_image)
        
        return jsonify({
            'output_id': output_id,
            'width': result_image.shape[1],
            'height': result_image.shape[0],
            'message': 'Processing completed'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/export', methods=['POST'])
def export_image():
    data = request.json
    output_id = data.get('output_id')
    format_type = data.get('format', 'jpeg').lower()
    quality = data.get('quality', 95)
    
    # Load processed image
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{output_id}.npy")
    if not os.path.exists(output_path):
        return jsonify({'error': 'Processed image not found'}), 404
    
    try:
        image = np.load(output_path)
        
        # Export based on format
        export_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{output_id}.{format_type}")
        
        if format_type == 'jpeg' or format_type == 'jpg':
            # Convert to 8-bit if needed
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)
            img = Image.fromarray(image)
            img.save(export_path, 'JPEG', quality=quality, optimize=True)
        
        elif format_type == 'png':
            # PNG supports 16-bit
            img = Image.fromarray(image)
            img.save(export_path, 'PNG', compress_level=6)
        
        elif format_type == 'avif':
            # Convert to 8-bit for AVIF
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)
            imageio.imwrite(export_path, image, format='AVIF', quality=quality)
        
        elif format_type == 'exr' or format_type == 'openexr':
            # OpenEXR - convert to float32
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            elif image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            imageio.imwrite(export_path.replace(format_type, 'exr'), image, format='EXR')
            export_path = export_path.replace(format_type, 'exr')
        
        elif format_type == 'tiff' or format_type == 'tif':
            img = Image.fromarray(image)
            img.save(export_path, 'TIFF')
        
        else:
            return jsonify({'error': f'Unsupported format: {format_type}'}), 400
        
        return send_file(export_path, as_attachment=True, download_name=f"edited.{format_type}")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/api/preview', methods=['POST'])
def get_preview():
    data = request.json
    output_id = data.get('output_id')
    max_size = data.get('max_size', 1200)
    
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{output_id}.npy")
    if not os.path.exists(output_path):
        return jsonify({'error': 'Processed image not found'}), 404
    
    try:
        image = np.load(output_path)
        
        # Convert to 8-bit for preview
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        
        # Resize for preview
        height, width = image.shape[:2]
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to JPEG for preview
        img = Image.fromarray(image)
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=90)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({'error': f'Preview generation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
