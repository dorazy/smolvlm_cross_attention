from flask import Flask, render_template, jsonify, request
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.colors import LinearSegmentedColormap
import math
import json
import os
import shutil
import psutil
import gc

app = Flask(__name__)

# Global variables to store loaded data
decoded_tokens = None
current_image = None
smolvlm_patch_size = 512
smolvlm_token_patch_size = 8

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size (physical memory)
        'vms': memory_info.vms / 1024 / 1024   # Virtual Memory Size
    }

def log_memory(location):
    """Log memory usage at a specific location"""
    memory = get_memory_usage()
    print(f"MEMORY [{location}]: RSS={memory['rss']:.1f}MB, VMS={memory['vms']:.1f}MB")

def load_data():
    """Load the decoded tokens only - attention data will be loaded per layer"""
    global decoded_tokens
    log_memory("load_data_start")
    try:
        decoded_tokens = np.load("decoded_tokens.npy", allow_pickle=True).tolist()
        log_memory("load_data_after_tokens")
        return True
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return False

def load_attention_layer(layer_index):
    """Load attention data for a specific layer"""
    try:
        log_memory(f"before_load_layer_{layer_index}")
        filename = f"attentions_5/attention_fp16_rounded_layer_{layer_index}.npz"
        
        # Check if file exists
        if not os.path.exists(filename):
            return None
            
        attention_layer = np.load(filename)['attention']
        log_memory(f"after_load_layer_{layer_index}")
        return attention_layer
    except FileNotFoundError as e:
        return None
    except Exception as e:
        return None

def load_image(image_path):
    """Load and process the image"""
    global current_image
    try:
        current_image = Image.open(image_path).convert("RGB")
        return True
    except Exception as e:
        print(f"Error loading image: {e}")
        return False

def get_word_attention(word, decoded_tokens, layer_index, word_index=None):
    """Get attention for a specific word and layer"""
    try:
        # Load the specific layer's attention data
        attention_layer = load_attention_layer(layer_index)
        if attention_layer is None:
            return None, None
            
        if word_index is not None:
            # Use provided index directly
            if 0 <= word_index < len(decoded_tokens):
                word_attn = attention_layer[0, :, word_index, :].copy()  # Copy to new array
                del attention_layer  # Immediately free the large layer data
                gc.collect()  # Force garbage collection
                log_memory("after_extracting_word_attention")
                return word_attn, word_index
            else:
                del attention_layer  # Free memory even on error
                return None, None
        else:
            # Fallback to searching for the word
            word_index = decoded_tokens.index(word)
            word_attn = attention_layer[0, :, word_index, :].copy()  # Copy to new array
            del attention_layer  # Immediately free the large layer data
            gc.collect()  # Force garbage collection
            log_memory("after_extracting_word_attention")
            return word_attn, word_index
    except (ValueError, IndexError) as e:
        return None, None

def build_image_patch_dict(decoded_tokens):
    """Build dictionary mapping patch tokens to indices"""
    patch_indices = {}
    for i, word in enumerate(decoded_tokens):
        if 'row' in word or 'global' in word:
            patch_indices[word] = [i+1+j for j in range(64)]
    return patch_indices

def reshape_patch(patch):
    """Reshape patch to full image dimensions"""
    expanded_size = smolvlm_token_patch_size**2
    expanded_patch = np.zeros((smolvlm_patch_size, smolvlm_patch_size), dtype=np.float32)
    square_patch = patch.reshape(smolvlm_token_patch_size, smolvlm_token_patch_size)
    
    for i in range(smolvlm_token_patch_size):
        for j in range(smolvlm_token_patch_size):
            # Direct assignment instead of creating temporary expanded_section
            value = square_patch[i, j]
            expanded_patch[i*expanded_size:i*expanded_size + expanded_size, 
                          j*expanded_size:j*expanded_size+expanded_size] = value
    
    del square_patch  # Clean up intermediate array
    return expanded_patch

def get_attn_map(word_attn, head, patch_indices, num_rows=3, num_cols=4):
    """Generate attention map for given head (layer already selected in word_attn)"""
    attn = word_attn[head, :]
    final_heatmap = np.zeros((512 * num_rows, 512 * num_cols), dtype=np.float32)  # Use float32 to save memory
    
    for i in range(num_rows):
        for j in range(num_cols):
            patch_key = f"<row_{i+1}_col_{j+1}>"
            if patch_key in patch_indices:
                indices = patch_indices[patch_key]
                attn_patch = attn[indices]
                p = reshape_patch(attn_patch)
                final_heatmap[i*512:i*512 + 512, j*512: j*512 + 512] = p
                del attn_patch, p  # Delete intermediate arrays immediately
    
    del attn  # Free the attention array
    return final_heatmap

def find_subarray_indices(arr, subarr):
    n, m = len(arr), len(subarr)
    for i in range(n - m + 1):
        if arr[i:i + m] == subarr:
            return i, i + m - 1  # return start and end (exclusive)
    return None  # not found

def create_attention_heatmap(final_heatmap, image, threshold=0.0):
    """Create attention heatmap and return only min/max values for color scale"""
    image_w, image_h = image.size
    resized_w = math.ceil(image_w / smolvlm_patch_size) * smolvlm_patch_size
    resized_h = math.ceil(image_h / smolvlm_patch_size) * smolvlm_patch_size
    
    log_memory("before_heatmap_normalization")
    
    # Capture min/max before normalization (only values actually used by frontend)
    original_min = float(final_heatmap.min())
    original_max = float(final_heatmap.max())
    
    # Apply threshold filter - set values below threshold to 0
    if threshold > 0.0:
        final_heatmap[final_heatmap < threshold] = 0.0
    
    # Check if heatmap has valid range
    heatmap_min = final_heatmap.min()
    heatmap_max = final_heatmap.max()
    heatmap_range = heatmap_max - heatmap_min
    
    if heatmap_range == 0:
        # In-place modification to save memory
        final_heatmap.fill(0.5)
        normalized_heatmap = final_heatmap
    else:
        # In-place normalization to save memory
        final_heatmap -= heatmap_min
        final_heatmap /= heatmap_range
        normalized_heatmap = final_heatmap
    
    log_memory("after_heatmap_normalization")
    
    # Create heatmap
    fig = plt.figure(figsize=(resized_w/100, resized_h/100), dpi=100)
    plt.imshow(normalized_heatmap, cmap='hot', alpha=0.7, interpolation='nearest')
    plt.axis("off")
    plt.gca().set_position([0, 0, 1, 1])
    
    log_memory("after_matplotlib_setup")
    
    # Save heatmap to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, transparent=True, dpi=100)
    buffer.seek(0)
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close(fig)
    plt.clf()
    
    # Aggressive cleanup
    del normalized_heatmap, final_heatmap
    gc.collect()
    
    log_memory("after_matplotlib_cleanup")
    
    # Return only the values used by frontend
    stats = {
        'min': original_min,
        'max': original_max
    }
    
    return heatmap_base64, stats, resized_w, resized_h

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/initialize')
def initialize():
    """Initialize the application with data"""
    log_memory("initialize_start")
    success = load_data()
    if success:
        # Load local image - try both locations
        image_paths = ["image.png", "static/image.png"]
        image_success = False
        image_url = None
        
        for image_path in image_paths:
            if os.path.exists(image_path):
                image_success = load_image(image_path)
                if image_success:
                    # If image is in root directory, copy it to static for serving
                    if image_path == "image.png":
                        import shutil
                        shutil.copy("image.png", "static/image.png")
                    image_url = '/static/image.png'
                    break
        
        if image_success:
            # Parse conversation structure from tokens
            prompt_text = ""
            assistant_tokens = []
            
            try:
                # Find conversation markers
                user_index = None
                assistant_index = None
                
                # Look for "User" marker
                for i, token in enumerate(decoded_tokens):
                    if token.strip().lower() == 'user':
                        user_index = i
                        break
                
                # Look for "Ass", "istant", ":" sequence
                for i in range(len(decoded_tokens) - 2):
                    if (decoded_tokens[i] == 'Ass' and 
                        decoded_tokens[i + 1] == 'istant' and 
                        decoded_tokens[i + 2] == ':'):
                        assistant_index = i + 2  # Point to the ':' token
                        break
                
                # Use hardcoded prompt
                prompt_text = "Can you describe this image?"
                print(f"Using hardcoded prompt: '{prompt_text}'")
                
                # Extract assistant response tokens (after assistant marker)
                if assistant_index is not None:
                    print(f"Found assistant at index {assistant_index}, extracting response tokens...")
                    for i in range(assistant_index + 1, len(decoded_tokens)):
                        token = decoded_tokens[i]
                        # Filter out patch tokens but keep text tokens and newlines
                        if not ('row' in token or 'global' in token or token.startswith('<')):
                            if token == '\n':
                                # Include newlines but mark as non-selectable
                                assistant_tokens.append({
                                    'text': token,
                                    'index': i,
                                    'is_selectable': False
                                })
                            elif token.strip() != '':
                                # Include regular text tokens as selectable
                                assistant_tokens.append({
                                    'text': token,  # Keep original token with whitespace
                                    'index': i,
                                    'is_selectable': True
                                })
                
                # Fallback if structure not found
                if not prompt_text:
                    prompt_text = "Can you describe this image?"
                
                if not assistant_tokens:
                    print("No assistant tokens found, using fallback to all content tokens")
                    # Fallback to all content tokens
                    for i, token in enumerate(decoded_tokens):
                        if not ('row' in token or 'global' in token or token.startswith('<')):
                            if token == '\n':
                                assistant_tokens.append({
                                    'text': token,
                                    'index': i,
                                    'is_selectable': False
                                })
                            elif token.strip() != '':
                                assistant_tokens.append({
                                    'text': token,
                                    'index': i,
                                    'is_selectable': True
                                })
                else:
                    print(f"Found {len(assistant_tokens)} assistant response tokens")
            
            except Exception as e:
                print(f"Error parsing conversation: {e}")
                # Fallback
                prompt_text = "describe this image"
                assistant_tokens = []
                for i, token in enumerate(decoded_tokens):
                    if not ('row' in token or 'global' in token or token.startswith('<')):
                        if token == '\n':
                            assistant_tokens.append({
                                'text': token,
                                'index': i,
                                'is_selectable': False
                            })
                        elif token.strip() != '':
                            assistant_tokens.append({
                                'text': token,
                                'index': i,
                                'is_selectable': True
                            })
            
            # Since we now load layers on-demand, we can assume 30 layers and get heads from first layer
            num_layers = 30
            num_heads = 0
            
            # Try to load first layer to get number of heads
            try:
                first_layer = load_attention_layer(0)
                if first_layer is not None:
                    num_heads = first_layer.shape[1]
            except Exception as e:
                print(f"Could not determine number of heads: {e}")
                num_heads = 9  # Default fallback
            
            return jsonify({
                'success': True,
                'prompt': prompt_text,
                'tokens': assistant_tokens,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'image_url': image_url
            })
        else:
            return jsonify({'success': False, 'error': 'Could not find image.png in current directory or static/ directory'})
    
    return jsonify({'success': False, 'error': 'Failed to load data'})

@app.route('/api/generate_attention_map', methods=['POST'])
def generate_attention_map():
    """Generate attention map for given parameters"""
    data = request.json
    word = data.get('word')
    word_index = data.get('word_index')
    layer = int(data.get('layer', 0))
    head = int(data.get('head', 0))
    threshold = float(data.get('threshold', 0.0))
    
    if decoded_tokens is None or current_image is None:
        return jsonify({'success': False, 'error': 'Data not loaded'})
    
    # Get word attention for the specific layer
    word_attn, word_index = get_word_attention(word, decoded_tokens, layer, word_index)
    if word_attn is None:
        return jsonify({'success': False, 'error': f'Word "{word}" not found in tokens or could not load layer {layer}'})
    
    # Build patch dictionary
    patch_indices = build_image_patch_dict(decoded_tokens)
    
    # Generate attention map
    try:
        attn_map = get_attn_map(word_attn, head, patch_indices)
        
        # Free word_attn immediately after use
        del word_attn
        gc.collect()
        
        heatmap_base64, stats, resized_w, resized_h = create_attention_heatmap(attn_map, current_image, threshold)
        # Note: attn_map is deleted inside create_attention_heatmap now
        
        # Final cleanup
        del patch_indices
        gc.collect()
        
        return jsonify({
            'success': True,
            'heatmap': heatmap_base64,
            'stats': stats,
            'heatmap_width': resized_w,
            'heatmap_height': resized_h,
            'word_index': word_index,
            'layer': layer,
            'head': head,
            'threshold': threshold
        })
    except Exception as e:
        # Cleanup on error
        try:
            del word_attn, attn_map, patch_indices
        except:
            pass
        gc.collect()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Set matplotlib to use minimal memory
    matplotlib.rcParams['figure.max_open_warning'] = 0
    plt.ioff()  # Turn off interactive mode to save memory
    
    app.run(debug=True, host='0.0.0.0', port=5001) 
