#!/usr/bin/env python3
"""
Test script to verify attention data format for VLM Attention Visualizer
Run this script to check if your data files are in the correct format.
"""

import numpy as np
import sys
import os

def test_attention_data():
    """Test the attention array format"""
    print("🔍 Testing attention_array.npy...")
    
    try:
        attention = np.load("attention_array.npy", allow_pickle=True)
        print(f"✅ Successfully loaded attention_array.npy")
        print(f"   Shape: {attention.shape if hasattr(attention, 'shape') else 'List/Object array'}")
        print(f"   Type: {type(attention)}")
        
        if isinstance(attention, np.ndarray) and len(attention) > 0:
            first_layer = attention[0]
            print(f"   First layer shape: {first_layer.shape}")
            print(f"   Expected format: [batch, heads, seq_len, seq_len]")
            
            if len(first_layer.shape) == 4:
                batch, heads, seq_len1, seq_len2 = first_layer.shape
                print(f"   Batch size: {batch}")
                print(f"   Number of heads: {heads}")
                print(f"   Sequence length: {seq_len1} x {seq_len2}")
                
                if seq_len1 == seq_len2:
                    print("✅ Attention matrix is square (good)")
                else:
                    print("⚠️  Attention matrix is not square")
            else:
                print(f"⚠️  Unexpected shape for attention layer: {first_layer.shape}")
        
        print(f"   Number of layers: {len(attention)}")
        
    except FileNotFoundError:
        print("❌ attention_array.npy not found in current directory")
        return False
    except Exception as e:
        print(f"❌ Error loading attention_array.npy: {e}")
        return False
    
    return True

def test_tokens_data():
    """Test the decoded tokens format"""
    print("\n🔍 Testing decoded_tokens.npy...")
    
    try:
        tokens = np.load("decoded_tokens.npy").tolist()
        print(f"✅ Successfully loaded decoded_tokens.npy")
        print(f"   Number of tokens: {len(tokens)}")
        print(f"   Type: {type(tokens)}")
        
        # Show first few tokens
        if len(tokens) > 0:
            sample_tokens = tokens[:10] if len(tokens) > 10 else tokens
            print(f"   Sample tokens: {sample_tokens}")
            
            # Check for patch tokens
            patch_tokens = [token for token in tokens if 'row' in str(token) or 'global' in str(token)]
            content_tokens = [token for token in tokens if not ('row' in str(token) or 'global' in str(token) or '<' in str(token))]
            
            print(f"   Patch tokens found: {len(patch_tokens)}")
            print(f"   Content tokens found: {len(content_tokens)}")
            
            if len(content_tokens) > 0:
                print(f"   Sample content tokens: {content_tokens[:5]}")
            
        else:
            print("⚠️  No tokens found in the array")
            
    except FileNotFoundError:
        print("❌ decoded_tokens.npy not found in current directory")
        return False
    except Exception as e:
        print(f"❌ Error loading decoded_tokens.npy: {e}")
        return False
    
    return True

def test_compatibility():
    """Test if attention and tokens are compatible"""
    print("\n🔍 Testing compatibility between attention and tokens...")
    
    try:
        attention = np.load("attention_array.npy", allow_pickle=True)
        tokens = np.load("decoded_tokens.npy").tolist()
        
        if len(attention) > 0:
            first_layer = attention[0]
            if len(first_layer.shape) == 4:
                seq_len = first_layer.shape[2]  # sequence length
                token_count = len(tokens)
                
                print(f"   Attention sequence length: {seq_len}")
                print(f"   Number of tokens: {token_count}")
                
                if seq_len == token_count:
                    print("✅ Sequence lengths match perfectly")
                elif abs(seq_len - token_count) <= 2:
                    print("✅ Sequence lengths are close (likely compatible)")
                else:
                    print(f"⚠️  Sequence length mismatch: {seq_len} vs {token_count}")
                    print("   This might cause issues with token selection")
            
    except Exception as e:
        print(f"❌ Error checking compatibility: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🧪 VLM Attention Visualizer - Data Format Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # List files in current directory
    files = os.listdir('.')
    npy_files = [f for f in files if f.endswith('.npy')]
    print(f"NumPy files found: {npy_files}")
    
    print("\n" + "=" * 50)
    
    # Run tests
    attention_ok = test_attention_data()
    tokens_ok = test_tokens_data()
    
    if attention_ok and tokens_ok:
        compatibility_ok = test_compatibility()
    else:
        compatibility_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Attention data: {'✅ OK' if attention_ok else '❌ Failed'}")
    print(f"   Token data: {'✅ OK' if tokens_ok else '❌ Failed'}")
    print(f"   Compatibility: {'✅ OK' if compatibility_ok else '❌ Failed'}")
    
    if attention_ok and tokens_ok and compatibility_ok:
        print("\n🎉 All tests passed! Your data should work with the visualizer.")
        print("   Run 'python app.py' to start the web application.")
    else:
        print("\n⚠️  Some tests failed. Please check your data files.")
        print("   Refer to the README.md for data format requirements.")
    
    return attention_ok and tokens_ok and compatibility_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 