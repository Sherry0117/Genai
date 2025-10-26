#!/usr/bin/env python3
"""
Test script for the CNN API endpoints.
This script tests the CNN classification functionality in the FastAPI.
"""

import requests
import base64
import json
from PIL import Image
import io
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000"

def create_test_image():
    """Create a simple test image for classification."""
    # Create a random 64x64 RGB image
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_base64

def test_cnn_info():
    """Test the CNN info endpoint."""
    print("Testing CNN info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/cnn_info")
        if response.status_code == 200:
            info = response.json()
            print("✓ CNN Info retrieved successfully:")
            print(f"  Model: {info['model_name']}")
            print(f"  Architecture: {info['architecture']}")
            print(f"  Input size: {info['input_size']}")
            print(f"  Classes: {info['classes']}")
            print(f"  Parameters: {info['parameters']:,}")
        else:
            print(f"✗ Failed to get CNN info: {response.status_code}")
    except Exception as e:
        print(f"✗ Error testing CNN info: {e}")

def test_classify_image():
    """Test the image classification endpoint."""
    print("\nTesting image classification endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Prepare request
        request_data = {
            "image": test_image,
            "top_k": 3
        }
        
        # Send request
        response = requests.post(f"{BASE_URL}/classify_image", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Image classification successful:")
            print(f"  Top {result['top_k']} predictions:")
            for i, pred in enumerate(result['predictions']):
                print(f"    {i+1}. {pred['class']}: {pred['confidence']:.4f}")
        else:
            print(f"✗ Failed to classify image: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Error testing image classification: {e}")

def test_api_health():
    """Test if the API is running."""
    print("Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✓ API is running")
            return True
        else:
            print(f"✗ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API is not running: {e}")
        return False

def main():
    print("CNN API Test Script")
    print("=" * 50)
    
    # Test API health
    if not test_api_health():
        print("\nPlease start the API server first:")
        print("  uvicorn app.main:app --reload")
        return
    
    # Test CNN info
    test_cnn_info()
    
    # Test image classification
    test_classify_image()
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    main()
