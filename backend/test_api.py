#!/usr/bin/env python3
"""
Simple test script for the iSpy FastAPI backend.
Run this to test API endpoints without the frontend.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    """Test the root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print("✅ Root endpoint:", response.json())
        return True
    except Exception as e:
        print("❌ Root endpoint failed:", str(e))
        return False

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("✅ Health endpoint:", response.json())
        return True
    except Exception as e:
        print("❌ Health endpoint failed:", str(e))
        return False

def test_chat():
    """Test the chat endpoint"""
    try:
        data = {"message": "Hello, can you help me?"}
        response = requests.post(
            f"{BASE_URL}/api/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        result = response.json()
        print("✅ Chat endpoint response:", result)
        return True
    except Exception as e:
        print("❌ Chat endpoint failed:", str(e))
        return False

def main():
    print("🧪 Testing iSpy Backend API...")
    print("Make sure the backend server is running on http://localhost:8000")
    print()
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Chat Endpoint", test_chat),
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"Testing {name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check if the server is running and configured correctly.")

if __name__ == "__main__":
    main()