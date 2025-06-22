import requests

def test_ollama():
    base_url = "http://localhost:11434"
    
    # Test 1: Check if Ollama is running
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        print(f"✓ Ollama is running on {base_url}")
        
        models = response.json().get('models', [])
        if models:
            print("Available models:")
            for model in models:
                print(f"  - {model['name']}")
        else:
            print("❌ No models installed")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("   Make sure to run: ollama serve")
        return False
    except requests.exceptions.Timeout:
        print("❌ Ollama connection timeout")
        return False
    
    # Test 2: Try to generate text with the first available model
    if models:
        model_name = models[0]['name']
        print(f"\nTesting generation with model: {model_name}")
        
        try:
            response = requests.post(f"{base_url}/api/generate", 
                json={
                    "model": model_name,
                    "prompt": "Hello! Respond with just 'Hi there!'",
                    "stream": False
                }, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '')
                print(f"✓ Generation successful: {result.strip()}")
                return True
            else:
                print(f"❌ Generation failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Generation timeout (model might be loading)")
            return False
    
    return False

if __name__ == '__main__':
    print("Testing Ollama connection...")
    success = test_ollama()
    
    if success:
        print("\n✓ Ollama is working correctly!")
        print("You can now test the MCP client.")
    else:
        print("\n❌ Ollama setup incomplete.")
        print("Steps to fix:")
        print("1. Run: ollama serve")
        print("2. Run: ollama pull llama3.2:1b")
        print("3. Test again with this script")