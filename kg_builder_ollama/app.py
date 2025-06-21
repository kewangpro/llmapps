"""
Paragraph to Knowledge Graph Converter
=====================================

This project converts text paragraphs into knowledge graphs using multiple approaches:
1. Ollama LLM for entity and relationship extraction
2. spaCy for NLP preprocessing and named entity recognition
3. NetworkX for graph visualization and analysis
4. Optional: OpenIE for triple extraction
"""

import requests
from dataclasses import dataclass
import sys
import subprocess
from kg import KnowledgeGraphExtractor

def setup_dependencies():
    """Helper function to setup required dependencies"""
    print("🔧 Setting up dependencies...")
    
    # Check if spaCy is installed
    try:
        import spacy
        print("✅ spaCy is installed")
        
        # Check if model is available
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model 'en_core_web_sm' is available")
        except OSError:
            print("❌ spaCy model 'en_core_web_sm' not found")
            print("📦 Installing spaCy model...")
            try:
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                             check=True)
                print("✅ spaCy model installed successfully!")
            except subprocess.CalledProcessError:
                print("❌ Failed to install spaCy model")
                print("🛠️ Please run manually: python -m spacy download en_core_web_sm")
    
    except ImportError:
        print("❌ spaCy not installed")
        print("📦 Installing spaCy...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "spacy"], check=True)
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            print("✅ spaCy installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install spaCy")
            print("🛠️ Please run manually: pip install spacy && python -m spacy download en_core_web_sm")
    
    # Check Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            models = response.json().get('models', [])
            if models:
                model_names = [model['name'] for model in models]
                print(f"📋 Available models: {model_names}")
                if 'llama3.2:latest' not in model_names:
                    print("⚠️ llama3.2:latest not found. Install with: ollama pull llama3.2:latest")
            else:
                print("⚠️ No models found. Install llama3.2 with: ollama pull llama3.2:latest")
        else:
            print("❌ Ollama is not responding")
    except requests.RequestException:
        print("❌ Ollama is not running")
        print("🛠️ Please start Ollama: ollama serve")
    
    print("🎯 Setup complete!")

def main():
    """Example usage of the Knowledge Graph Extractor"""
    
    print("🚀 Knowledge Graph Extractor Demo")
    print("=" * 50)
    
    # Check setup
    setup_dependencies()
    print()
    
    # Sample text
    sample_text = """
    Alice works for Acme Corp. Bob works for Globex. Alice knows Bob. Charlie founded Acme Corp. 
    Globex acquired Acme Corp. Both Alice and Bob report to Charlie.
    """
    
    # Initialize extractor
    extractor = KnowledgeGraphExtractor(ollama_model="llama3.2:latest")
    
    print("📝 Input Text:")
    print(sample_text)
    
    # Extract using different methods
    print("\n🤖 Extracting with Ollama (llama3.2:latest)...")
    try:
        ollama_triples = extractor.extract_with_ollama(sample_text)
        print(f"   Found {len(ollama_triples)} triples from Ollama")
    except Exception as e:
        print(f"   ⚠️ Ollama extraction failed: {e}")
        ollama_triples = []
    
    print("\n🔍 Extracting with spaCy/Patterns...")
    spacy_triples = extractor.extract_with_spacy(sample_text)
    print(f"   Found {len(spacy_triples)} triples from spaCy/Patterns")
    
    # Combine results
    all_triples = ollama_triples + spacy_triples
    
    if all_triples:
        print(f"\n📊 Total {len(all_triples)} triples found:")
        for i, triple in enumerate(all_triples[:10], 1):  # Show first 10
            print(f"{i}. {triple.subject} → {triple.predicate} → {triple.object}")
        
        # Build and analyze graph
        print("\n🕸️ Building knowledge graph...")
        graph = extractor.build_graph(all_triples)
        
        print("\n📈 Graph Analysis:")
        analysis = extractor.analyze_graph()
        for key, value in analysis.items():
            print(f"   {key}: {value}")
        
        # Visualizations
        print("\n🎨 Creating visualizations...")
        try:
            extractor.visualize_graph_matplotlib()
        except Exception as e:
            print(f"   ⚠️ Visualization failed: {e}")
        
        # Export
        print("\n💾 Exporting graph...")
        json_export = extractor.export_graph("json")
        print("   JSON export preview:")
        print("   " + (json_export[:200] + "..." if len(json_export) > 200 else json_export))
    
    else:
        print("⚠️ No triples extracted.")
        print("💡 Tips:")
        print("   - Make sure Ollama is running: ollama serve")
        print("   - Install llama3.2: ollama pull llama3.2:latest")
        print("   - Check spaCy installation: python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main()