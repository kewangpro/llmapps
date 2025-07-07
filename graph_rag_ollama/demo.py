"""
Paragraph to Knowledge Graph Converter
=====================================

This project converts text paragraphs into knowledge graphs using multiple approaches:
1. Ollama LLM for entity and relationship extraction
2. spaCy for NLP preprocessing and named entity recognition
3. NetworkX for graph visualization and analysis
4. Optional: OpenIE for triple extraction
"""

import json
import re
import requests
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
import sys
import subprocess

# Try to import spacy with error handling
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    print("⚠️ spaCy not installed. Install with: pip install spacy")
    SPACY_AVAILABLE = False
    spacy = None

@dataclass
class Triple:
    """Represents a knowledge triple (subject, predicate, object)"""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0

@dataclass
class Entity:
    """Represents an entity with its type and mentions"""
    name: str
    entity_type: str
    mentions: List[str]
    confidence: float = 0.0

class KnowledgeGraphExtractor:
    """Main class for extracting knowledge graphs from text"""
    
    def __init__(self, ollama_model: str = "llama3.2:latest", ollama_url: str = "http://localhost:11434"):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.nlp = self._load_spacy_model()
        self.graph = nx.DiGraph()
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling and auto-installation"""
        if not SPACY_AVAILABLE:
            print("📦 spaCy not available. Using basic text processing only.")
            return None
        
        try:
            # Try to load the model
            return spacy.load("en_core_web_sm")
        except OSError:
            print("📦 spaCy model 'en_core_web_sm' not found.")
            print("🔧 Attempting to download and install...")
            
            try:
                # Try to download the model
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                             check=True, capture_output=True)
                print("✅ Model downloaded successfully!")
                return spacy.load("en_core_web_sm")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to download model: {e}")
                print("🛠️ Please run manually: python -m spacy download en_core_web_sm")
                return None
            except Exception as e:
                print(f"❌ Error loading spaCy: {e}")
                return None
        
    def extract_with_ollama(self, text: str) -> List[Triple]:
        """Extract knowledge triples using Ollama LLM"""
        
        prompt = f"""
        Extract knowledge triples from the following text. Return them in the exact format: Subject | Predicate | Object
        
        Rules:
        - Extract only factual relationships between entities
        - Use specific, clear predicates (action verbs or relationship types)
        - Keep subjects and objects as concrete nouns or proper nouns
        - One triple per line
        - No additional text or explanations
        - Focus on the most important and clear relationships
        
        Examples:
        Alice | works_for | Acme Corp
        Bob | reports_to | Charlie
        Charlie | founded | Acme Corp
        Globex | acquired | Acme Corp
        
        Text: {text}
        
        Extracted triples:
        """
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Lower temperature for more consistent output
                        "top_p": 0.9,
                        "num_predict": 200   # Limit response length
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_triples_from_response(result.get("response", ""))
            else:
                print(f"Ollama API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return []
    
    def _parse_triples_from_response(self, response: str) -> List[Triple]:
        """Parse triples from Ollama response"""
        triples = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 3:
                    subject, predicate, obj = parts[0], parts[1], parts[2]
                    if subject and predicate and obj:
                        triples.append(Triple(
                            subject=self._clean_entity(subject),
                            predicate=self._clean_predicate(predicate),
                            object=self._clean_entity(obj),
                            confidence=0.8
                        ))
        
        return triples
    
    def extract_with_spacy(self, text: str) -> List[Triple]:
        """Extract knowledge using spaCy NLP pipeline"""
        if not self.nlp:
            print("⚠️ spaCy not available. Using basic pattern matching instead.")
            return self._extract_with_patterns(text)
        
        doc = self.nlp(text)
        triples = []
        
        # Extract entities
        entities = {ent.text: ent.label_ for ent in doc.ents}
        
        # Simple dependency parsing for relationships
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"] and token.head.pos_ == "VERB":
                subject = token.text
                predicate = token.head.lemma_
                
                # Find object
                for child in token.head.children:
                    if child.dep_ in ["dobj", "pobj", "attr"]:
                        obj = child.text
                        triples.append(Triple(
                            subject=self._clean_entity(subject),
                            predicate=self._clean_predicate(predicate),
                            object=self._clean_entity(obj),
                            confidence=0.6
                        ))
                        break
        
        return triples
    
    def _extract_with_patterns(self, text: str) -> List[Triple]:
        """Fallback extraction using regex patterns when spaCy is unavailable"""
        triples = []
        
        # Simple patterns for common relationships
        patterns = [
            # "X works for Y"
            (r'(\w+(?:\s+\w+)*)\s+works?\s+for\s+(\w+(?:\s+\w+)*)', 'works_for'),
            # "X reports to Y"
            (r'(\w+(?:\s+\w+)*)\s+reports?\s+to\s+(\w+(?:\s+\w+)*)', 'reports_to'),
            # "X knows Y"
            (r'(\w+(?:\s+\w+)*)\s+knows?\s+(\w+(?:\s+\w+)*)', 'knows'),
            # "X founded Y"
            (r'(\w+(?:\s+\w+)*)\s+founded\s+(\w+(?:\s+\w+)*)', 'founded'),
            # "X acquired Y"
            (r'(\w+(?:\s+\w+)*)\s+acquired\s+(\w+(?:\s+\w+)*)', 'acquired'),
            # "Both X and Y [verb] Z" - handle compound subjects
            (r'[Bb]oth\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)\s+(\w+)\s+(?:to\s+)?(\w+(?:\s+\w+)*)', 'compound_relation'),
            # "X was born in Y"
            (r'(\w+(?:\s+\w+)*)\s+was\s+born\s+in\s+(\w+(?:\s+\w+)*)', 'was_born_in'),
            # "X died in Y"
            (r'(\w+(?:\s+\w+)*)\s+died\s+in\s+(\w+(?:\s+\w+)*)', 'died_in'),
            # "X worked at Y"
            (r'(\w+(?:\s+\w+)*)\s+worked\s+at\s+(\w+(?:\s+\w+)*)', 'worked_at'),
            # "X won Y"
            (r'(\w+(?:\s+\w+)*)\s+won\s+(?:the\s+)?(\w+(?:\s+\w+)*)', 'won'),
            # "X developed Y"
            (r'(\w+(?:\s+\w+)*)\s+developed\s+(?:the\s+)?(\w+(?:\s+\w+)*)', 'developed'),
            # "X is Y"
            (r'(\w+(?:\s+\w+)*)\s+(?:is|was)\s+a\s+(\w+(?:\s+\w+)*)', 'is_a'),
        ]
        
        for pattern, relation in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if relation == 'compound_relation':
                    # Handle "Both X and Y verb Z" pattern
                    subject1 = self._clean_entity(match.group(1))
                    subject2 = self._clean_entity(match.group(2))
                    verb = match.group(3).lower()
                    obj = self._clean_entity(match.group(4))
                    
                    # Determine the relation based on the verb
                    if verb in ['report', 'reports']:
                        relation_name = 'reports_to'
                    elif verb in ['work', 'works']:
                        relation_name = 'works_for'
                    else:
                        relation_name = verb
                    
                    # Create two triples for compound subjects
                    triples.append(Triple(
                        subject=subject1,
                        predicate=relation_name,
                        object=obj,
                        confidence=0.6
                    ))
                    triples.append(Triple(
                        subject=subject2,
                        predicate=relation_name,
                        object=obj,
                        confidence=0.6
                    ))
                else:
                    subject = self._clean_entity(match.group(1))
                    obj = self._clean_entity(match.group(2))
                    triples.append(Triple(
                        subject=subject,
                        predicate=relation,
                        object=obj,
                        confidence=0.5
                    ))
        
        return triples
    
    def extract_entities_advanced(self, text: str) -> List[Entity]:
        """Advanced entity extraction combining spaCy and custom rules"""
        entities = []
        entity_dict = defaultdict(list)
        
        if self.nlp:
            # spaCy named entities
            doc = self.nlp(text)
            for ent in doc.ents:
                entity_dict[ent.text].append(ent.label_)
        else:
            # Fallback: basic entity patterns
            # Proper nouns (capitalized words)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            for noun in proper_nouns:
                entity_dict[noun].append("PERSON")  # Default to person
        
        # Custom patterns for additional entities
        # Dates
        date_pattern = r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|\b\w+\s+\d{1,2},\s+\d{4}\b'
        dates = re.findall(date_pattern, text)
        for date in dates:
            entity_dict[date].append("DATE")
        
        # Universities/Organizations
        org_pattern = r'\b\w+\s+University\b|\b\w+\s+Institute\b|\b\w+\s+College\b'
        orgs = re.findall(org_pattern, text, re.IGNORECASE)
        for org in orgs:
            entity_dict[org].append("ORG")
        
        # Convert to Entity objects
        for name, types in entity_dict.items():
            entities.append(Entity(
                name=name,
                entity_type=max(set(types), key=types.count),  # Most common type
                mentions=[name],
                confidence=0.7
            ))
        
        return entities
    
    def _clean_entity(self, entity: str) -> str:
        """Clean and normalize entity names"""
        entity = entity.strip().strip('"\'')
        entity = re.sub(r'^(the|a|an)\s+', '', entity.lower())
        return entity.title()
    
    def _clean_predicate(self, predicate: str) -> str:
        """Clean and normalize predicate names"""
        predicate = predicate.strip().lower()
        predicate = re.sub(r'[^\w\s]', '', predicate)
        return predicate.replace(' ', '_')
    
    def build_graph(self, triples: List[Triple]) -> nx.DiGraph:
        """Build NetworkX graph from triples"""
        self.graph.clear()
        
        for triple in triples:
            self.graph.add_edge(
                triple.subject,
                triple.object,
                relation=triple.predicate,
                weight=triple.confidence
            )
        
        return self.graph
    
    def visualize_graph_matplotlib(self, save_path: str = None):
        """Visualize graph using matplotlib"""
        if len(self.graph.nodes()) == 0:
            print("No graph to visualize. Extract triples first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color='lightblue',
            node_size=2000,
            alpha=0.7
        )
        
        # Draw edges with proper arrow direction
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=25,
            arrowstyle='->',
            alpha=0.7,
            width=2,
            connectionstyle="arc3,rad=0.1"  # Slight curve to see direction better
        )
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight='bold')
        
        # Draw edge labels (relationships) with better positioning
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=9, 
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        plt.title("Knowledge Graph", size=16, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_graph(self) -> Dict:
        """Analyze graph properties and statistics"""
        if len(self.graph.nodes()) == 0:
            return {"error": "No graph to analyze"}
        
        analysis = {
            "nodes": len(self.graph.nodes()),
            "edges": len(self.graph.edges()),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
            "avg_clustering": nx.average_clustering(self.graph.to_undirected()),
        }
        
        # Central nodes
        try:
            centrality = nx.degree_centrality(self.graph)
            analysis["most_central_nodes"] = sorted(
                centrality.items(), key=lambda x: x[1], reverse=True
            )[:5]
        except:
            analysis["most_central_nodes"] = []
        
        # Common relationships
        relations = [data.get('relation', '') for _, _, data in self.graph.edges(data=True)]
        relation_counts = pd.Series(relations).value_counts().head(5).to_dict()
        analysis["common_relations"] = relation_counts
        
        return analysis
    
    def export_graph(self, format: str = "json", filepath: str = None) -> str:
        """Export graph in various formats"""
        if format.lower() == "json":
            graph_data = {
                "nodes": [{"id": node, "label": node} for node in self.graph.nodes()],
                "edges": [
                    {
                        "source": edge[0],
                        "target": edge[1],
                        "relation": self.graph[edge[0]][edge[1]].get('relation', ''),
                        "weight": self.graph[edge[0]][edge[1]].get('weight', 1.0)
                    }
                    for edge in self.graph.edges()
                ]
            }
            
            result = json.dumps(graph_data, indent=2)
            
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(result)
            
            return result
        
        elif format.lower() == "gexf":
            if filepath:
                nx.write_gexf(self.graph, filepath)
                return f"Graph exported to {filepath}"
            else:
                return "Filepath required for GEXF export"
        
        else:
            return f"Unsupported format: {format}"

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
