"""
Export utilities for knowledge graph data.
"""

import json
import logging
import networkx as nx
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from analytics import GraphAnalytics

logger = logging.getLogger(__name__)


class GraphExporter:
    """Handles exporting graph data in various formats."""
    
    def __init__(self, networkx_graph: Optional[nx.Graph] = None):
        self.networkx_graph = networkx_graph
        self.analytics = GraphAnalytics(networkx_graph)
    
    def set_graph(self, networkx_graph: nx.Graph):
        """Set the NetworkX graph for export."""
        self.networkx_graph = networkx_graph
        self.analytics.set_graph(networkx_graph)
    
    def export_graph_data(self, export_dir: str = "./exports") -> dict:
        """Export graph data to various formats."""
        try:
            export_path = Path(export_dir)
            export_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_info = {
                'timestamp': timestamp,
                'export_dir': str(export_path),
                'files': []
            }
            
            # Export NetworkX graph
            if self.networkx_graph:
                # JSON format
                json_file = export_path / f"graph_{timestamp}.json"
                graph_data = nx.node_link_data(self.networkx_graph)
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2)
                export_info['files'].append(str(json_file))
                
                # CSV format for edges
                edges_file = export_path / f"edges_{timestamp}.csv"
                edges_df = pd.DataFrame([
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'relation': self.networkx_graph.get_edge_data(edge[0], edge[1]).get('relation', 'connected')
                    }
                    for edge in self.networkx_graph.edges()
                ])
                edges_df.to_csv(edges_file, index=False)
                export_info['files'].append(str(edges_file))
                
                # CSV format for nodes
                nodes_file = export_path / f"nodes_{timestamp}.csv"
                nodes_df = pd.DataFrame([
                    {
                        'node': node,
                        'degree': self.networkx_graph.degree(node),
                        'type': self.networkx_graph.nodes[node].get('type', 'entity')
                    }
                    for node in self.networkx_graph.nodes()
                ])
                nodes_df.to_csv(nodes_file, index=False)
                export_info['files'].append(str(nodes_file))
            
            # Export analytics
            analytics_file = export_path / f"analytics_{timestamp}.json"
            metrics = self.analytics.get_graph_metrics()
            analytics_data = {
                'timestamp': timestamp,
                'node_count': metrics.node_count,
                'edge_count': metrics.edge_count,
                'density': metrics.density,
                'avg_degree': metrics.avg_degree,
                'clustering_coefficient': metrics.clustering_coefficient,
                'communities': metrics.communities
            }
            
            with open(analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics_data, f, indent=2)
            export_info['files'].append(str(analytics_file))
            
            logger.info(f"✅ Graph data exported to {export_path}")
            return export_info
            
        except Exception as e:
            logger.error(f"❌ Failed to export graph data: {e}")
            raise
    
    def export_to_json(self, filepath: str) -> str:
        """Export graph to JSON format."""
        try:
            if not self.networkx_graph:
                raise ValueError("No graph available for export")
            
            graph_data = {
                "nodes": [{"id": node, "label": node} for node in self.networkx_graph.nodes()],
                "edges": [
                    {
                        "source": edge[0],
                        "target": edge[1],
                        "relation": self.networkx_graph[edge[0]][edge[1]].get('relation', ''),
                        "weight": self.networkx_graph[edge[0]][edge[1]].get('weight', 1.0)
                    }
                    for edge in self.networkx_graph.edges()
                ]
            }
            
            result = json.dumps(graph_data, indent=2)
            
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result)
                logger.info(f"✅ Graph exported to JSON: {filepath}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to export to JSON: {e}")
            raise
    
    def export_to_gexf(self, filepath: str) -> str:
        """Export graph to GEXF format."""
        try:
            if not self.networkx_graph:
                raise ValueError("No graph available for export")
            
            nx.write_gexf(self.networkx_graph, filepath)
            logger.info(f"✅ Graph exported to GEXF: {filepath}")
            return f"Graph exported to {filepath}"
            
        except Exception as e:
            logger.error(f"❌ Failed to export to GEXF: {e}")
            raise
    
    def export_centrality_report(self, filepath: str) -> str:
        """Export centrality analysis report."""
        try:
            metrics = self.analytics.get_graph_metrics()
            
            report = {
                'graph_overview': {
                    'node_count': metrics.node_count,
                    'edge_count': metrics.edge_count,
                    'density': metrics.density,
                    'avg_degree': metrics.avg_degree,
                    'clustering_coefficient': metrics.clustering_coefficient
                },
                'centrality_analysis': {},
                'communities': metrics.communities,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add centrality scores for top entities
            for centrality_type, scores in metrics.centrality_scores.items():
                if scores:
                    top_entities = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
                    report['centrality_analysis'][centrality_type] = {
                        'top_entities': top_entities,
                        'avg_score': sum(scores.values()) / len(scores),
                        'max_score': max(scores.values()),
                        'min_score': min(scores.values())
                    }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✅ Centrality report exported: {filepath}")
            return f"Centrality report exported to {filepath}"
            
        except Exception as e:
            logger.error(f"❌ Failed to export centrality report: {e}")
            raise