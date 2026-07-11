"""
Graph analytics and metrics calculation.
"""

import logging
import networkx as nx
from typing import Optional

from models import GraphMetrics

logger = logging.getLogger(__name__)


class GraphAnalytics:
    """Handles graph analysis and metrics computation."""
    
    def __init__(self, networkx_graph: Optional[nx.Graph] = None):
        self.networkx_graph = networkx_graph
    
    def set_graph(self, networkx_graph: nx.Graph):
        """Set the NetworkX graph for analysis."""
        self.networkx_graph = networkx_graph
    
    def get_graph_metrics(self) -> GraphMetrics:
        """Get comprehensive graph analytics."""
        try:
            if not self.networkx_graph:
                raise ValueError("NetworkX graph not available")
            
            G = self.networkx_graph
            
            # Basic metrics
            node_count = len(G.nodes())
            edge_count = len(G.edges())
            density = nx.density(G)
            
            # Degree statistics
            degrees = dict(G.degree())
            avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            
            # Clustering coefficient
            clustering_coeff = nx.average_clustering(G)
            
            # Centrality measures
            centrality_scores = {}
            if node_count > 0 and edge_count > 0:
                centrality_scores['betweenness'] = nx.betweenness_centrality(G)
                centrality_scores['closeness'] = nx.closeness_centrality(G)
                centrality_scores['pagerank'] = nx.pagerank(G)
            
            # Community detection
            communities = []
            if edge_count > 0:
                try:
                    import networkx.algorithms.community as nx_comm
                    communities = list(nx_comm.greedy_modularity_communities(G))
                    communities = [list(community) for community in communities]
                except Exception as e:
                    logger.warning(f"⚠️ Community detection failed: {e}")
            
            return GraphMetrics(
                node_count=node_count,
                edge_count=edge_count,
                density=density,
                avg_degree=avg_degree,
                clustering_coefficient=clustering_coeff,
                centrality_scores=centrality_scores,
                communities=communities
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get graph analytics: {e}")
            return GraphMetrics(0, 0, 0, 0, 0, {}, [])
    
    def get_top_entities(self, metric: str = 'pagerank', top_k: int = 10) -> list:
        """Get top entities by specified centrality metric."""
        try:
            metrics = self.get_graph_metrics()
            centrality_scores = metrics.centrality_scores.get(metric, {})
            
            if not centrality_scores:
                return []
            
            return sorted(
                centrality_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Failed to get top entities: {e}")
            return []
    
    def find_shortest_path(self, source: str, target: str) -> Optional[list]:
        """Find shortest path between two entities."""
        try:
            if not self.networkx_graph:
                return None
            
            if source not in self.networkx_graph or target not in self.networkx_graph:
                return None
            
            return nx.shortest_path(self.networkx_graph, source, target)
            
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.error(f"❌ Failed to find shortest path: {e}")
            return None
    
    def get_node_neighbors(self, node: str, max_neighbors: int = 10) -> list:
        """Get neighbors of a specific node."""
        try:
            if not self.networkx_graph or node not in self.networkx_graph:
                return []
            
            neighbors = list(self.networkx_graph.neighbors(node))
            return neighbors[:max_neighbors]
            
        except Exception as e:
            logger.error(f"❌ Failed to get node neighbors: {e}")
            return []
    
    def get_subgraph(self, nodes: list) -> Optional[nx.Graph]:
        """Get subgraph containing specified nodes."""
        try:
            if not self.networkx_graph:
                return None
            
            valid_nodes = [node for node in nodes if node in self.networkx_graph]
            if not valid_nodes:
                return None
            
            return self.networkx_graph.subgraph(valid_nodes)
            
        except Exception as e:
            logger.error(f"❌ Failed to get subgraph: {e}")
            return None