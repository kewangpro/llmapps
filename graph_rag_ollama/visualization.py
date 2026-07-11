"""
Graph visualization using Plotly and NetworkX.
"""

import logging
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional, List

from analytics import GraphAnalytics
from models import GraphMetrics

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Handles graph visualization and dashboard creation."""
    
    def __init__(self, networkx_graph: Optional[nx.Graph] = None):
        self.networkx_graph = networkx_graph
        self.analytics = GraphAnalytics(networkx_graph)
    
    def set_graph(self, networkx_graph: nx.Graph):
        """Set the NetworkX graph for visualization."""
        self.networkx_graph = networkx_graph
        self.analytics.set_graph(networkx_graph)
    
    def visualize_graph(
        self, 
        layout: str = "spring",
        node_limit: int = 100,
        show_labels: bool = True,
        highlight_entities: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive graph visualization using Plotly.
        
        Args:
            layout: Graph layout algorithm ("spring", "circular", "random")
            node_limit: Maximum number of nodes to display
            show_labels: Whether to show node labels
            highlight_entities: List of entities to highlight
            save_path: Path to save the visualization
        """
        try:
            if not self.networkx_graph or len(self.networkx_graph.nodes()) == 0:
                raise ValueError("No graph data available for visualization. Please build the knowledge graph first.")

            G = self.networkx_graph
            
            # Limit nodes for performance
            if len(G.nodes()) > node_limit:
                # Get most central nodes
                centrality = nx.degree_centrality(G)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:node_limit]
                top_node_names = [node[0] for node in top_nodes]
                G = G.subgraph(top_node_names)
            
            # Generate layout
            if layout == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == "circular":
                pos = nx.circular_layout(G)
            else:
                pos = nx.random_layout(G)
            
            # Extract edges
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Get edge information
                edge_data = G.get_edge_data(edge[0], edge[1])
                relation = edge_data.get('relation', 'connected') if edge_data else 'connected'
                edge_info.append(f"{edge[0]} -> {relation} -> {edge[1]}")
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='Relationships'
            )
            
            # Extract nodes
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            node_hover_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node information
                adjacencies = list(G.neighbors(node))
                degree = len(adjacencies)
                
                # Create hover text
                hover_info = f'<b>{node}</b><br>'
                hover_info += f'Connections: {degree}<br>'
                if adjacencies:
                    hover_info += f'Connected to: {", ".join(adjacencies[:3])}'
                    if len(adjacencies) > 3:
                        hover_info += f' and {len(adjacencies) - 3} more...'
                
                node_hover_text.append(hover_info)
                
                # Set node text for labels
                node_text.append(node if show_labels else '')
                
                # Color and size nodes based on degree and highlights
                if highlight_entities and node in highlight_entities:
                    node_colors.append('#ff4444')  # Red for highlighted
                    node_sizes.append(max(15, min(30, degree * 2)))
                else:
                    # Color based on degree (more connections = darker)
                    intensity = min(degree / 10.0, 1.0)  # Normalize degree
                    node_colors.append(f'rgba(100, 149, 237, {0.3 + 0.7 * intensity})')
                    node_sizes.append(max(8, min(20, degree + 5)))
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text' if show_labels else 'markers',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                textfont=dict(size=8, color='black'),
                hovertext=node_hover_text,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                name='Entities'
            )
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text=f'Knowledge Graph Visualization ({len(G.nodes())} nodes, {len(G.edges())} edges)',
                        font=dict(size=16)
                    ),
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=60),
                    annotations=[
                        dict(
                            text="Interactive Knowledge Graph - Hover over nodes for details",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color="grey", size=10)
                        )
                    ],
                    xaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False,
                        scaleanchor="y",
                        scaleratio=1
                    ),
                    yaxis=dict(
                        showgrid=False, 
                        zeroline=False, 
                        showticklabels=False
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=12)
                )
            )
            
            # Add some styling for better appearance
            fig.update_layout(
                width=1000,
                height=800,
                dragmode='pan'
            )
            
            # Save if requested
            if save_path:
                fig.write_html(save_path)
                logger.info(f"💾 Graph visualization saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Failed to create graph visualization: {e}")
            raise
    
    def create_analytics_dashboard(self) -> go.Figure:
        """Create comprehensive analytics dashboard."""
        try:
            metrics = self.analytics.get_graph_metrics()
            
            # Create subplots with mixed subplot types
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Graph Overview', 
                    'Node Degree Distribution',
                    'Top Entities by Centrality',
                    'Community Sizes'
                ),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "domain"}]]  # Changed last one to domain for pie chart
            )
            
            # 1. Graph Overview (Key Metrics as Bar Chart)
            overview_metrics = ['Entities', 'Relationships', 'Communities', 'Avg Degree']
            overview_values = [
                metrics.node_count, 
                metrics.edge_count, 
                len(metrics.communities), 
                round(metrics.avg_degree, 1)
            ]
            
            fig.add_trace(
                go.Bar(
                    x=overview_metrics,
                    y=overview_values,
                    name="Graph Metrics",
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                ),
                row=1, col=1
            )
            
            # 2. Degree distribution
            if self.networkx_graph and len(self.networkx_graph.nodes()) > 0:
                degrees = [d for n, d in self.networkx_graph.degree()]
                fig.add_trace(
                    go.Histogram(
                        x=degrees, 
                        name="Degree Distribution",
                        nbinsx=min(20, max(degrees) + 1) if degrees else 10,
                        marker_color='#ff7f0e'
                    ),
                    row=1, col=2
                )
            else:
                # Empty placeholder
                fig.add_trace(
                    go.Bar(x=['No Data'], y=[0], name="No Data Available"),
                    row=1, col=2
                )
            
            # 3. Top entities by centrality
            if metrics.centrality_scores.get('pagerank') and len(metrics.centrality_scores['pagerank']) > 0:
                top_entities = sorted(
                    metrics.centrality_scores['pagerank'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                if top_entities:
                    entities, scores = zip(*top_entities)
                    
                    # Truncate long entity names for display
                    display_entities = [
                        entity[:20] + '...' if len(entity) > 20 else entity 
                        for entity in entities
                    ]
                    
                    fig.add_trace(
                        go.Bar(
                            x=display_entities, 
                            y=list(scores), 
                            name="PageRank Centrality",
                            marker_color='#2ca02c',
                            hovertext=[f"{entity}: {score:.4f}" for entity, score in zip(entities, scores)],
                            hoverinfo='text'
                        ),
                        row=2, col=1
                    )
                else:
                    fig.add_trace(
                        go.Bar(x=['No Data'], y=[0], name="No Centrality Data"),
                        row=2, col=1
                    )
            else:
                fig.add_trace(
                    go.Bar(x=['No Data'], y=[0], name="No Centrality Data"),
                    row=2, col=1
                )
            
            # 4. Community sizes - Now using pie chart in domain subplot
            if metrics.communities and len(metrics.communities) > 0:
                community_sizes = [len(community) for community in metrics.communities]
                community_labels = [f"Community {i+1}" for i in range(len(community_sizes))]
                
                # Only show if we have meaningful communities
                if max(community_sizes) > 1:
                    fig.add_trace(
                        go.Pie(
                            labels=community_labels, 
                            values=community_sizes, 
                            name="Communities",
                            hoverinfo='label+value+percent'
                        ),
                        row=2, col=2
                    )
                else:
                    # Use bar chart instead for single nodes
                    fig.add_trace(
                        go.Bar(x=['Single Nodes'], y=[len(community_sizes)], name="No Communities"),
                        row=2, col=2
                    )
            else:
                # Use bar chart for no community data
                fig.add_trace(
                    go.Bar(x=['No Communities'], y=[0], name="No Community Data"),
                    row=2, col=2
                )
            
            # Update layout with proper title formatting
            fig.update_layout(
                title=dict(
                    text="Knowledge Graph Analytics Dashboard",
                    font=dict(size=20)
                ),
                showlegend=False,
                height=800,
                width=1200,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Update x-axis labels for readability
            fig.update_xaxes(tickangle=45, row=2, col=1)  # Rotate entity names
            
            # Add some spacing between subplots
            fig.update_layout(
                margin=dict(t=80, b=50, l=50, r=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Failed to create analytics dashboard: {e}")
            # Return a simple error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating dashboard: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Analytics Dashboard - Error",
                width=800, height=400
            )
            return fig