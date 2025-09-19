#!/usr/bin/env python3
"""
Visualization Tool
Creates charts and visualizations from data input
"""

import json
import sys
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import to_html
from typing import Dict, Any, List, Union
from datetime import datetime
import os

def create_visualization(data: str, chart_type: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create visualization from data
    
    Args:
        data: Input data (JSON array, CSV, or other structured format)
        chart_type: Type of chart to create
        options: Additional chart options and configuration
    
    Returns:
        Dictionary with visualization results
    """
    try:
        if options is None:
            options = {}
            
        chart_types = {
            "line": create_line_chart,
            "bar": create_bar_chart,
            "scatter": create_scatter_plot,
            "pie": create_pie_chart,
            "histogram": create_histogram,
            "box": create_box_plot,
            "heatmap": create_heatmap,
            "area": create_area_chart,
            "bubble": create_bubble_chart,
            "treemap": create_treemap
        }
        
        if chart_type not in chart_types:
            return {
                "tool": "visualization",
                "success": False,
                "error": f"Unknown chart type: {chart_type}. Available: {', '.join(chart_types.keys())}"
            }
        
        # Parse input data
        df = parse_input_data(data)
        
        if df is None or df.empty:
            return {
                "tool": "visualization",
                "success": False,
                "error": "Could not parse input data or data is empty"
            }
        
        # Create the visualization
        result = chart_types[chart_type](df, options)
        
        return {
            "tool": "visualization",
            "success": True,
            "chart_type": chart_type,
            "data_rows": len(df),
            "data_columns": len(df.columns),
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat(),
            **result
        }
        
    except Exception as e:
        return {
            "tool": "visualization",
            "success": False,
            "error": str(e)
        }

def parse_input_data(data: str) -> pd.DataFrame:
    """Parse input data into pandas DataFrame"""
    try:
        # Try JSON first
        if data.strip().startswith('[') or data.strip().startswith('{'):
            json_data = json.loads(data)
            if isinstance(json_data, list):
                return pd.DataFrame(json_data)
            else:
                return pd.DataFrame([json_data])
        
        # Try CSV
        else:
            return pd.read_csv(io.StringIO(data))
            
    except Exception as e:
        # Last resort: try to split by lines and commas
        try:
            lines = data.strip().split('\n')
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split(',')]
                rows = []
                for line in lines[1:]:
                    if line.strip():
                        rows.append([cell.strip() for cell in line.split(',')])
                return pd.DataFrame(rows, columns=headers)
        except:
            pass
        
        return None

def create_line_chart(df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    """Create a line chart, supporting multiple time series"""
    x_col = options.get('x_column', df.columns[0])
    y_col = options.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
    color_col = options.get('color_column')  # For grouping multiple lines
    title = options.get('title', f'{y_col} over {x_col}')
    
    # Auto-detect if we should group by a categorical column for multiple lines
    if color_col is None and len(df.columns) > 2:
        # Look for a categorical column that could represent different series
        for col in df.columns:
            if col not in [x_col, y_col] and df[col].dtype == 'object':
                color_col = col
                break
    
    # Create Plotly chart
    if color_col and color_col in df.columns:
        # Multiple time series (one line per category)
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title,
                     labels={x_col: x_col.replace('_', ' ').title(), 
                            y_col: y_col.replace('_', ' ').title()})
        
        # Sort by x-axis to ensure proper line connections
        df_sorted = df.sort_values(x_col)
        
        # Create matplotlib version with multiple lines
        plt.figure(figsize=(12, 8))
        for category in df_sorted[color_col].unique():
            if pd.notna(category):  # Skip NaN values
                category_data = df_sorted[df_sorted[color_col] == category]
                plt.plot(category_data[x_col], category_data[y_col], 
                        marker='o', label=category, linewidth=2, markersize=4)
        
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        config_info = {"x_column": x_col, "y_column": y_col, "color_column": color_col, "title": title}
        message = f"Created line chart with {len(df)} data points across {df[color_col].nunique()} series"
    else:
        # Single time series
        fig = px.line(df, x=x_col, y=y_col, title=title,
                     labels={x_col: x_col.replace('_', ' ').title(), 
                            y_col: y_col.replace('_', ' ').title()})
        
        # Sort by x-axis
        df_sorted = df.sort_values(x_col)
        
        # Create matplotlib version
        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted[x_col], df_sorted[y_col], marker='o', linewidth=2, markersize=4)
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        config_info = {"x_column": x_col, "y_column": y_col, "title": title}
        message = f"Created line chart with {len(df)} data points"
    
    # Save as HTML
    html_content = to_html(fig, include_plotlyjs=True, div_id="chart")
    
    # Save matplotlib to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_html": html_content,
        "chart_image_base64": img_base64,
        "chart_config": config_info,
        "message": message
    }

def create_bar_chart(df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    """Create a bar chart"""
    x_col = options.get('x_column', df.columns[0])
    y_col = options.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
    title = options.get('title', f'{y_col} by {x_col}')
    
    # Create Plotly chart
    fig = px.bar(df, x=x_col, y=y_col, title=title)
    html_content = to_html(fig, include_plotlyjs=True, div_id="chart")
    
    # Create matplotlib version
    plt.figure(figsize=(10, 6))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_html": html_content,
        "chart_image_base64": img_base64,
        "chart_config": {"x_column": x_col, "y_column": y_col, "title": title},
        "message": f"Created bar chart with {len(df)} bars"
    }

def create_scatter_plot(df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    """Create a scatter plot"""
    x_col = options.get('x_column', df.columns[0])
    y_col = options.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
    title = options.get('title', f'{y_col} vs {x_col}')
    
    fig = px.scatter(df, x=x_col, y=y_col, title=title)
    html_content = to_html(fig, include_plotlyjs=True, div_id="chart")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.6)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_html": html_content,
        "chart_image_base64": img_base64,
        "chart_config": {"x_column": x_col, "y_column": y_col, "title": title},
        "message": f"Created scatter plot with {len(df)} points"
    }

def create_pie_chart(df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    """Create a pie chart"""
    label_col = options.get('label_column', df.columns[0])
    value_col = options.get('value_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
    title = options.get('title', f'Distribution of {value_col}')
    
    fig = px.pie(df, names=label_col, values=value_col, title=title)
    html_content = to_html(fig, include_plotlyjs=True, div_id="chart")
    
    plt.figure(figsize=(8, 8))
    plt.pie(df[value_col], labels=df[label_col], autopct='%1.1f%%')
    plt.title(title)
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_html": html_content,
        "chart_image_base64": img_base64,
        "chart_config": {"label_column": label_col, "value_column": value_col, "title": title},
        "message": f"Created pie chart with {len(df)} segments"
    }

def create_histogram(df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    """Create a histogram"""
    column = options.get('column', df.select_dtypes(include=['number']).columns[0])
    bins = options.get('bins', 20)
    title = options.get('title', f'Distribution of {column}')
    
    fig = px.histogram(df, x=column, nbins=bins, title=title)
    html_content = to_html(fig, include_plotlyjs=True, div_id="chart")
    
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_html": html_content,
        "chart_image_base64": img_base64,
        "chart_config": {"column": column, "bins": bins, "title": title},
        "message": f"Created histogram with {bins} bins"
    }

def create_box_plot(df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    """Create a box plot"""
    y_col = options.get('y_column', df.select_dtypes(include=['number']).columns[0])
    x_col = options.get('x_column', None)
    title = options.get('title', f'Box Plot of {y_col}')
    
    if x_col and x_col in df.columns:
        fig = px.box(df, x=x_col, y=y_col, title=title)
    else:
        fig = px.box(df, y=y_col, title=title)
    
    html_content = to_html(fig, include_plotlyjs=True, div_id="chart")
    
    plt.figure(figsize=(10, 6))
    if x_col and x_col in df.columns:
        df.boxplot(column=y_col, by=x_col)
    else:
        df.boxplot(column=y_col)
    plt.title(title)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_html": html_content,
        "chart_image_base64": img_base64,
        "chart_config": {"y_column": y_col, "x_column": x_col, "title": title},
        "message": "Created box plot"
    }

def create_heatmap(df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    """Create a heatmap"""
    title = options.get('title', 'Correlation Heatmap')
    
    # Create correlation matrix for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(corr_matrix, title=title, text_auto=True)
    html_content = to_html(fig, include_plotlyjs=True, div_id="chart")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_html": html_content,
        "chart_image_base64": img_base64,
        "chart_config": {"title": title, "variables": list(corr_matrix.columns)},
        "message": f"Created correlation heatmap for {len(corr_matrix.columns)} variables"
    }

def create_area_chart(df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    """Create an area chart"""
    x_col = options.get('x_column', df.columns[0])
    y_col = options.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
    title = options.get('title', f'{y_col} over {x_col} (Area)')
    
    fig = px.area(df, x=x_col, y=y_col, title=title)
    html_content = to_html(fig, include_plotlyjs=True, div_id="chart")
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(df[x_col], df[y_col], alpha=0.7)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_html": html_content,
        "chart_image_base64": img_base64,
        "chart_config": {"x_column": x_col, "y_column": y_col, "title": title},
        "message": f"Created area chart with {len(df)} data points"
    }

def create_bubble_chart(df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    """Create a bubble chart"""
    x_col = options.get('x_column', df.columns[0])
    y_col = options.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
    size_col = options.get('size_column', df.columns[2] if len(df.columns) > 2 else y_col)
    title = options.get('title', f'{y_col} vs {x_col} (Bubble Size: {size_col})')
    
    fig = px.scatter(df, x=x_col, y=y_col, size=size_col, title=title)
    html_content = to_html(fig, include_plotlyjs=True, div_id="chart")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], s=df[size_col], alpha=0.6)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_html": html_content,
        "chart_image_base64": img_base64,
        "chart_config": {"x_column": x_col, "y_column": y_col, "size_column": size_col, "title": title},
        "message": f"Created bubble chart with {len(df)} bubbles"
    }

def create_treemap(df: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    """Create a treemap"""
    label_col = options.get('label_column', df.columns[0])
    value_col = options.get('value_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
    title = options.get('title', f'Treemap of {value_col}')
    
    fig = px.treemap(df, names=label_col, values=value_col, title=title)
    html_content = to_html(fig, include_plotlyjs=True, div_id="chart")
    
    # Treemap is complex in matplotlib, so we'll use a simpler representation
    plt.figure(figsize=(10, 6))
    plt.bar(df[label_col], df[value_col])
    plt.title(f"{title} (Bar representation)")
    plt.xlabel(label_col)
    plt.ylabel(value_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_html": html_content,
        "chart_image_base64": img_base64,
        "chart_config": {"label_column": label_col, "value_column": value_col, "title": title},
        "message": f"Created treemap with {len(df)} segments"
    }

def main():
    """CLI interface for the visualization tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: visualization.py <json_args>"}))
        sys.exit(1)
    
    try:
        args = json.loads(sys.argv[1])
        data = args.get("data", "")
        chart_type = args.get("chart_type", "")
        options = args.get("options", {})
        
        if not data:
            print(json.dumps({"error": "data is required"}))
            sys.exit(1)
        
        if not chart_type:
            print(json.dumps({"error": "chart_type is required"}))
            sys.exit(1)
        
        result = create_visualization(data, chart_type, options)
        print(json.dumps(result, indent=2))
        
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()