"""
Cost Analysis Tool
Performs cost analysis operations on COGS data files
"""

import pandas as pd
import chardet
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io
from typing import Dict, Any


def analyze_cogs_data(file_path: str, query: str) -> Dict[str, Any]:
    """Analyze COGS data and generate insights"""
    # Detect encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        encoding_info = chardet.detect(raw_data)
        encoding = encoding_info.get('encoding', 'utf-8')

    # Read CSV with detected encoding
    if 'utf-16' in encoding.lower():
        # Try reading normally first
        df = pd.read_csv(file_path, encoding='utf-16le', sep='\t')
        # Check if we need to skip first row (common with UTF-16 files)
        if df.columns[0] == 'Unnamed: 0' or 'Unnamed:' in str(df.columns[0]):
            df = pd.read_csv(file_path, encoding='utf-16le', sep='\t', skiprows=1)
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    elif 'utf-8' in encoding.lower():
        df = pd.read_csv(file_path, encoding='utf-8')
    else:
        df = pd.read_csv(file_path, encoding=encoding)

    # Generate cost analysis insights and charts
    insights = generate_cost_insights(df, query)
    chart_data = generate_cost_charts(df, insights, query)

    # Return structure consistent with visualization tool
    result = {
        "data_summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "date_range": get_date_range(df)
        },
        "cost_insights": insights,
        "recommendations": generate_recommendations(df, insights)
    }

    # Add chart data at top level (same as visualization tool)
    result.update({
        "chart_html": chart_data.get("chart_html", ""),
        "chart_image_base64": chart_data.get("chart_image_base64", ""),
        "chart_config": chart_data.get("chart_config", {}),
        "message": chart_data.get("message", "Cost analysis completed")
    })

    return result


def generate_cost_insights(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    insights = {}
    month_columns = [col for col in df.columns if col.startswith('2025-')]
    for col in month_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if 'Business Unit' in df.columns:
        business_unit_costs = df.groupby('Business Unit')[month_columns].sum()
        business_unit_costs = business_unit_costs[business_unit_costs.sum(axis=1) > 0]
        if not business_unit_costs.empty:
            insights['cost_per_business_unit'] = {
                'summary': f"Analysis across {len(business_unit_costs)} business units",
                'top_business_units': business_unit_costs.sum(axis=1).nlargest(5).to_dict(),
                'monthly_trends': business_unit_costs.to_dict()
            }
    if 'AWS Product' in df.columns:
        aws_product_costs = df.groupby('AWS Product')[month_columns].sum()
        aws_product_costs = aws_product_costs[aws_product_costs.sum(axis=1) > 0]
        if not aws_product_costs.empty:
            insights['cost_per_aws_product'] = {
                'summary': f"Analysis across {len(aws_product_costs)} AWS products",
                'top_aws_products': aws_product_costs.sum(axis=1).nlargest(5).to_dict(),
                'monthly_trends': aws_product_costs.to_dict()
            }
    if 'Service Group New' in df.columns:
        service_group_costs = df.groupby('Service Group New')[month_columns].sum()
        service_group_costs = service_group_costs[service_group_costs.sum(axis=1) > 0]
        if not service_group_costs.empty:
            insights['cost_per_service_group'] = {
                'summary': f"Analysis across {len(service_group_costs)} service groups",
                'top_service_groups': service_group_costs.sum(axis=1).nlargest(5).to_dict(),
                'monthly_trends': service_group_costs.to_dict()
            }
    total_monthly_costs = df[month_columns].sum()
    insights['overall_trends'] = {
        'total_cost_by_month': total_monthly_costs.to_dict(),
        'cost_growth': calculate_growth_rates(total_monthly_costs)
    }
    if not total_monthly_costs.empty and total_monthly_costs.sum() > 0:
        insights['overall_trends']['peak_month'] = total_monthly_costs.idxmax()
        insights['overall_trends']['lowest_month'] = total_monthly_costs.idxmin()
    return insights


def get_date_range(df: pd.DataFrame) -> Dict[str, str]:
    month_columns = [col for col in df.columns if col.startswith('2025-')]
    if month_columns:
        return {
            "start_month": min(month_columns),
            "end_month": max(month_columns),
            "total_months": len(month_columns)
        }
    return {}


def calculate_growth_rates(monthly_costs: pd.Series) -> Dict[str, float]:
    growth_rates = {}
    for i in range(1, len(monthly_costs)):
        prev_month = monthly_costs.iloc[i-1]
        curr_month = monthly_costs.iloc[i]
        if prev_month > 0:
            growth_rate = ((curr_month - prev_month) / prev_month) * 100
            growth_rates[f"{monthly_costs.index[i-1]}_to_{monthly_costs.index[i]}"] = round(growth_rate, 2)
    return growth_rates


def generate_recommendations(df: pd.DataFrame, insights: Dict[str, Any]) -> list:
    recommendations = []
    if 'cost_per_business_unit' in insights:
        top_units = insights['cost_per_business_unit']['top_business_units']
        if top_units:
            top_unit = list(top_units.keys())[0]
            recommendations.append(f"Focus on optimizing costs for '{top_unit}' - highest spending business unit")
    if 'overall_trends' in insights:
        growth_rates = insights['overall_trends'].get('cost_growth', {})
        high_growth = [k for k, v in growth_rates.items() if v > 20]
        if high_growth:
            recommendations.append(f"Investigate cost spikes in periods: {', '.join(high_growth)}")
    if 'cost_per_aws_product' in insights:
        top_products = insights['cost_per_aws_product']['top_aws_products']
        if top_products:
            top_product = list(top_products.keys())[0]
            recommendations.append(f"Consider optimization strategies for '{top_product}' - highest cost AWS service")
    return recommendations


def generate_cost_charts(df: pd.DataFrame, insights: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Generate cost analysis charts based on the query and insights"""
    try:
        # Find month columns
        month_columns = [col for col in df.columns if col.startswith('2025-')]
        if not month_columns:
            return {"message": "No month data found for charting"}

        # Create chart based on what data is available and query focus
        if ("aws product" in query.lower() or "per month" in query.lower()) and 'cost_per_aws_product' in insights:
            # Create AWS product monthly trends line chart using same approach as visualization
            monthly_trends = insights['cost_per_aws_product']['monthly_trends']
            if monthly_trends:
                # Convert monthly trends to DataFrame format like visualization expects
                data_rows = []
                for month, products_costs in monthly_trends.items():
                    for product, cost in products_costs.items():
                        data_rows.append({
                            'Month': month,
                            'AWS_Product': product,
                            'Cost': cost
                        })

                trends_df = pd.DataFrame(data_rows)

                # Create Plotly line chart using px.line like visualization does
                fig = px.line(trends_df, x='Month', y='Cost', color='AWS_Product',
                            title="Cost Trends by AWS Product Over Time",
                            labels={'Month': 'Month', 'Cost': 'Cost ($)', 'AWS_Product': 'AWS Product'})

                chart_html = fig.to_html(include_plotlyjs='cdn')

                # Create matplotlib version for base64 - same style as visualization
                plt.figure(figsize=(12, 8))
                for product in trends_df['AWS_Product'].unique():
                    if pd.notna(product):  # Skip NaN values
                        product_data = trends_df[trends_df['AWS_Product'] == product].sort_values('Month')
                        plt.plot(product_data['Month'], product_data['Cost'],
                               marker='o', linewidth=2, markersize=4, label=product)

                plt.title("Cost Trends by AWS Product Over Time")
                plt.xlabel("Month")
                plt.ylabel("Cost ($)")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                return {
                    "chart_html": chart_html,
                    "chart_image_base64": img_base64,
                    "chart_config": {"x_column": "Month", "y_column": "Cost", "color_column": "AWS_Product", "title": "Cost Trends by AWS Product Over Time"},
                    "message": f"Created line chart with {len(trends_df)} data points across {trends_df['AWS_Product'].nunique()} series"
                }

        # Default: monthly cost trends
        if 'overall_trends' in insights and insights['overall_trends']['total_cost_by_month']:
            monthly_data = insights['overall_trends']['total_cost_by_month']
            months = list(monthly_data.keys())
            costs = list(monthly_data.values())

            # Create Plotly line chart
            fig = px.line(x=months, y=costs,
                        title="Monthly Cost Trends",
                        labels={'x': 'Month', 'y': 'Total Cost ($)'})
            fig.update_traces(mode='lines+markers')

            chart_html = fig.to_html(include_plotlyjs='cdn')

            # Create matplotlib version for base64
            plt.figure(figsize=(12, 6))
            plt.plot(months, costs, marker='o', linewidth=2, markersize=6)
            plt.title("Monthly Cost Trends")
            plt.xlabel("Month")
            plt.ylabel("Total Cost ($)")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return {
                "chart_html": chart_html,
                "chart_image_base64": img_base64,
                "chart_config": {"chart_type": "line", "months": len(months)},
                "message": f"Created monthly cost trend chart with {len(months)} months"
            }

        return {"message": "No suitable data found for charting"}

    except Exception as e:
        return {"message": f"Chart generation failed: {str(e)}"}


def main():
    """Main function for command line execution"""
    import sys
    import json
    import os
    from datetime import datetime

    try:
        if len(sys.argv) != 2:
            raise ValueError("Expected exactly one JSON argument")

        # Parse JSON arguments
        args = json.loads(sys.argv[1])
        file_path = args.get("file_path", "")
        query = args.get("query", "")

        if not file_path:
            raise ValueError("file_path is required")

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Analyze the cost data
        result = analyze_cogs_data(file_path, query)

        # Format successful response (same structure as visualization tool)
        response = {
            "tool": "cost_analysis",
            "success": True,
            "file_path": file_path,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            **result  # Spread chart fields to top level like visualization tool
        }

        print(json.dumps(response, indent=2))

    except FileNotFoundError as e:
        print(json.dumps({
            "tool": "cost_analysis",
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }))
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(json.dumps({
            "tool": "cost_analysis",
            "success": False,
            "error": f"Invalid JSON arguments: {e}",
            "timestamp": datetime.now().isoformat()
        }))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "tool": "cost_analysis",
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
