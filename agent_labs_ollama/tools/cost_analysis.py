"""
Cost Analysis Tool
Reads and organizes COGS data files for analysis
"""

import pandas as pd
import chardet
import logging
import base64
import os
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger("CostAnalysisTool")


def save_to_outputs_folder(content: str, filename: str) -> str:
    """Save content to outputs folder and return the full path"""
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return output_path


def analyze_cogs_data(file_path: str) -> Dict[str, Any]:
    """Read and organize COGS data without visualization"""
    logger.info(f"Starting cost analysis for file: {file_path}")

    # Detect encoding
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        encoding_info = chardet.detect(raw_data)
        encoding = encoding_info.get('encoding', 'utf-8')
        logger.debug(f"Detected file encoding: {encoding}")

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

    # Generate cost analysis insights
    insights = generate_cost_insights(df)
    logger.info(f"Analysis complete: {len(df)} rows, {len(df.columns)} columns")

    # Extract monthly data for CSV generation
    monthly_data = extract_monthly_data(df)

    # Generate CSV data file for download using existing monthly data
    cost_summary_csv = format_monthly_data_as_csv(monthly_data)

    # Save CSV file to outputs directory
    cost_filename = f"cost_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = save_to_outputs_folder(cost_summary_csv, cost_filename)

    # Create downloadable file data
    file_base64 = base64.b64encode(cost_summary_csv.encode('utf-8')).decode('utf-8')
    file_size_mb = len(cost_summary_csv.encode('utf-8')) / (1024 * 1024)

    # Return organized data similar to forecast tool
    result = {
        "data_summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "date_range": get_date_range(df)
        },
        "cost_insights": insights,
        "recommendations": generate_recommendations(insights),
        # Add raw data for visualization tool (similar to stock historical_data)
        "monthly_data": monthly_data,
        # Add downloadable CSV file data (similar to forecast tool)
        "output_path": output_path,
        "file_size_mb": round(file_size_mb, 4),
        "cost_analysis_file_data": {
            "base64": file_base64,
            "filename": cost_filename,
            "mime_type": "text/csv",
            "content_preview": cost_summary_csv[:300] + "..." if len(cost_summary_csv) > 300 else cost_summary_csv
        }
    }

    return result


def generate_cost_insights(df: pd.DataFrame) -> Dict[str, Any]:
    insights = {}
    month_columns = [col for col in df.columns if col.startswith('2025-')]
    for col in month_columns:
        # Fix: Handle comma-formatted numbers properly before converting to numeric
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    if 'Business Unit' in df.columns:
        business_unit_costs = df.groupby('Business Unit')[month_columns].sum()
        business_unit_costs = business_unit_costs[business_unit_costs.sum(axis=1) > 0]
        if not business_unit_costs.empty:
            insights['cost_per_business_unit'] = {
                'summary': f"Analysis across {len(business_unit_costs)} business units",
                'top_business_units': business_unit_costs.sum(axis=1).sort_values(ascending=False).to_dict(),
                'monthly_trends': business_unit_costs.to_dict()
            }
    if 'AWS Product' in df.columns:
        aws_product_costs = df.groupby('AWS Product')[month_columns].sum()
        aws_product_costs = aws_product_costs[aws_product_costs.sum(axis=1) > 0]
        if not aws_product_costs.empty:
            insights['cost_per_aws_product'] = {
                'summary': f"Analysis across {len(aws_product_costs)} AWS products",
                'top_aws_products': aws_product_costs.sum(axis=1).sort_values(ascending=False).to_dict(),
                'monthly_trends': aws_product_costs.to_dict()
            }
    if 'Service Group New' in df.columns:
        service_group_costs = df.groupby('Service Group New')[month_columns].sum()
        service_group_costs = service_group_costs[service_group_costs.sum(axis=1) > 0]
        if not service_group_costs.empty:
            # Limit to top 50 service groups to reduce LLM payload
            top_50_service_groups = service_group_costs.sum(axis=1).sort_values(ascending=False).head(50).index
            top_service_group_costs = service_group_costs.loc[top_50_service_groups]
            insights['cost_per_service_group'] = {
                'summary': f"Analysis of top 50 service groups (out of {len(service_group_costs)} total)",
                'top_service_groups': service_group_costs.sum(axis=1).sort_values(ascending=False).head(50).to_dict(),
                'monthly_trends': top_service_group_costs.to_dict()
            }

    # Add analysis for top 50 services
    if 'Service Name' in df.columns:
        service_costs = df.groupby('Service Name')[month_columns].sum()
        service_costs = service_costs[service_costs.sum(axis=1) > 0]
        if not service_costs.empty:
            # Limit to top 50 services to reduce LLM payload
            top_50_services = service_costs.sum(axis=1).sort_values(ascending=False).head(50).index
            top_service_costs = service_costs.loc[top_50_services]
            insights['cost_per_service'] = {
                'summary': f"Analysis of top 50 services (out of {len(service_costs)} total)",
                'top_services': service_costs.sum(axis=1).sort_values(ascending=False).head(50).to_dict(),
                'monthly_trends': top_service_costs.to_dict()
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


def extract_monthly_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract monthly data in format suitable for visualization tool"""
    month_columns = [col for col in df.columns if col.startswith('2025-')]

    # Prepare data for different visualization dimensions
    result = {
        "months": sorted(month_columns),
        "total_costs": {},
        "business_unit_costs": {},
        "aws_product_costs": {},
        "service_group_costs": {}
    }

    # Monthly totals
    for month in month_columns:
        df[month] = pd.to_numeric(df[month].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        result["total_costs"][month] = float(df[month].sum())

    # Business unit breakdown
    if 'Business Unit' in df.columns:
        business_unit_costs = df.groupby('Business Unit')[month_columns].sum()
        for unit in business_unit_costs.index:
            if pd.notna(unit):
                result["business_unit_costs"][unit] = {}
                for month in month_columns:
                    result["business_unit_costs"][unit][month] = float(business_unit_costs.loc[unit, month])

    # AWS product breakdown
    if 'AWS Product' in df.columns:
        aws_product_costs = df.groupby('AWS Product')[month_columns].sum()
        for product in aws_product_costs.index:
            if pd.notna(product):
                result["aws_product_costs"][product] = {}
                for month in month_columns:
                    result["aws_product_costs"][product][month] = float(aws_product_costs.loc[product, month])

    # Service group breakdown
    if 'Service Group New' in df.columns:
        service_group_costs = df.groupby('Service Group New')[month_columns].sum()
        for group in service_group_costs.index:
            if pd.notna(group):
                result["service_group_costs"][group] = {}
                for month in month_columns:
                    result["service_group_costs"][group][month] = float(service_group_costs.loc[group, month])

    return result


def format_monthly_data_as_csv(monthly_data: Dict[str, Any]) -> str:
    """Format monthly data as CSV string for download"""
    try:
        months = monthly_data.get("months", [])

        # Default to business unit breakdown (matches current tool output)
        business_unit_costs = monthly_data.get("business_unit_costs", {})
        if business_unit_costs and months:
            csv_data = "month,business_unit,cost\n"
            for unit, unit_costs in business_unit_costs.items():
                for month in months:
                    cost = unit_costs.get(month, 0)
                    csv_data += f"{month},{unit},{cost}\n"
            return csv_data

        # Fallback to monthly totals
        total_costs = monthly_data.get("total_costs", {})
        if total_costs and months:
            csv_data = "month,total_cost\n"
            for month in months:
                cost = total_costs.get(month, 0)
                csv_data += f"{month},{cost}\n"
            return csv_data

        # Final fallback
        return "month,cost\nNo data available"

    except Exception as e:
        logger.error(f"Error formatting monthly data as CSV: {str(e)}")
        return f"month,error\n2025-01,Error formatting data: {str(e)}\n"


def generate_recommendations(insights: Dict[str, Any]) -> list:
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

        if not file_path:
            raise ValueError("file_path is required")

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Analyze the cost data (no query or analysis_intent needed)
        result = analyze_cogs_data(file_path)

        # Format successful response (similar to stock analysis tool)
        response = {
            "tool": "cost_analysis",
            "success": True,
            "file_path": file_path,
            "timestamp": datetime.now().isoformat(),
            **result
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