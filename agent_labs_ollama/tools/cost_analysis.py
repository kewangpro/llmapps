if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print(json.dumps({"tool": "cost_analysis", "success": False, "error": "No JSON input provided"}))
        sys.exit(1)
    try:
        params = json.loads(sys.argv[1])
        file_path = params.get("file_path")
        query = params.get("query", "")
        result = analyze_cogs_data(file_path, query)
        print(json.dumps({"tool": "cost_analysis", "success": True, "result": result}))
    except Exception as e:
        print(json.dumps({"tool": "cost_analysis", "success": False, "error": str(e)}))
        sys.exit(1)
"""
Cost Analysis Tool
Performs cost analysis operations on COGS data files
"""

import pandas as pd
import chardet
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
        df = pd.read_csv(file_path, encoding='utf-16le', sep='\t')
        if df.iloc[0].isna().all() or df.columns[0].startswith('\ufeff'):
            df = pd.read_csv(file_path, encoding='utf-16le', sep='\t', skiprows=1)
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    elif 'utf-8' in encoding.lower():
        df = pd.read_csv(file_path, encoding='utf-8')
    else:
        df = pd.read_csv(file_path, encoding=encoding)

    # Generate cost analysis insights
    insights = generate_cost_insights(df, query)
    return {
        "data_summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "date_range": get_date_range(df)
        },
        "cost_insights": insights,
        "recommendations": generate_recommendations(df, insights)
    }


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
