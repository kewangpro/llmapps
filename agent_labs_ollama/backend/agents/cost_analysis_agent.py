"""
Cost Analysis Agent - Specialized agent for cost analysis operations
"""

import json
import logging
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger("MultiAgentSystem")


class CostAnalysisAgent(BaseAgent):
    """Specialized agent for cost analysis operations"""

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute cost analysis with intelligent parameter extraction"""
        try:
            logger.info(f"📊 CostAnalysisAgent analyzing: '{query}'")
            
            # Use default COGS file unless a different file is attached
            if "FILE_PATH:" in query:
                file_path = query.split("FILE_PATH:")[-1].strip()
                clean_query = query.split("FILE_PATH:")[0].strip()
                logger.info(f"📎 Using attached file: {file_path}")
            else:
                # Use absolute path to ensure file is found regardless of working directory
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                file_path = os.path.join(project_root, "data", "Any_COGS_1-8.csv")
                clean_query = query
                logger.info(f"📊 Using default COGS file: {file_path}")
            
            # Process the COGS data file
            try:
                result = self._analyze_cogs_data(file_path, clean_query)
                
                return {
                    "agent": "CostAnalysisAgent",
                    "tool": "cost_analysis",
                    "parameters": {"file_path": file_path, "query": clean_query},
                    "result": result,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"📊 Failed to analyze COGS data: {str(e)}")
                return {
                    "agent": "CostAnalysisAgent",
                    "success": False,
                    "error": f"Failed to analyze COGS data: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"📊 CostAnalysisAgent error: {str(e)}")
            return {
                "agent": "CostAnalysisAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_cogs_data(self, file_path: str, query: str) -> Dict[str, Any]:
        """Analyze COGS data and generate insights"""
        try:
            # Check file encoding and read appropriately
            import chardet
            
            # Detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                encoding_info = chardet.detect(raw_data)
                encoding = encoding_info.get('encoding', 'utf-8')
            
            logger.info(f"📊 Detected file encoding: {encoding}")
            
            # Read CSV directly with detected encoding - handle UTF-16 properly
            if 'utf-16' in encoding.lower():
                # For UTF-16 files, read with proper encoding and tab separator
                df = pd.read_csv(file_path, encoding='utf-16le', sep='\t')
                
                # The first row might be empty/BOM, check if we need to skip it
                if df.iloc[0].isna().all() or df.columns[0].startswith('\ufeff'):
                    # Re-read skipping the first row and using the second row as header
                    df = pd.read_csv(file_path, encoding='utf-16le', sep='\t', skiprows=1)
                
                # Clean up column names (remove BOM and extra whitespace)
                df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
                
            elif 'utf-8' in encoding.lower():
                df = pd.read_csv(file_path, encoding='utf-8')
            else:
                # Fallback to detected encoding
                df = pd.read_csv(file_path, encoding=encoding)
            
            logger.info(f"📊 Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"📊 Columns: {list(df.columns)}")
            
            # Generate cost analysis insights
            insights = self._generate_cost_insights(df, query)
            
            return {
                "data_summary": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "columns": list(df.columns),
                    "date_range": self._get_date_range(df)
                },
                "cost_insights": insights,
                "recommendations": self._generate_recommendations(df, insights)
            }
            
        except Exception as e:
            logger.error(f"📊 Error analyzing COGS data: {str(e)}")
            raise

    def _generate_cost_insights(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Generate specific cost insights based on the data"""
        insights = {}
        
        try:
            # Get month columns (cost data columns)
            month_columns = [col for col in df.columns if col.startswith('2025-')]
            
            # Convert month columns to numeric, handling empty strings
            for col in month_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # 1. Cost per Business Unit per Month
            if 'Business Unit' in df.columns:
                business_unit_costs = df.groupby('Business Unit')[month_columns].sum()
                # Filter out empty rows
                business_unit_costs = business_unit_costs[business_unit_costs.sum(axis=1) > 0]
                if not business_unit_costs.empty:
                    insights['cost_per_business_unit'] = {
                        'summary': f"Analysis across {len(business_unit_costs)} business units",
                        'top_business_units': business_unit_costs.sum(axis=1).nlargest(5).to_dict(),
                        'monthly_trends': business_unit_costs.to_dict()
                    }
            
            # 2. Cost per AWS Product per Month  
            if 'AWS Product' in df.columns:
                aws_product_costs = df.groupby('AWS Product')[month_columns].sum()
                # Filter out empty rows
                aws_product_costs = aws_product_costs[aws_product_costs.sum(axis=1) > 0]
                if not aws_product_costs.empty:
                    insights['cost_per_aws_product'] = {
                        'summary': f"Analysis across {len(aws_product_costs)} AWS products",
                        'top_aws_products': aws_product_costs.sum(axis=1).nlargest(5).to_dict(),
                        'monthly_trends': aws_product_costs.to_dict()
                    }
            
            # 3. Cost per Service Group per Month
            if 'Service Group New' in df.columns:
                service_group_costs = df.groupby('Service Group New')[month_columns].sum()
                # Filter out empty rows
                service_group_costs = service_group_costs[service_group_costs.sum(axis=1) > 0]
                if not service_group_costs.empty:
                    insights['cost_per_service_group'] = {
                        'summary': f"Analysis across {len(service_group_costs)} service groups", 
                        'top_service_groups': service_group_costs.sum(axis=1).nlargest(5).to_dict(),
                        'monthly_trends': service_group_costs.to_dict()
                    }
            
            # 4. Overall cost trends
            total_monthly_costs = df[month_columns].sum()
            insights['overall_trends'] = {
                'total_cost_by_month': total_monthly_costs.to_dict(),
                'cost_growth': self._calculate_growth_rates(total_monthly_costs)
            }
            
            # Only add peak/lowest if we have data
            if not total_monthly_costs.empty and total_monthly_costs.sum() > 0:
                insights['overall_trends']['peak_month'] = total_monthly_costs.idxmax()
                insights['overall_trends']['lowest_month'] = total_monthly_costs.idxmin()
            
            return insights
            
        except Exception as e:
            logger.error(f"📊 Error generating insights: {str(e)}")
            return {"error": str(e)}

    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get the date range of the data"""
        month_columns = [col for col in df.columns if col.startswith('2025-')]
        if month_columns:
            return {
                "start_month": min(month_columns),
                "end_month": max(month_columns),
                "total_months": len(month_columns)
            }
        return {}

    def _calculate_growth_rates(self, monthly_costs: pd.Series) -> Dict[str, float]:
        """Calculate month-over-month growth rates"""
        growth_rates = {}
        for i in range(1, len(monthly_costs)):
            prev_month = monthly_costs.iloc[i-1]
            curr_month = monthly_costs.iloc[i]
            if prev_month > 0:
                growth_rate = ((curr_month - prev_month) / prev_month) * 100
                growth_rates[f"{monthly_costs.index[i-1]}_to_{monthly_costs.index[i]}"] = round(growth_rate, 2)
        return growth_rates

    def _generate_recommendations(self, df: pd.DataFrame, insights: Dict[str, Any]) -> list:
        """Generate actionable recommendations based on cost analysis"""
        recommendations = []
        
        try:
            # Check for high-cost business units
            if 'cost_per_business_unit' in insights:
                top_units = insights['cost_per_business_unit']['top_business_units']
                if top_units:
                    top_unit = list(top_units.keys())[0]
                    recommendations.append(f"Focus on optimizing costs for '{top_unit}' - highest spending business unit")
            
            # Check for cost growth trends
            if 'overall_trends' in insights:
                growth_rates = insights['overall_trends'].get('cost_growth', {})
                high_growth = [k for k, v in growth_rates.items() if v > 20]
                if high_growth:
                    recommendations.append(f"Investigate cost spikes in periods: {', '.join(high_growth)}")
            
            # Check for AWS product optimization
            if 'cost_per_aws_product' in insights:
                top_products = insights['cost_per_aws_product']['top_aws_products']
                if top_products:
                    top_product = list(top_products.keys())[0]
                    recommendations.append(f"Consider optimization strategies for '{top_product}' - highest cost AWS service")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"📊 Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations due to data processing error"]