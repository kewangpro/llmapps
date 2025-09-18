#!/usr/bin/env python3
"""
System Info Tool
Gets system information and metrics
"""

import json
import sys
import platform
import psutil
import os
from datetime import datetime
from typing import Dict, Any

def _get_cpu_freq():
    """Get CPU frequency safely, handling macOS issues"""
    try:
        freq = psutil.cpu_freq()
        return freq._asdict() if freq else {"current": "N/A", "min": "N/A", "max": "N/A"}
    except (OSError, AttributeError):
        return {"current": "N/A", "min": "N/A", "max": "N/A"}

def _safe_cpu_times():
    """Get CPU times safely"""
    try:
        return psutil.cpu_times()._asdict()
    except (OSError, AttributeError):
        return {"user": "N/A", "system": "N/A", "idle": "N/A"}

def get_system_info(metric: str = "overview") -> Dict[str, Any]:
    """
    Get system information based on the requested metric

    Args:
        metric: Type of system info to retrieve (overview, cpu, memory, disk, network)

    Returns:
        Dictionary with system information
    """
    try:
        if metric == "overview" or metric == "all":
            return {
                "tool": "system_info",
                "success": True,
                "metric": metric,
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "platform": platform.system(),
                    "platform_release": platform.release(),
                    "platform_version": platform.version(),
                    "architecture": platform.machine(),
                    "hostname": platform.node(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version()
                },
                "cpu": {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "total_cores": psutil.cpu_count(logical=True),
                    "cpu_usage_percent": psutil.cpu_percent(interval=1),
                    "cpu_frequency": _get_cpu_freq()
                },
                "memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                    "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                    "percentage": psutil.virtual_memory().percent
                },
                "disk": {
                    "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                    "used_gb": round(psutil.disk_usage('/').used / (1024**3), 2),
                    "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                    "percentage": round((psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100, 2)
                }
            }

        elif metric == "cpu":
            return {
                "tool": "system_info",
                "success": True,
                "metric": metric,
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "total_cores": psutil.cpu_count(logical=True),
                    "cpu_usage_percent": psutil.cpu_percent(interval=1),
                    "cpu_frequency": _get_cpu_freq(),
                    "cpu_times": _safe_cpu_times(),
                    "per_cpu_usage": psutil.cpu_percent(interval=1, percpu=True)
                }
            }

        elif metric == "memory":
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                "tool": "system_info",
                "success": True,
                "metric": metric,
                "timestamp": datetime.now().isoformat(),
                "memory": {
                    "virtual": {
                        "total_gb": round(memory.total / (1024**3), 2),
                        "available_gb": round(memory.available / (1024**3), 2),
                        "used_gb": round(memory.used / (1024**3), 2),
                        "percentage": memory.percent,
                        "buffers_gb": round(memory.buffers / (1024**3), 2) if hasattr(memory, 'buffers') else 0,
                        "cached_gb": round(memory.cached / (1024**3), 2) if hasattr(memory, 'cached') else 0
                    },
                    "swap": {
                        "total_gb": round(swap.total / (1024**3), 2),
                        "used_gb": round(swap.used / (1024**3), 2),
                        "free_gb": round(swap.free / (1024**3), 2),
                        "percentage": swap.percent
                    }
                }
            }

        elif metric == "disk":
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            return {
                "tool": "system_info",
                "success": True,
                "metric": metric,
                "timestamp": datetime.now().isoformat(),
                "disk": {
                    "usage": {
                        "total_gb": round(disk_usage.total / (1024**3), 2),
                        "used_gb": round(disk_usage.used / (1024**3), 2),
                        "free_gb": round(disk_usage.free / (1024**3), 2),
                        "percentage": round((disk_usage.used / disk_usage.total) * 100, 2)
                    },
                    "io": disk_io._asdict() if disk_io else None
                }
            }

        elif metric == "network":
            network_io = psutil.net_io_counters()
            try:
                # This may require elevated permissions on some systems
                network_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, PermissionError):
                network_connections = "Permission denied - run with elevated privileges"

            return {
                "tool": "system_info",
                "success": True,
                "metric": metric,
                "timestamp": datetime.now().isoformat(),
                "network": {
                    "io": network_io._asdict() if network_io else None,
                    "active_connections": network_connections,
                    "interfaces": list(psutil.net_if_addrs().keys())
                }
            }

        else:
            return {
                "tool": "system_info",
                "success": False,
                "error": f"Unknown metric: {metric}. Available: overview, cpu, memory, disk, network"
            }

    except Exception as e:
        return {
            "tool": "system_info",
            "success": False,
            "error": str(e)
        }

def main():
    """CLI interface for the system info tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: system_info.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        metric = args.get("metric", "overview")

        result = get_system_info(metric)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()