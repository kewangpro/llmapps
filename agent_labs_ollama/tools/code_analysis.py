#!/usr/bin/env python3
"""
Code Analysis Tool
Analyzes code files for patterns, complexity, and potential issues
"""

import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, Any, List

def analyze_code(file_path: str, analysis_type: str = "overview") -> Dict[str, Any]:
    """
    Analyze a code file for various metrics and patterns

    Args:
        file_path: Path to the code file to analyze
        analysis_type: Type of analysis (overview, complexity, patterns, security)

    Returns:
        Dictionary with analysis results
    """
    try:
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            return {"error": f"File does not exist: {file_path}"}

        if not file_path.is_file():
            return {"error": f"Path is not a file: {file_path}"}

        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding or treat as binary
            return {"error": f"Cannot read file as text: {file_path}"}

        lines = content.split('\n')
        file_extension = file_path.suffix.lower()

        # Basic metrics
        basic_metrics = {
            "file_path": str(file_path),
            "file_size_bytes": len(content),
            "line_count": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "file_extension": file_extension
        }

        if analysis_type == "overview" or analysis_type == "all":
            return analyze_overview(content, lines, basic_metrics)
        elif analysis_type == "complexity":
            return analyze_complexity(content, lines, basic_metrics, file_extension)
        elif analysis_type == "patterns":
            return analyze_patterns(content, lines, basic_metrics, file_extension)
        elif analysis_type == "security":
            return analyze_security(content, lines, basic_metrics, file_extension)
        else:
            return {
                "tool": "code_analysis",
                "success": False,
                "error": f"Unknown analysis type: {analysis_type}. Available: overview, complexity, patterns, security"
            }

    except Exception as e:
        return {
            "tool": "code_analysis",
            "success": False,
            "error": str(e)
        }

def analyze_overview(content: str, lines: List[str], basic_metrics: Dict) -> Dict[str, Any]:
    """Basic overview analysis"""
    comment_lines = 0
    blank_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
            comment_lines += 1

    return {
        "tool": "code_analysis",
        "success": True,
        "analysis_type": "overview",
        **basic_metrics,
        "metrics": {
            "blank_lines": blank_lines,
            "comment_lines": comment_lines,
            "code_lines": basic_metrics["non_empty_lines"] - comment_lines,
            "average_line_length": round(len(content) / len(lines), 2) if lines else 0,
            "longest_line": max(len(line) for line in lines) if lines else 0
        },
        "summary": f"File has {basic_metrics['line_count']} total lines with {basic_metrics['non_empty_lines'] - comment_lines} lines of code"
    }

def analyze_complexity(content: str, lines: List[str], basic_metrics: Dict, file_extension: str) -> Dict[str, Any]:
    """Analyze code complexity"""

    # Function/method detection patterns by language
    function_patterns = {
        '.py': [r'def\s+\w+\s*\(', r'class\s+\w+'],
        '.js': [r'function\s+\w+\s*\(', r'const\s+\w+\s*=\s*\(', r'class\s+\w+'],
        '.ts': [r'function\s+\w+\s*\(', r'const\s+\w+\s*=\s*\(', r'class\s+\w+'],
        '.java': [r'public\s+\w+\s+\w+\s*\(', r'private\s+\w+\s+\w+\s*\(', r'class\s+\w+'],
        '.cpp': [r'\w+\s+\w+\s*\(', r'class\s+\w+'],
        '.c': [r'\w+\s+\w+\s*\(']
    }

    # Complexity indicators
    complexity_patterns = [
        r'\bif\b', r'\belse\b', r'\belif\b', r'\bfor\b', r'\bwhile\b',
        r'\btry\b', r'\bcatch\b', r'\bswitch\b', r'\bcase\b'
    ]

    functions = []
    complexity_score = 0

    # Count functions/classes
    if file_extension in function_patterns:
        for pattern in function_patterns[file_extension]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            functions.extend(matches)

    # Count complexity indicators
    for pattern in complexity_patterns:
        complexity_score += len(re.findall(pattern, content, re.IGNORECASE))

    return {
        "tool": "code_analysis",
        "success": True,
        "analysis_type": "complexity",
        **basic_metrics,
        "complexity": {
            "functions_classes_count": len(functions),
            "cyclomatic_complexity_estimate": complexity_score,
            "complexity_per_line": round(complexity_score / basic_metrics["non_empty_lines"], 4) if basic_metrics["non_empty_lines"] > 0 else 0,
            "detected_functions": functions[:10]  # First 10 functions
        },
        "summary": f"Found {len(functions)} functions/classes with estimated complexity score of {complexity_score}"
    }

def analyze_patterns(content: str, lines: List[str], basic_metrics: Dict, file_extension: str) -> Dict[str, Any]:
    """Analyze code patterns and style"""

    patterns_found = []

    # Common patterns to look for
    pattern_checks = {
        "TODO comments": r'(TODO|FIXME|HACK|XXX)',
        "Long lines (>100 chars)": lambda: len([line for line in lines if len(line) > 100]),
        "Empty catch blocks": r'catch\s*\([^)]*\)\s*\{\s*\}',
        "Magic numbers": r'\b\d{2,}\b',  # Numbers with 2+ digits
        "Hardcoded strings": r'"[^"]{10,}"',  # Strings longer than 10 chars
        "Deep nesting": lambda: len([line for line in lines if len(line) - len(line.lstrip()) > 16])  # >4 levels of indentation
    }

    for pattern_name, pattern in pattern_checks.items():
        if callable(pattern):
            count = pattern()
        else:
            count = len(re.findall(pattern, content, re.IGNORECASE))

        if count > 0:
            patterns_found.append({
                "pattern": pattern_name,
                "count": count
            })

    return {
        "tool": "code_analysis",
        "success": True,
        "analysis_type": "patterns",
        **basic_metrics,
        "patterns": patterns_found,
        "summary": f"Found {len(patterns_found)} different code patterns/issues"
    }

def analyze_security(content: str, lines: List[str], basic_metrics: Dict, file_extension: str) -> Dict[str, Any]:
    """Basic security analysis"""

    security_issues = []

    # Common security anti-patterns
    security_patterns = {
        "Hardcoded passwords": r'(password|pwd|pass)\s*=\s*["\'][^"\']+["\']',
        "SQL injection risk": r'(SELECT|INSERT|UPDATE|DELETE).*\+.*',
        "Eval usage": r'\beval\s*\(',
        "Exec usage": r'\bexec\s*\(',
        "Shell injection risk": r'(os\.system|subprocess|shell=True)',
        "Hardcoded API keys": r'(api[_-]?key|token)\s*=\s*["\'][^"\']{20,}["\']',
        "HTTP instead of HTTPS": r'http://[^"\'\s]+',
        "Weak crypto": r'(md5|sha1)\s*\(',
    }

    for issue_name, pattern in security_patterns.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            security_issues.append({
                "issue": issue_name,
                "count": len(matches),
                "examples": matches[:3]  # First 3 examples
            })

    return {
        "tool": "code_analysis",
        "success": True,
        "analysis_type": "security",
        **basic_metrics,
        "security_issues": security_issues,
        "risk_level": "HIGH" if len(security_issues) > 3 else "MEDIUM" if len(security_issues) > 1 else "LOW",
        "summary": f"Found {len(security_issues)} potential security issues"
    }

def main():
    """CLI interface for the code analysis tool"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: code_analysis.py <json_args>"}))
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
        file_path = args.get("file_path", "")
        analysis_type = args.get("analysis_type", "overview")

        if not file_path:
            print(json.dumps({"error": "file_path is required"}))
            sys.exit(1)

        result = analyze_code(file_path, analysis_type)
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON arguments: {e}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()