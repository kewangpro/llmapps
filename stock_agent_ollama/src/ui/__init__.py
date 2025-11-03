"""
User Interface Components

This module provides the web interface for Stock Agent Pro:
- Design system (colors, styles, components)
- Page implementations (analysis, dashboard, trading, portfolio, models)
- Professional light theme
- Wide horizontal layouts
"""

from src.ui.design_system import Colors, HTMLComponents, TableStyles

# Page components available in src.ui.pages
from src.ui import pages

__all__ = [
    'Colors',
    'HTMLComponents',
    'TableStyles',
    'pages',
]
