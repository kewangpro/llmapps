"""
Professional Finance Platform Design System
Colors, typography, spacing, and styling constants
"""

# ============================================================================
# COLOR PALETTE (Light Theme)
# ============================================================================

class Colors:
    """Professional light theme color palette"""

    # Primary Backgrounds
    BG_PRIMARY = '#FFFFFF'      # Main background (white)
    BG_SECONDARY = '#F8F9FA'    # Card/panel background (very light gray)
    BG_TERTIARY = '#E9ECEF'     # Elevated elements (light gray)
    BG_HOVER = '#DEE2E6'        # Interactive hover (medium light gray)

    # Borders
    BORDER_SUBTLE = '#DEE2E6'   # Dividers, borders (light gray)
    BORDER_FOCUS = '#ADB5BD'    # Active borders (medium gray)

    # Text
    TEXT_PRIMARY = '#212529'    # Primary text (almost black)
    TEXT_SECONDARY = '#495057'  # Secondary text (dark gray)
    TEXT_MUTED = '#6C757D'      # Tertiary text (medium gray)

    # Semantic Colors
    SUCCESS_GREEN = '#0F9D58'   # Positive returns (darker green for contrast)
    SUCCESS_BG = '#D1FAE5'      # Success background (light green)

    DANGER_RED = '#DC3545'      # Negative returns (darker red for contrast)
    DANGER_BG = '#FFE5E7'       # Danger background (light red)

    WARNING_YELLOW = '#F59E0B'  # Warnings, neutral
    WARNING_BG = '#FEF3C7'      # Warning background (light yellow)

    INFO_BLUE = '#0D6EFD'       # Informational (darker blue for contrast)
    INFO_BG = '#D0E7FF'         # Info background (light blue)

    # Accent Colors
    ACCENT_PURPLE = '#7C3AED'   # Primary actions (darker purple for contrast)
    ACCENT_CYAN = '#0891B2'     # Secondary actions (darker cyan for contrast)

    # Chart Colors
    CHART_UP = '#0F9D58'        # Candlestick up (green)
    CHART_DOWN = '#DC3545'      # Candlestick down (red)
    CHART_VOLUME = '#6C757D'    # Volume bars (gray)

    CHART_MA_20 = '#F59E0B'     # Moving average 20 (orange)
    CHART_MA_50 = '#7C3AED'     # Moving average 50 (purple)
    CHART_MA_200 = '#0891B2'    # Moving average 200 (cyan)

    CHART_BB_UPPER = '#DC3545'  # Bollinger upper (red)
    CHART_BB_LOWER = '#0D6EFD'  # Bollinger lower (blue)

    # Professional subtle backgrounds (no gradients)
    HEADER_BG = '#F8F9FA'           # Subtle gray for headers
    CARD_BORDER = '#DEE2E6'         # Card borders
    SHADOW_SUBTLE = '0 1px 3px rgba(0, 0, 0, 0.08)'  # Subtle shadow


# ============================================================================
# TYPOGRAPHY
# ============================================================================

class Typography:
    """Typography scale and font families"""

    # Font Families
    FONT_PRIMARY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    FONT_MONO = "'JetBrains Mono', 'Fira Code', 'Consolas', monospace"

    # Font Sizes (rem)
    TEXT_XS = '0.75rem'     # 12px - Captions
    TEXT_SM = '0.875rem'    # 14px - Body small
    TEXT_BASE = '1rem'      # 16px - Body
    TEXT_LG = '1.125rem'    # 18px - Subheading
    TEXT_XL = '1.5rem'      # 24px - Heading
    TEXT_2XL = '2rem'       # 32px - Page title

    # Font Weights
    WEIGHT_NORMAL = '400'
    WEIGHT_MEDIUM = '500'
    WEIGHT_SEMIBOLD = '600'
    WEIGHT_BOLD = '700'


# ============================================================================
# SPACING SYSTEM
# ============================================================================

class Spacing:
    """Consistent spacing scale"""

    SPACE_1 = '0.25rem'   # 4px
    SPACE_2 = '0.5rem'    # 8px
    SPACE_3 = '0.75rem'   # 12px
    SPACE_4 = '1rem'      # 16px
    SPACE_6 = '1.5rem'    # 24px
    SPACE_8 = '2rem'      # 32px
    SPACE_12 = '3rem'     # 48px


# ============================================================================
# COMPONENT STYLES
# ============================================================================

class Styles:
    """Reusable component style generators"""

    @staticmethod
    def card(padding='1.5rem', margin='0'):
        """Standard card style"""
        return {
            'background': Colors.BG_SECONDARY,
            'border': f'1px solid {Colors.BORDER_SUBTLE}',
            'border-radius': '8px',
            'padding': padding,
            'margin': margin,
            'box-shadow': '0 1px 3px rgba(0, 0, 0, 0.3)',
        }

    @staticmethod
    def card_header():
        """Card header style"""
        return {
            'display': 'flex',
            'justify-content': 'space-between',
            'align-items': 'center',
            'margin-bottom': Spacing.SPACE_4,
            'padding-bottom': Spacing.SPACE_3,
            'border-bottom': f'1px solid {Colors.BORDER_SUBTLE}',
        }

    @staticmethod
    def gradient_card(background=None, padding='15px'):
        """Professional card for headers - no gradients"""
        return {
            'background': background or Colors.BG_SECONDARY,
            'color': Colors.TEXT_PRIMARY,
            'padding': padding,
            'border-radius': '8px',
            'margin-bottom': '15px',
            'border': f'1px solid {Colors.BORDER_SUBTLE}',
            'box-shadow': Colors.SHADOW_SUBTLE,
        }

    @staticmethod
    def stat_card():
        """Statistics card style"""
        return {
            'background': Colors.BG_TERTIARY,
            'border': f'1px solid {Colors.BORDER_SUBTLE}',
            'border-radius': '8px',
            'padding': Spacing.SPACE_6,
            'text-align': 'center',
        }


# ============================================================================
# HTML COMPONENT GENERATORS
# ============================================================================

class HTMLComponents:
    """Generate common HTML components with consistent styling"""

    @staticmethod
    def page_header(title: str, subtitle: str = '') -> str:
        """Generate a professional page header"""
        subtitle_html = f"<p style='margin: 5px 0 0 0; color: {Colors.TEXT_SECONDARY}; font-size: {Typography.TEXT_SM};'>{subtitle}</p>" if subtitle else ''

        return f"""
        <div style='background: {Colors.BG_SECONDARY};
                    color: {Colors.TEXT_PRIMARY};
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    box-shadow: {Colors.SHADOW_SUBTLE};'>
            <h1 style='margin: 0; font-size: {Typography.TEXT_2XL}; font-weight: {Typography.WEIGHT_BOLD};'>{title}</h1>
            {subtitle_html}
        </div>
        """

    @staticmethod
    def stat_card(label: str, value: str, change: str = '', icon: str = '') -> str:
        """Generate a stat card"""
        change_html = f"<div style='font-size: {Typography.TEXT_SM}; margin-top: 5px;'>{change}</div>" if change else ''
        icon_html = f"{icon} " if icon else ''

        return f"""
        <div style='background: {Colors.BG_TERTIARY};
                    border: 1px solid {Colors.BORDER_SUBTLE};
                    border-radius: 8px;
                    padding: {Spacing.SPACE_6};
                    text-align: center;'>
            <div style='font-size: {Typography.TEXT_SM};
                        color: {Colors.TEXT_SECONDARY};
                        margin-bottom: 8px;'>{icon_html}{label}</div>
            <div style='font-size: {Typography.TEXT_2XL};
                        font-weight: {Typography.WEIGHT_BOLD};
                        color: {Colors.TEXT_PRIMARY};
                        font-family: {Typography.FONT_MONO};'>{value}</div>
            {change_html}
        </div>
        """

    @staticmethod
    def section_header(title: str, subtitle: str = '') -> str:
        """Generate a professional section header"""
        subtitle_html = f"<p style='margin: 5px 0 0 0; color: {Colors.TEXT_SECONDARY}; font-size: {Typography.TEXT_SM};'>{subtitle}</p>" if subtitle else ''

        return f"""
        <div style='margin: 20px 0 15px 0;
                    padding-bottom: 10px;
                    border-bottom: 2px solid {Colors.BORDER_SUBTLE};'>
            <h2 style='margin: 0;
                       font-size: {Typography.TEXT_XL};
                       font-weight: {Typography.WEIGHT_SEMIBOLD};
                       color: {Colors.TEXT_PRIMARY};'>{title}</h2>
            {subtitle_html}
        </div>
        """

    @staticmethod
    def badge(text: str, type: str = 'info') -> str:
        """Generate a badge (BUY/SELL/HOLD, etc.)"""
        colors = {
            'success': (Colors.SUCCESS_BG, Colors.SUCCESS_GREEN),
            'danger': (Colors.DANGER_BG, Colors.DANGER_RED),
            'warning': (Colors.WARNING_BG, Colors.WARNING_YELLOW),
            'info': (Colors.INFO_BG, Colors.INFO_BLUE),
        }

        bg, fg = colors.get(type, colors['info'])

        return f"""
        <span style='display: inline-flex;
                     align-items: center;
                     padding: {Spacing.SPACE_1} {Spacing.SPACE_3};
                     border-radius: 12px;
                     font-size: {Typography.TEXT_XS};
                     font-weight: {Typography.WEIGHT_SEMIBOLD};
                     text-transform: uppercase;
                     letter-spacing: 0.05em;
                     background: {bg};
                     color: {fg};'>{text}</span>
        """

    @staticmethod
    def error_message(title: str, message: str, suggestions: list = None) -> str:
        """Generate an error message"""
        suggestions_html = ''
        if suggestions:
            suggestions_html = '<div style="margin-top: 15px; font-size: 14px;"><strong>Try:</strong><ul style="margin: 5px 0 0 20px; padding: 0;">'
            for suggestion in suggestions:
                suggestions_html += f'<li>{suggestion}</li>'
            suggestions_html += '</ul></div>'

        return f"""
        <div style='background: {Colors.DANGER_BG};
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid {Colors.DANGER_RED};'>
            <h3 style='margin: 0 0 10px 0; color: {Colors.DANGER_RED};'>❌ {title}</h3>
            <p style='color: {Colors.TEXT_SECONDARY}; margin: 0;'>{message}</p>
            {suggestions_html}
        </div>
        """

    @staticmethod
    def info_message(title: str, message: str) -> str:
        """Generate an info message"""
        return f"""
        <div style='background: {Colors.INFO_BG};
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid {Colors.INFO_BLUE};'>
            <h3 style='margin: 0 0 10px 0; color: {Colors.INFO_BLUE};'>ℹ️ {title}</h3>
            <p style='color: {Colors.TEXT_SECONDARY}; margin: 0;'>{message}</p>
        </div>
        """

    @staticmethod
    def loading_skeleton(height: str = '100px') -> str:
        """Generate a loading skeleton"""
        return f"""
        <div style='background: {Colors.BG_TERTIARY};
                    border-radius: 8px;
                    height: {height};
                    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;'>
        </div>
        <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        </style>
        """

    @staticmethod
    def price_change(price: float, prev_price: float, show_percent: bool = True) -> str:
        """Generate formatted price change display"""
        if prev_price == 0:
            return '<span style="color: {Colors.TEXT_MUTED};">N/A</span>'

        change = price - prev_price
        change_pct = (change / prev_price * 100) if prev_price else 0
        color = Colors.SUCCESS_GREEN if change >= 0 else Colors.DANGER_RED
        symbol = '▲' if change >= 0 else '▼'

        pct_text = f' ({change_pct:+.2f}%)' if show_percent else ''

        return f"""
        <span style='color: {color}; font-weight: {Typography.WEIGHT_SEMIBOLD};'>
            {symbol} ${abs(change):.2f}{pct_text}
        </span>
        """

    @staticmethod
    def disclaimer() -> str:
        """Generate educational disclaimer as footer"""
        return """
        <div style='background: #F8F9FA;
                    border-top: 1px solid #DEE2E6;
                    padding: 12px 20px;
                    text-align: center;
                    font-size: 11px;
                    color: #6C757D;
                    margin-top: 40px;
                    position: relative;
                    bottom: 0;
                    width: 100%;'>
            ⚠️ <strong>Educational Disclaimer:</strong> For educational purposes only. Not financial advice. Past performance does not guarantee future results. Always consult qualified financial professionals before making investment decisions.
        </div>
        """


# ============================================================================
# TABLE STYLES
# ============================================================================

class TableStyles:
    """Professional table styling"""

    @staticmethod
    def get_base_style() -> str:
        """Get base table CSS"""
        return f"""
        <style>
        .pro-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: {Typography.TEXT_SM};
            background: {Colors.BG_SECONDARY};
            border-radius: 8px;
            overflow: hidden;
        }}

        .pro-table th {{
            text-align: left;
            padding: {Spacing.SPACE_3} {Spacing.SPACE_4};
            font-weight: {Typography.WEIGHT_MEDIUM};
            color: {Colors.TEXT_SECONDARY};
            border-bottom: 1px solid {Colors.BORDER_SUBTLE};
            text-transform: uppercase;
            font-size: {Typography.TEXT_XS};
            letter-spacing: 0.05em;
            background: {Colors.BG_TERTIARY};
        }}

        .pro-table td {{
            padding: {Spacing.SPACE_3} {Spacing.SPACE_4};
            border-bottom: 1px solid {Colors.BORDER_SUBTLE};
            color: {Colors.TEXT_PRIMARY};
        }}

        .pro-table tr:hover {{
            background: {Colors.BG_HOVER};
        }}

        .pro-table td.numeric {{
            font-family: {Typography.FONT_MONO};
            text-align: right;
        }}

        .pro-table td.positive {{
            color: {Colors.SUCCESS_GREEN};
            font-weight: {Typography.WEIGHT_SEMIBOLD};
        }}

        .pro-table td.negative {{
            color: {Colors.DANGER_RED};
            font-weight: {Typography.WEIGHT_SEMIBOLD};
        }}
        </style>
        """

    @staticmethod
    def generate_table(headers: list, rows: list, css_classes: list = None) -> str:
        """Generate a professional table"""
        if css_classes is None:
            css_classes = [[] for _ in rows[0]] if rows else []

        table_html = TableStyles.get_base_style()
        table_html += '<table class="pro-table"><thead><tr>'

        for header in headers:
            table_html += f'<th>{header}</th>'

        table_html += '</tr></thead><tbody>'

        for row in rows:
            table_html += '<tr>'
            for i, cell in enumerate(row):
                classes = ' '.join(css_classes[i]) if i < len(css_classes) and css_classes[i] else ''
                table_html += f'<td class="{classes}">{cell}</td>'
            table_html += '</tr>'

        table_html += '</tbody></table>'

        return table_html


# ============================================================================
# PLOTLY CHART THEME
# ============================================================================

class ChartTheme:
    """Professional light theme for Plotly charts"""

    @staticmethod
    def get_layout_template() -> dict:
        """Get Plotly layout template with professional styling"""
        return {
            'template': 'plotly_white',
            'paper_bgcolor': Colors.BG_PRIMARY,
            'plot_bgcolor': Colors.BG_SECONDARY,
            'font': {
                'family': Typography.FONT_PRIMARY,
                'color': Colors.TEXT_PRIMARY,
                'size': 12,
            },
            'xaxis': {
                'gridcolor': Colors.BORDER_SUBTLE,
                'zerolinecolor': Colors.BORDER_SUBTLE,
                'linecolor': Colors.BORDER_SUBTLE,
            },
            'yaxis': {
                'gridcolor': Colors.BORDER_SUBTLE,
                'zerolinecolor': Colors.BORDER_SUBTLE,
                'linecolor': Colors.BORDER_SUBTLE,
            },
            'margin': {'l': 60, 'r': 30, 't': 60, 'b': 60},
            'hovermode': 'x unified',
            'hoverlabel': {
                'bgcolor': Colors.BG_TERTIARY,
                'font': {
                    'family': Typography.FONT_MONO,
                    'size': 12,
                },
            },
        }

    @staticmethod
    def get_candlestick_colors() -> dict:
        """Get candlestick chart colors"""
        return {
            'increasing': {'line': {'color': Colors.CHART_UP}, 'fillcolor': Colors.CHART_UP},
            'decreasing': {'line': {'color': Colors.CHART_DOWN}, 'fillcolor': Colors.CHART_DOWN},
        }
