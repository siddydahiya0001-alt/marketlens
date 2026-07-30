"""Export helpers: Excel, HTML report, and chart PNG downloads."""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buffer.getvalue()


def dict_of_frames_to_excel_bytes(sheets: dict) -> bytes:
    """sheets: {sheet_name: DataFrame}"""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=str(name)[:31], index=False)
    return buffer.getvalue()


def chart_to_png_bytes(fig) -> bytes | None:
    """Convert a Plotly figure to PNG bytes using kaleido. Returns None if unavailable."""
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


def build_html_report(context: dict) -> str:
    """Build a simple, self-contained HTML summary report.

    context keys: file_name, sheet_name, question, columns_used, quality_findings,
    main_answer, explanation, confidence, recommendations, limitations, generated_at
    """
    generated_at = context.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M")

    def _list_html(items):
        if not items:
            return "<p><em>None</em></p>"
        return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MarketLens Report - {context.get('file_name', '')}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 860px; margin: 40px auto; color: #1f2933; }}
  h1 {{ color: #1a5276; }}
  h2 {{ color: #21618c; border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 32px; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  .box {{ background: #f4f8fb; border-left: 4px solid #2e86c1; padding: 12px 16px; margin: 12px 0; }}
  .warn {{ background: #fdf2e9; border-left: 4px solid #e67e22; padding: 12px 16px; margin: 12px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  td, th {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9em; }}
</style>
</head>
<body>
  <h1>MarketLens Summary Report</h1>
  <p class="meta">Generated {generated_at}</p>

  <h2>File details</h2>
  <p>File: <strong>{context.get('file_name', 'N/A')}</strong> | Worksheet: <strong>{context.get('sheet_name', 'N/A')}</strong></p>

  <h2>Analysis question</h2>
  <p>{context.get('question', 'N/A')}</p>

  <h2>Columns used</h2>
  {_list_html(context.get('columns_used', []))}

  <h2>Data-quality findings</h2>
  {_list_html(context.get('quality_findings', []))}

  <h2>Main result</h2>
  <div class="box">
    <p><strong>{context.get('main_answer', '')}</strong></p>
    <p>{context.get('explanation', '')}</p>
    <p>Confidence: <strong>{context.get('confidence', '')}</strong></p>
  </div>

  <h2>Recommendations</h2>
  {_list_html(context.get('recommendations', []))}

  <h2>Limitations</h2>
  <div class="warn">{_list_html(context.get('limitations', []))}</div>

  <h2>Technical appendix</h2>
  {context.get('technical_appendix_html', '<p><em>Not shown - enable technical details before exporting.</em></p>')}

</body>
</html>"""
    return html


def html_report_to_pdf_bytes(html: str) -> bytes | None:
    """Best-effort HTML -> PDF conversion without external paid services.

    Uses reportlab as a text-based fallback rendering since headless-browser
    PDF engines are not guaranteed to be installed. Returns None if it fails
    so the caller can fall back to offering the HTML report only.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        import re

        text_only = re.sub("<[^<]+?>", "\n", html)
        lines = [line.strip() for line in text_only.splitlines() if line.strip()]

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        for line in lines:
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 4))
        doc.build(story)
        return buffer.getvalue()
    except Exception:
        return None
