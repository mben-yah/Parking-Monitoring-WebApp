"""
PlateVision — Parking Ticket PDF & APITemplate.io Generator
Supports:
1. APITemplate.io REST API Integration (HTML/CSS + JSON data -> PDF)
2. Native ReportLab PDF Generation (Offline 80mm thermal ticket)
3. Printable HTML Thermal Receipt (Browser native print / save as PDF)
"""

import io
import json
import os
import requests
from datetime import datetime


def generate_native_pdf_ticket(ticket: dict) -> bytes:
    """Generate a clean, printable 80mm PDF ticket using ReportLab."""
    from reportlab.lib.pagesizes import mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.graphics.shapes import Drawing, Rect

    buffer = io.BytesIO()
    page_width = 80 * mm
    page_height = 145 * mm
    doc = SimpleDocTemplate(
        buffer,
        pagesize=(page_width, page_height),
        rightMargin=4 * mm,
        leftMargin=4 * mm,
        topMargin=5 * mm,
        bottomMargin=5 * mm
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TicketTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=16,
        alignment=1,
        textColor=colors.HexColor('#0f172a')
    )
    
    sub_style = ParagraphStyle(
        'TicketSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.HexColor('#64748b')
    )

    bold_style = ParagraphStyle(
        'TicketBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )

    text_style = ParagraphStyle(
        'TicketText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#334155')
    )

    plate_box_style = ParagraphStyle(
        'PlateBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=18,
        alignment=1,
        textColor=colors.HexColor('#0284c7')
    )

    story = []

    # Header
    story.append(Paragraph("🅿️ PARKING TICKET", title_style))
    story.append(Paragraph("Commercial Access Control", sub_style))
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceAfter=3*mm))

    # Ticket ID & Status
    t_id = ticket.get("ticket_id", "TKT-UNKNOWN")
    status = ticket.get("status", "ACTIVE")
    story.append(Paragraph(f"<b>Ticket ID:</b> {t_id}", bold_style))
    story.append(Paragraph(f"<b>Status:</b> <font color='{'#16a34a' if status=='PAID' else '#d97706'}'>{status}</font>", text_style))
    story.append(Spacer(1, 2 * mm))

    # Plate Number Display Box
    plate_text = ticket.get("plate_text", "UNKNOWN")
    story.append(Paragraph(f"🚘 {plate_text}", plate_box_style))
    story.append(Spacer(1, 3 * mm))

    # Table Details
    entry_val = str(ticket.get("entry_time", ""))[:19].replace("T", " ")
    exit_val = str(ticket.get("exit_time", "Active"))[:19].replace("T", " ") if ticket.get("exit_time") else "In Parking"
    rate_h = ticket.get("rate_per_hour", 10.0)
    amount = ticket.get("amount_paid", ticket.get("total_amount", 0.0))

    data = [
        [Paragraph("<b>Entry Time:</b>", text_style), Paragraph(entry_val, text_style)],
        [Paragraph("<b>Exit Time:</b>", text_style), Paragraph(exit_val, text_style)],
        [Paragraph("<b>Hourly Rate:</b>", text_style), Paragraph(f"{rate_h:.2f} MAD/hr", text_style)],
        [Paragraph("<b>Total Amount:</b>", bold_style), Paragraph(f"<b>{amount:.2f} MAD</b>", bold_style)],
    ]

    t = Table(data, colWidths=[30 * mm, 42 * mm])
    t.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('LINEBELOW', (0,-1), (-1,-1), 1, colors.HexColor('#0284c7')),
    ]))
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # Simulated Barcode Drawing
    d = Drawing(72 * mm, 12 * mm)
    d.add(Rect(0, 0, 72 * mm, 12 * mm, fillColor=colors.HexColor('#f8fafc'), strokeColor=colors.HexColor('#cbd5e1')))
    import random
    x = 4 * mm
    while x < 68 * mm:
        w = random.choice([0.8, 1.5, 2.2]) * mm
        d.add(Rect(x, 2 * mm, w, 8 * mm, fillColor=colors.HexColor('#0f172a'), strokeColor=None))
        x += w + random.choice([0.6, 1.2]) * mm
    story.append(d)

    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Thank you for using Magnetite Vision Parking System by Ben Yahia", sub_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_apitemplate_pdf(ticket: dict, api_key: str | None = None, template_id: str | None = None) -> bytes | str:
    """APITemplate.io Integration Endpoint."""
    api_key = api_key or os.environ.get("APITEMPLATE_API_KEY")
    if not api_key:
        raise ValueError("APITEMPLATE_API_KEY is not configured in environment or request.")

    endpoint = "https://rest.apitemplate.io/v2/create-pdf-from-html"
    html_content = generate_html_ticket(ticket, printable=False)
    
    payload = {
        "html": html_content,
        "css": "body { font-family: sans-serif; padding: 20px; }",
        "data": ticket
    }
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    response = requests.post(endpoint, json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    res_json = response.json()
    
    pdf_url = res_json.get("download_url") or res_json.get("transaction_ref")
    if not pdf_url:
        raise RuntimeError(f"APITemplate.io error: {res_json}")
        
    pdf_bytes = requests.get(pdf_url, timeout=15).content
    return pdf_bytes


def generate_html_ticket(ticket: dict, printable: bool = True) -> str:
    """Generate printable HTML thermal ticket."""
    t_id = ticket.get("ticket_id", "TKT-UNKNOWN")
    plate = ticket.get("plate_text", "UNKNOWN")
    status = ticket.get("status", "ACTIVE")
    entry_val = str(ticket.get("entry_time", ""))[:19].replace("T", " ")
    exit_val = str(ticket.get("exit_time", "Active In Parking"))[:19].replace("T", " ") if ticket.get("exit_time") else "In Parking"
    rate_h = ticket.get("rate_per_hour", 10.0)
    amount = ticket.get("amount_paid", ticket.get("total_amount", 0.0))

    print_btn = """
    <div style="text-align: center; margin-top: 20px;" class="no-print">
        <button onclick="window.print()" style="background: #0284c7; color: white; border: none; padding: 10px 20px; font-size: 14px; font-weight: bold; border-radius: 8px; cursor: pointer;">
            🖨️ Print Ticket / Save PDF
        </button>
    </div>
    """ if printable else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Parking Ticket — {t_id}</title>
    <style>
        @page {{ size: 80mm 140mm; margin: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            width: 76mm;
            margin: 0 auto;
            padding: 10px;
            background: #f8fafc;
            color: #0f172a;
        }}
        .ticket-card {{
            background: white;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            padding: 14px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}
        .header {{ text-align: center; border-bottom: 2px dashed #cbd5e1; padding-bottom: 10px; margin-bottom: 12px; }}
        .header h2 {{ margin: 0; font-size: 18px; color: #0f172a; }}
        .header p {{ margin: 3px 0 0; font-size: 11px; color: #64748b; }}
        .plate-box {{
            background: #f0f9ff;
            border: 2px solid #0284c7;
            border-radius: 8px;
            text-align: center;
            padding: 8px;
            font-size: 20px;
            font-weight: 800;
            color: #0284c7;
            letter-spacing: 1px;
            margin: 12px 0;
        }}
        .info-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 12px; }}
        .info-table td {{ padding: 4px 0; }}
        .info-table td.label {{ color: #64748b; font-weight: 500; }}
        .info-table td.value {{ text-align: right; font-weight: 700; color: #1e293b; }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            font-size: 10px;
            font-weight: 700;
            border-radius: 12px;
        }}
        .badge-active {{ background: #fef3c7; color: #d97706; }}
        .badge-paid {{ background: #dcfce7; color: #15803d; }}
        .barcode {{
            text-align: center;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            letter-spacing: 3px;
            background: #f1f5f9;
            padding: 8px;
            border-radius: 6px;
            margin-top: 10px;
        }}
        @media print {{
            .no-print {{ display: none; }}
            body {{ background: white; width: 100%; padding: 0; }}
            .ticket-card {{ border: none; box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="ticket-card">
        <div class="header">
            <h2>🅿️ PARKING TICKET</h2>
            <p>Commercial Access Control</p>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px;">
            <span>ID: <strong>{t_id}</strong></span>
            <span class="badge {'badge-paid' if status=='PAID' else 'badge-active'}">{status}</span>
        </div>

        <div class="plate-box">
            🚘 {plate}
        </div>

        <table class="info-table">
            <tr>
                <td class="label">Entry Time</td>
                <td class="value">{entry_val}</td>
            </tr>
            <tr>
                <td class="label">Exit Time</td>
                <td class="value">{exit_val}</td>
            </tr>
            <tr>
                <td class="label">Rate</td>
                <td class="value">{rate_h:.2f} MAD / hr</td>
            </tr>
            <tr style="border-top: 1px solid #e2e8f0;">
                <td class="label" style="font-weight:700; color:#0f172a; padding-top:8px;">Total Due</td>
                <td class="value" style="font-size:16px; color:#0284c7; padding-top:8px;">{amount:.2f} MAD</td>
            </tr>
        </table>

        <div class="barcode">
            ||||| | |||| ||| |||||| |<br>
            <span style="font-size:10px; letter-spacing:1px;">{t_id}</span>
        </div>
    </div>
    {print_btn}
</body>
</html>
"""

