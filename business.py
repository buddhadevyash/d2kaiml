import os
import streamlit as st
from google.generativeai import GenerativeModel
import google.generativeai as genai
from PIL import Image
import pytesseract
import pandas as pd
import PyPDF2
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = GenerativeModel("gemini-2.0-flash")

# Load industry context from file
try:
    with open("data/industry.txt", "r") as file:
        industry_context = file.read()
except FileNotFoundError:
    industry_context = "No industry context file found."

# Function to extract text from different file types
def extract_text(file):
    if file.type in ["image/png", "image/jpeg"]:
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    elif file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "text/csv":
        df = pd.read_csv(file)
        return df.to_string()
    else:
        st.error("Unsupported file format")
        return ""

# Function to completely strip all HTML/XML tags for PDF safety
def strip_all_tags(text):
    if text is None:
        return "No content available."
    
    # If text is not a string, convert it to a string
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return "Content could not be converted to text."
    
    # Remove all HTML/XML tags completely
    text = re.sub(r'<[^>]*>', '', text)
    
    # Escape any remaining < or > characters
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    
    # Replace special characters
    text = text.replace('\t', '    ')  # Replace tabs with spaces
    text = text.replace('\u2022', '* ')  # Replace bullet points
    text = text.replace('*', '• ')  # Replace asterisks with bullet points
    
    # Remove any other potentially problematic characters
    text = re.sub(r'[^\x20-\x7E\n\r\t ]', ' ', text)
    
    return text

# Function to generate Gantt charts
def generate_gantt_chart(tasks, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Convert dates to datetime objects
    for task in tasks:
        task["Start"] = datetime.strptime(task["Start"], "%Y-%m-%d")
        task["End"] = datetime.strptime(task["End"], "%Y-%m-%d")
    
    # Plot each task
    for i, task in enumerate(tasks):
        ax.barh(task["Task"], task["End"] - task["Start"], left=task["Start"], color='skyblue', edgecolor='black')
    
    # Format the x-axis to show dates
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.xlabel("Timeline")
    plt.ylabel("Tasks")
    plt.title(title)
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(f"{title.replace(' ', '_')}_gantt.png", bbox_inches='tight')
    plt.close()

# Function to parse tables from the AI response
def parse_timeline_tables(text):
    # Look for sections that appear to contain tables
    table_pattern = r"(?:Timeline|Implementation Timeline|Action Plan|Quarterly Plan)[\s\S]*?(?:\|\s*[-]+\s*\|\s*[-]+\s*\|[\s\S]*?(?=\n\n|\Z))"
    
    tables = re.findall(table_pattern, text, re.IGNORECASE)
    
    result = []
    for table_text in tables:
        lines = table_text.strip().split('\n')
        
        # Extract the table header (if any)
        header = ""
        for i, line in enumerate(lines):
            if '|' in line:
                if i > 0 and '|' not in lines[i-1]:
                    header = lines[i-1].strip()
                break
        
        # Extract the table rows
        rows = []
        for line in lines:
            if '|' in line:
                # Clean and split the row
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if cells and any(cells):  # Skip empty rows
                    rows.append(cells)
        
        if rows:
            result.append({"header": header, "rows": rows})
    
    return result

# Function to generate PDF safely with Gantt charts
def generate_pdf(business_models):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.darkgreen
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8
    )
    
    normal_style = styles['Normal']
    
    # Build the document content
    content = []
    
    try:
        # Title
        content.append(Paragraph("Custom Business Models Report", title_style))
        content.append(Spacer(1, 12))
        
        # Revenue Model
        content.append(Paragraph("1. Revenue Model", subtitle_style))
        revenue_text = strip_all_tags(business_models.get("revenue_model", "No revenue model generated."))
        content.append(Paragraph(revenue_text, normal_style))
        content.append(Spacer(1, 12))
        
        # Add timeline tables for revenue model
        if "revenue_timeline" in business_models and business_models["revenue_timeline"]:
            content.append(Paragraph("Revenue Model Implementation Timeline", table_header_style))
            for table_data in business_models["revenue_timeline"]:
                if table_data["header"]:
                    content.append(Paragraph(table_data["header"], normal_style))
                
                if table_data["rows"]:
                    table_style = TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ])
                    
                    table = Table(table_data["rows"])
                    table.setStyle(table_style)
                    content.append(table)
                    content.append(Spacer(1, 10))
        
        content.append(Spacer(1, 12))
        
        # Growth Model
        content.append(Paragraph("2. Growth Model", subtitle_style))
        growth_text = strip_all_tags(business_models.get("growth_model", "No growth model generated."))
        content.append(Paragraph(growth_text, normal_style))
        content.append(Spacer(1, 12))
        
        # Add timeline tables for growth model
        if "growth_timeline" in business_models and business_models["growth_timeline"]:
            content.append(Paragraph("Growth Model Implementation Timeline", table_header_style))
            for table_data in business_models["growth_timeline"]:
                if table_data["header"]:
                    content.append(Paragraph(table_data["header"], normal_style))
                
                if table_data["rows"]:
                    table_style = TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ])
                    
                    table = Table(table_data["rows"])
                    table.setStyle(table_style)
                    content.append(table)
                    content.append(Spacer(1, 10))
        
        content.append(Spacer(1, 12))
        
        # Profit Maximization Model
        content.append(Paragraph("3. Profit Maximization Model", subtitle_style))
        profit_text = strip_all_tags(business_models.get("profit_model", "No profit model generated."))
        content.append(Paragraph(profit_text, normal_style))
        content.append(Spacer(1, 12))
        
        # Add timeline tables for profit model
        if "profit_timeline" in business_models and business_models["profit_timeline"]:
            content.append(Paragraph("Profit Maximization Implementation Timeline", table_header_style))
            for table_data in business_models["profit_timeline"]:
                if table_data["header"]:
                    content.append(Paragraph(table_data["header"], normal_style))
                
                if table_data["rows"]:
                    table_style = TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lavender),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ])
                    
                    table = Table(table_data["rows"])
                    table.setStyle(table_style)
                    content.append(table)
                    content.append(Spacer(1, 10))
        
        content.append(Spacer(1, 12))
        
        # Projections
        content.append(Paragraph("4. Financial Projections", subtitle_style))
        projections_text = strip_all_tags(business_models.get("projections", "No projections generated."))
        content.append(Paragraph(projections_text, normal_style))
        
        # Build the document
        doc.build(content)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        # Create a simple error PDF with minimal content
        buffer = BytesIO()
        simple_doc = SimpleDocTemplate(buffer, pagesize=letter)
        error_content = [Paragraph("Error generating PDF. The content contains formatting that could not be processed.", normal_style)]
        simple_doc.build(error_content)
        buffer.seek(0)
        return buffer

# Function to generate business models using Gemini
def generate_business_models(financial_params, annual_report_text):
    prompt = f"""
    Based on the following financial parameters and annual report, generate three detailed business models:
    
    Financial Parameters:
    - Annual Revenue: ${financial_params["annual_revenue"]:,}
    - Profit Margin: {financial_params["profit_margin"]}%
    - Market Growth Rate: {financial_params["market_growth_rate"]}%
    - Customer Acquisition Cost: ${financial_params["customer_acquisition_cost"]}
    - Customer Lifetime Value: ${financial_params["customer_lifetime_value"]}
    
    Annual Report Summary:
    {annual_report_text[:3000]}
    
    Industry Context:
    {industry_context[:1000]}
    
    Generate three specific business models:
    1. Revenue Model: A detailed model focused on diversifying and optimizing revenue streams
    2. Growth Model: A strategic approach to scale operations and market share
    3. Profit Maximization Model: Specific tactics to maximize profit in the next fiscal year
    
    Also include a fourth section for financial projections for the next fiscal year based on these models.
    
    VERY IMPORTANT: For each model, include a detailed implementation timeline table with the following columns:
    - Timeline/Quarter
    - Action/Measure
    - Expected Impact
    - KPI/Metrics
    
    Format the timeline as a markdown table with | symbols for columns.
    Example table format:
    
    | Timeline | Action/Measure | Expected Impact | KPI/Metrics |
    | -------- | -------------- | --------------- | ----------- |
    | Q1 2025  | Action 1       | Impact 1        | KPI 1       |
    | Q2 2025  | Action 2       | Impact 2        | KPI 2       |
    
    IMPORTANT: Format your response with clear section headers (1. Revenue Model, 2. Growth Model, 3. Profit Maximization Model, 4. Financial Projections).
    Do not use HTML tags in your response. Use plain text formatting with asterisks for emphasis or bullet points.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Split the response into sections
        result = {}
        
        # Extract sections using regex patterns
        revenue_pattern = r'(?i)(?:^|\n)(?:1\.?\s*|#\s*|)(?:revenue model|revenue)'
        growth_pattern = r'(?i)(?:^|\n)(?:2\.?\s*|#\s*|)(?:growth model|growth)'
        profit_pattern = r'(?i)(?:^|\n)(?:3\.?\s*|#\s*|)(?:profit|profit maximization)'
        projection_pattern = r'(?i)(?:^|\n)(?:4\.?\s*|#\s*|)(?:financial projection|projection)'
        
        # Find all section positions
        revenue_match = re.search(revenue_pattern, response_text)
        growth_match = re.search(growth_pattern, response_text)
        profit_match = re.search(profit_pattern, response_text)
        projection_match = re.search(projection_pattern, response_text)
        
        # Default end of text
        text_end = len(response_text)
        
        # Get indices or set to end of string
        revenue_start = revenue_match.start() if revenue_match else 0
        growth_start = growth_match.start() if growth_match else text_end
        profit_start = profit_match.start() if profit_match else text_end
        projection_start = projection_match.start() if projection_match else text_end
        
        # Create a list of section boundaries
        sections = [
            ("revenue_model", revenue_start, growth_start if growth_start > revenue_start else text_end),
            ("growth_model", growth_start, profit_start if profit_start > growth_start else text_end),
            ("profit_model", profit_start, projection_start if projection_start > profit_start else text_end),
            ("projections", projection_start, text_end)
        ]
        
        # Extract each section and look for tables
        for section_name, start, end in sections:
            if start < end and start < text_end:
                section_text = response_text[start:end].strip()
                result[section_name] = section_text
                
                # Extract timeline tables from each section
                timeline_tables = parse_timeline_tables(section_text)
                if timeline_tables:
                    result[f"{section_name.split('_')[0]}_timeline"] = timeline_tables
            else:
                result[section_name] = f"No {section_name.replace('_', ' ')} found in response."
        
        return result
        
    except Exception as e:
        st.error(f"Error generating business models: {str(e)}")
        return {
            "revenue_model": "Error generating model.",
            "growth_model": "Error generating model.",
            "profit_model": "Error generating model.",
            "projections": "Error generating projections."
        }

# Streamlit UI with improved styling
st.set_page_config(
    page_title="AI Business Model Generator",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4285F4;
    }
    h1 {
        color: #1E3A8A;
    }
    h2 {
        color: #2563EB;
    }
    h3 {
        color: #3B82F6;
        margin-top: 1.5rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    /* Make file uploader more prominent */
    .stFileUploader {
        padding: 1rem;
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
    }
    /* Add progress bar styling */
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    /* Table styling */
    table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 1.5rem;
    }
    th {
        background-color: #f8fafc;
        padding: 0.75rem;
        text-align: left;
        font-weight: bold;
        border: 1px solid #e2e8f0;
    }
    td {
        padding: 0.75rem;
        border: 1px solid #e2e8f0;
    }
    tr:nth-child(even) {
        background-color: #f8fafc;
    }
    tr:hover {
        background-color: #edf2f7;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("🚀 AI-Powered Business Model Generator")
st.markdown("Generate custom business models with implementation timelines to maximize profit and drive growth")

# Create sidebar for parameters
with st.sidebar:
    st.header("Financial Parameters")
    st.markdown("Enter your company's key financial metrics:")
    
    param1 = st.number_input("Annual Revenue ($)", min_value=0, value=1000000, format="%d", help="Your company's current annual revenue")
    st.markdown("---")
    
    param2 = st.slider("Profit Margin (%)", min_value=0, max_value=100, value=15, help="Current profit margin percentage")
    st.markdown("---")
    
    param3 = st.slider("Market Growth Rate (%)", min_value=-20, max_value=50, value=5, help="Industry growth rate percentage")
    st.markdown("---")
    
    param4 = st.number_input("Customer Acquisition Cost ($)", min_value=0, value=500, help="Average cost to acquire a new customer")
    st.markdown("---")
    
    param5 = st.number_input("Customer Lifetime Value ($)", min_value=0, value=2000, help="Average revenue from a customer over their lifetime")
    
    st.markdown("---")
    st.info("📊 Calculated Metrics")
    st.metric("CAC:LTV Ratio", f"1:{round(param5/param4, 1)}")
    st.metric("Estimated Annual Profit", f"${int(param1 * param2 / 100):,}")

# Main area
col1, col2 = st.columns([2, 1])
with col1:
    st.header("📊 Upload Annual Report")
    uploaded_file = st.file_uploader("Upload your annual report (PDF, Image, or CSV)", type=["pdf", "png", "jpg", "jpeg", "csv"])

with col2:
    st.write("")
    st.write("")
    st.info("We'll analyze your annual report and financial parameters to generate custom business models with implementation timelines optimized for your specific situation.")

# Store financial parameters in a dictionary
financial_params = {
    "annual_revenue": param1,
    "profit_margin": param2,
    "market_growth_rate": param3,
    "customer_acquisition_cost": param4,
    "customer_lifetime_value": param5
}

# Add a progress container
progress_container = st.container()

# Process when generate button is clicked
generate_button = st.button("Generate Business Models with Timelines", use_container_width=True)

if generate_button and uploaded_file is not None:
    # Display progress bar
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    # Update progress
    progress_bar.progress(10)
    status_text.text("Reading annual report...")
    
    # Extract text from annual report
    annual_report_text = extract_text(uploaded_file)
    
    if not annual_report_text:
        st.error("Could not extract text from the uploaded file.")
        st.stop()
    
    # Update progress
    progress_bar.progress(30)
    status_text.text("Analyzing financial data...")
    
    # Update progress
    progress_bar.progress(50)
    status_text.text("Generating business models with Gemini Flash 2.0...")
    
    # Generate business models
    business_models = generate_business_models(financial_params, annual_report_text)
    
    # Update progress
    progress_bar.progress(80)
    status_text.text("Preparing results...")
    
    # Display the generated models
    st.markdown("---")
    st.header("📈 Generated Business Models")
    
    tabs = st.tabs(["💰 Revenue Model", "📊 Growth Model", "💎 Profit Maximization", "🔮 Projections"])
    
    # Revenue Model Tab
    with tabs[0]:
        st.markdown(business_models["revenue_model"])
        # Display timeline tables if available
        if "revenue_timeline" in business_models and business_models["revenue_timeline"]:
            st.subheader("Implementation Timeline")
            for table_data in business_models["revenue_timeline"]:
                if table_data["header"]:
                    st.write(table_data["header"])
                
                if table_data["rows"] and len(table_data["rows"]) > 0:
                    # Create DataFrame from rows for display
                    if len(table_data["rows"]) > 1:  # Make sure we have at least header and one data row
                        df = pd.DataFrame(table_data["rows"][1:], columns=table_data["rows"][0])
                        st.dataframe(df, use_container_width=True)
    
    # Growth Model Tab
    with tabs[1]:
        st.markdown(business_models["growth_model"])
        # Display timeline tables if available
        if "growth_timeline" in business_models and business_models["growth_timeline"]:
            st.subheader("Implementation Timeline")
            for table_data in business_models["growth_timeline"]:
                if table_data["header"]:
                    st.write(table_data["header"])
                
                if table_data["rows"] and len(table_data["rows"]) > 0:
                    # Create DataFrame from rows for display
                    if len(table_data["rows"]) > 1:  # Make sure we have at least header and one data row
                        df = pd.DataFrame(table_data["rows"][1:], columns=table_data["rows"][0])
                        st.dataframe(df, use_container_width=True)
    
    # Profit Model Tab
    with tabs[2]:
        st.markdown(business_models["profit_model"])
        # Display timeline tables if available
        if "profit_timeline" in business_models and business_models["profit_timeline"]:
            st.subheader("Implementation Timeline")
            for table_data in business_models["profit_timeline"]:
                if table_data["header"]:
                    st.write(table_data["header"])
                
                if table_data["rows"] and len(table_data["rows"]) > 0:
                    # Create DataFrame from rows for display
                    if len(table_data["rows"]) > 1:  # Make sure we have at least header and one data row
                        df = pd.DataFrame(table_data["rows"][1:], columns=table_data["rows"][0])
                        st.dataframe(df, use_container_width=True)
    
    # Projections Tab
    with tabs[3]:
        st.markdown(business_models["projections"])
    
    # Generate PDF for download
    progress_bar.progress(90)
    status_text.text("Generating PDF report with timelines...")
    
    pdf_buffer = generate_pdf(business_models)
    
    # Complete progress
    progress_bar.progress(100)
    status_text.text("Complete! Your business models with implementation timelines are ready to download.")
    
    # Store progress in session state
    st.session_state.progress = 100
    
    # Provide download button
    st.markdown("---")
    st.subheader("🔄 Download Your Report")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Download Business Models with Timelines as PDF",
            data=pdf_buffer,
            file_name="business_models_timeline_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
elif generate_button and uploaded_file is None:
    st.warning("⚠️ Please upload an annual report file to generate business models.")

# Add explanatory text at the bottom
st.markdown("---")
with st.expander("How to use this tool effectively"):
    st.markdown("""
    ### Maximizing Value from This Tool
    
    1. **Provide accurate financial parameters** - The quality of the models depends on the accuracy of your inputs
    2. **Upload comprehensive annual reports** - More detailed information leads to more specific recommendations
    3. **Review all three models and their implementation timelines** - Each model offers different strategic approaches with specific action plans
    4. **Use the timeline tables to plan your execution** - The tables provide structured quarterly actions with expected impacts and KPIs
    5. **Share the PDF report** with your team to facilitate discussion about strategic direction and implementation
    6. **Track progress against the timeline** - Use the suggested KPIs to measure success of each implementation phase
    """)

# Add metrics dashboard
if "progress" in st.session_state and st.session_state.progress == 100:
    st.markdown("---")
    st.subheader("📊 Key Performance Indicators")
    
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.metric(
            label="Current Revenue", 
            value=f"${financial_params['annual_revenue']:,}",
            delta=f"{financial_params['market_growth_rate']}% Market Growth"
        )
    
    with kpi2:
        st.metric(
            label="Profit Margin", 
            value=f"{financial_params['profit_margin']}%",
            delta="Target: 25%" if financial_params['profit_margin'] < 25 else "Healthy"
        )
    
    with kpi3:
        cac_ltv = financial_params['customer_lifetime_value'] / financial_params['customer_acquisition_cost']
        st.metric(
            label="CAC:LTV Ratio", 
            value=f"1:{cac_ltv:.1f}",
            delta="Good" if cac_ltv >= 3 else "Needs Improvement"
        )