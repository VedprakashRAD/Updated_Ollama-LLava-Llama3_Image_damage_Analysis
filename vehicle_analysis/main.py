from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import httpx
import os
from typing import List, Dict, Union, Optional
import json
from datetime import datetime
import base64
from PIL import Image
import io
import logging
import sys
import qrcode
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import ImageEnhance

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create directories if they don't exist
for dir_name in ['static', 'uploads', 'reports']:
    os.makedirs(dir_name, exist_ok=True)

# Mount static directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# Templates
templates = Jinja2Templates(directory="templates")

# Available models
MODELS = {
    "llava": "Standard LLaVA model",
    "llava-llama3": "Advanced LLaVA-Llama3 model"
}

async def check_ollama_status():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            return response.status_code == 200
    except:
        return False

async def ensure_model(model_name: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/pull",
                json={"name": model_name}
            )
            return True
    except Exception as e:
        logger.error(f"Error ensuring model {model_name}: {str(e)}")
        return False

async def analyze_image_with_ollama(image_path: str, model: str = "llava") -> Dict[str, str]:
    # Check if Ollama is running
    if not await check_ollama_status():
        logger.error("Ollama service is not running")
        raise HTTPException(
            status_code=503,
            detail="Ollama service is not running. Please make sure Ollama is installed and running."
        )

    # Prepare the image for Ollama
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")
    
    # First, do a pre-analysis to identify vehicle make, model and color more accurately
    try:
        logger.debug(f"Performing pre-analysis with model {model}: {image_path}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            initial_response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": """Look at this vehicle image and answer ONLY these specific questions:
1. What is the exact make and model of this vehicle? (Example: Maruti Suzuki Swift, Honda City, Toyota Innova)
2. What is the precise color of the vehicle? Be very specific with color shade (Example: Pearl White, Midnight Blue, Metallic Silver, Racing Red)
3. Is there visible damage to the vehicle? Answer yes or no.

Format your answer as:
MAKE_MODEL: [vehicle make and model]
COLOR: [specific color name]
DAMAGE: [Yes/No]""",
                    "images": [image_data],
                    "stream": False
                }
            )
            
            initial_result = None
            if initial_response.status_code == 200:
                pre_analysis = initial_response.json()
                initial_result = pre_analysis["response"]
                logger.debug(f"Pre-analysis completed: {initial_result}")
            else:
                logger.warning(f"Pre-analysis failed, proceeding with standard prompt only")
    except Exception as e:
        logger.warning(f"Error during pre-analysis: {str(e)}, proceeding with standard prompt only")
        initial_result = None
    
    # Extract vehicle make/model and color from initial result
    vehicle_make_model = "Unknown"
    vehicle_color = "Unknown"
    has_damage = "Unknown"
    
    if initial_result:
        try:
            for line in initial_result.split('\n'):
                if line.startswith('MAKE_MODEL:'):
                    vehicle_make_model = line.split(':', 1)[1].strip()
                elif line.startswith('COLOR:'):
                    vehicle_color = line.split(':', 1)[1].strip()
                elif line.startswith('DAMAGE:'):
                    has_damage = line.split(':', 1)[1].strip()
        except Exception as e:
            logger.warning(f"Error parsing pre-analysis: {str(e)}")
    
    # Ollama API endpoint for full analysis
    try:
        logger.debug(f"Sending request to Ollama for image analysis with model {model}: {image_path}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Construct prompt with pre-analysis information if available
            damage_focus = ""
            if has_damage == "Yes":
                damage_focus = """Pay extra attention to all damage details. Ensure you describe:
- The exact nature of each damaged component (dents, scratches, cracks, etc.)
- The severity of damage (minor, moderate, severe)
- The precise location of damage on the vehicle
- Any visible structural issues"""
            elif has_damage == "No":
                damage_focus = """This vehicle appears undamaged. Specifically confirm if there are truly no damages visible, or if there might be minor scratches or dents that could be overlooked at first glance."""
            
            enhanced_prompt = f"""Analyze this vehicle image and provide a detailed report in the following format:

VEHICLE DETAILS
- Make & Model: {vehicle_make_model if vehicle_make_model != "Unknown" else "[Identify the exact make, model and variant]"}
- Year of Manufacture: [year if visible, otherwise 'Not visible']
- Odometer Reading: [reading if visible, otherwise 'Not visible']
- Vehicle Color: {vehicle_color if vehicle_color != "Unknown" else "[Be extremely precise about the color - use specific shade names]"}
- Vehicle Registration Number: [number if visible, otherwise 'Not visible']
- Registered State: [state name if visible, otherwise 'Not visible']
- RTO Name: [RTO details if visible, otherwise 'Not visible']
- Vehicle Angle: [front, rear, side, interior, or multiple angles]

DASHBOARD & CONDITION
- Warning Lights: [status of warning lights, or 'Not visible']
- Fuel Level: [current fuel level, or 'Not visible']
- Engine Temperature: [temperature reading, or 'Not visible']
- Speedometer Reading: [current speed, or 'Not visible']
- Tachometer Reading: [RPM reading, or 'Not visible']

DAMAGE ANALYSIS
{damage_focus}
First state clearly whether the vehicle appears damaged or not. If the vehicle shows no signs of damage, state "No visible damage detected" and skip to Mechanical Assessment.
- Front Bumper & Grille: [condition details or 'No damage detected']
- Headlights & Fender: [condition details or 'No damage detected']
- Hood Alignment & Latch: [condition details or 'No damage detected']
- Side Panels: [condition details or 'No damage detected']
- Doors: [condition details or 'No damage detected']
- Windows & Mirrors: [condition details or 'No damage detected']
- Rear Components: [condition details or 'No damage detected']

MECHANICAL ASSESSMENT
- Engine Bay: [visible condition or 'Not visible']
- Battery & Electrical: [status or 'Not visible']
- Suspension & Underbody: [visible issues or 'Not visible']
- Fluid Leaks: [any visible leaks or 'None detected']
- Tire Condition: [all tires status or 'Not clearly visible']

COST ESTIMATES
If no damage is detected, state "No repairs needed - ₹0" and skip the detailed breakdown.
For any damage detected:
- Front End Repairs: [cost range in ₹]
- Body Work: [cost range in ₹]
- Mechanical Repairs: [cost range in ₹]
- Electrical Work: [cost range in ₹]
- Paint & Finishing: [cost range in ₹]
- Total Estimated Cost: [total range in ₹]

RECOMMENDATION
Based on the analysis, provide a brief conclusion and recommendation:
1. Is the vehicle in good condition?
2. Are there any critical repairs needed?
3. Should the vehicle undergo professional inspection?

Please be specific about all visible damage, issues, provide accurate color identification, and detailed cost estimates in INR (₹). If the vehicle shows no signs of damage, clearly state this fact."""

            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": enhanced_prompt,
                    "images": [image_data],
                    "stream": False
                }
            )
            
            logger.debug(f"Ollama response status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Ollama analysis completed successfully for {image_path} with model {model}")
                return {
                    "model": model,
                    "response": result["response"]
                }
            else:
                logger.error(f"Error from Ollama service: {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error from Ollama service: {response.text}"
                )
    except httpx.TimeoutException:
        logger.error(f"Timeout during analysis of {image_path} with model {model}")
        raise HTTPException(status_code=504, detail=f"Analysis timeout for model {model} - please try again")
    except Exception as e:
        logger.error(f"Error during analysis of {image_path} with model {model}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during analysis with model {model}: {str(e)}")

async def generate_pdf_report(analysis_results: List[Dict[str, Union[str, Dict[str, str]]]], image_paths: List[str], timestamp: str):
    report_path = f"reports/vehicle_report_{timestamp}.pdf"
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Create custom styles
    damage_style = ParagraphStyle(
        'DamageStyle',
        parent=styles['Normal'],
        textColor=colors.HexColor('#e74c3c'),
        fontName='Helvetica-Bold'
    )
    
    no_damage_style = ParagraphStyle(
        'NoDamageStyle',
        parent=styles['Normal'],
        textColor=colors.HexColor('#2ecc71'),
        fontName='Helvetica-Bold'
    )
    
    cost_style = ParagraphStyle(
        'CostStyle',
        parent=styles['Normal'],
        textColor=colors.HexColor('#16a085'),
        fontName='Helvetica-Bold'
    )
    
    color_style = ParagraphStyle(
        'ColorStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold'
    )
    
    header_with_border = ParagraphStyle(
        'HeaderWithBorder',
        parent=styles['Heading3'],
        borderWidth=1,
        borderColor=colors.HexColor('#eee'),
        borderPadding=5,
        backColor=colors.HexColor('#f9f9f9'),
        borderRadius=2
    )
    
    try:
        # Create header with lightning bolt emoji
        header_table_data = [
            [
                Paragraph(
                    '<font color="#FFB800" size="24">⚡</font>',
                    ParagraphStyle('emoji', alignment=1, leading=30)
                ),
                Paragraph(
                    "ReadyAssist",
                    ParagraphStyle('Header', parent=styles['Heading1'], fontSize=24)
                )
            ]
        ]
        
        header_table = Table(header_table_data, colWidths=[0.5*inch, 5*inch])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ('ALIGN', (1, 0), (1, 0), 'LEFT'),
        ]))
        
        story.append(header_table)
        story.append(Spacer(1, 10))
        
        # Add subtitle
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#666666'),
            alignment=1,  # Center alignment
            spaceAfter=20
        )
        story.append(Paragraph("Vehicle Damage Analysis Report", subtitle_style))
        
        # Add timestamp
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=1  # Center alignment
        )
        formatted_date = datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
        story.append(Paragraph(f"Report Generated: {formatted_date}", date_style))
        story.append(Spacer(1, 20))

        # Generate QR code for sharing the report
        report_filename = f"vehicle_report_{timestamp}.pdf"
        report_url = f"http://localhost:8009/report/{report_filename}"
        qr_img_path = f"reports/qr_{timestamp}.png"
        
        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(report_url)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img.save(qr_img_path)
        
        # Add QR code to the report with caption
        story.append(Paragraph("Scan to access digital report:", styles['Normal']))
        story.append(Spacer(1, 5))
        story.append(RLImage(qr_img_path, width=1.5*inch, height=1.5*inch))
        story.append(Spacer(1, 10))

        # Add vehicle images
        for i, image_path in enumerate(image_paths):
            if os.path.exists(image_path):
                try:
                    # Verify image can be opened before adding to PDF
                    with Image.open(image_path) as img:
                        # If image opens successfully, add it to the PDF
                        story.append(Paragraph(f"Vehicle Image {i+1}", styles['Heading2']))
                        story.append(RLImage(image_path, width=6*inch, height=4*inch))
                        story.append(Spacer(1, 12))
                except Exception as e:
                    logger.error(f"Error loading vehicle image {image_path}: {str(e)}")
                    # Add error message instead of image
                    story.append(Paragraph(f"Vehicle Image {i+1} (Failed to load)", styles['Heading2']))
                    story.append(Paragraph("Error: Unable to load image", styles['Normal']))
                    story.append(Spacer(1, 12))

        # Add analysis results
        for i, analysis_data in enumerate(analysis_results):
            story.append(Paragraph(f"Analysis for Image {i+1}", styles['Heading2']))
            story.append(Spacer(1, 6))
            
            # Add the models used for this analysis
            models_used = []
            for model_result in analysis_data.get("model_results", []):
                models_used.append(model_result.get("model", "unknown"))
            
            if models_used:
                story.append(Paragraph(f"Models used: {', '.join(models_used)}", styles['Italic']))
                story.append(Spacer(1, 12))
            
            # Get the combined analysis from both models or just the response if only one model
            analysis = analysis_data.get("combined_analysis", "")
            if not analysis and "response" in analysis_data:
                analysis = analysis_data["response"]
            
            # Split analysis into sections and format them
            sections = analysis.split('\n\n')
            for section in sections:
                if section.strip():
                    # Add section title
                    section_lines = section.split('\n')
                    section_title = section_lines[0].strip()
                    
                    # Use special styling for damage section
                    if section_title == "DAMAGE ANALYSIS":
                        has_no_damage = any("No visible damage detected" in line for line in section_lines)
                        bg_color = "#f9f9e0" if has_no_damage else "#fff0f0"
                        
                        # Create a styled title for the damage section
                        damage_title_style = ParagraphStyle(
                            'DamageTitle',
                            parent=styles['Heading3'],
                            borderWidth=1,
                            borderColor=colors.HexColor("#e0e0e0"),
                            borderPadding=5,
                            backColor=colors.HexColor(bg_color)
                        )
                        story.append(Paragraph(section_title, damage_title_style))
                    elif section_title == "RECOMMENDATION":
                        # Special styling for recommendations
                        story.append(Paragraph(section_title, header_with_border))
                    else:
                        story.append(Paragraph(section_title, styles['Heading3']))
                    
                    story.append(Spacer(1, 6))
                    
                    # Add section content with special formatting for certain fields
                    content_lines = section_lines[1:]
                    for line in content_lines:
                        if not line.strip():
                            continue
                        
                        if ":" in line:
                            key, value = line.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Apply special styling for vehicle color
                            if "Vehicle Color" in key and value and value != "Not visible":
                                story.append(Paragraph(f"<b>{key}:</b> <b>{value}</b>", color_style))
                            # Apply special styling for damage details
                            elif (("Bumper" in key or "Panel" in key or "Door" in key or 
                                 "Hood" in key or "Light" in key or "Window" in key) and 
                                value and not "No damage" in value):
                                story.append(Paragraph(f"<b>{key}:</b> {value}", damage_style))
                            # Apply special styling for "No damage detected"
                            elif "No damage" in value:
                                story.append(Paragraph(f"<b>{key}:</b> {value}", no_damage_style))
                            # Apply special styling for cost
                            elif "Cost" in key and "₹" in value:
                                story.append(Paragraph(f"<b>{key}:</b> {value}", cost_style))
                            else:
                                story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
                        else:
                            story.append(Paragraph(line, styles['Normal']))
                        
                    story.append(Spacer(1, 12))
        
        # Add footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=1  # Center alignment
        )
        story.append(Spacer(1, 30))
        story.append(Paragraph("Sundaravijayam Automobile Services Private Limited", footer_style))
        story.append(Paragraph("839/2, 24th Main Rd, Behind Thirumala Theatre", footer_style))
        story.append(Paragraph("1st Sector, HSR Layout, Bengaluru, Karnataka 560102", footer_style))
        story.append(Paragraph("www.readyassist.in", footer_style))
        
        # Build the PDF
        doc.build(story)
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")

def combine_analysis_results(results: List[Dict[str, str]]) -> str:
    """Combine multiple model analysis results into a single, comprehensive report."""
    if not results:
        return ""
    
    if len(results) == 1:
        return results[0]["response"]
    
    # Structure to store the combined data
    combined_sections = {}
    
    for result in results:
        analysis = result["response"]
        model = result["model"]
        
        # Split into sections
        sections = analysis.split('\n\n')
        for section in sections:
            if not section.strip():
                continue
                
            section_lines = section.split('\n')
            section_title = section_lines[0].strip()
            
            if section_title not in combined_sections:
                combined_sections[section_title] = {
                    "content": {},
                    "models": []
                }
            
            # Add model to the list of models that provided info for this section
            if model not in combined_sections[section_title]["models"]:
                combined_sections[section_title]["models"].append(model)
            
            # Process section content (items after the title)
            for line in section_lines[1:]:
                if not line.strip():
                    continue
                    
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key not in combined_sections[section_title]["content"]:
                        combined_sections[section_title]["content"][key] = []
                    
                    # Add the value if it's not already present
                    if value not in [v.strip() for v in combined_sections[section_title]["content"][key]]:
                        combined_sections[section_title]["content"][key].append(value)
    
    # Generate combined report
    combined_analysis = []
    for section_title, section_data in combined_sections.items():
        combined_analysis.append(section_title)
        
        for key, values in section_data["content"].items():
            # For a single value, just add it
            if len(values) == 1:
                combined_analysis.append(f"- {key}: {values[0]}")
            # For multiple values, combine them or indicate different model opinions
            else:
                # Remove duplicates while preserving order
                unique_values = []
                for value in values:
                    if value not in unique_values:
                        unique_values.append(value)
                
                if len(unique_values) == 1:
                    combined_analysis.append(f"- {key}: {unique_values[0]}")
                else:
                    combined_analysis.append(f"- {key}: {' | '.join(unique_values)}")
        
        combined_analysis.append("")  # Add empty line after section
    
    return "\n".join(combined_analysis)

async def preprocess_image(image_path: str) -> str:
    """Preprocess the image to enhance quality for better analysis"""
    try:
        output_path = f"{image_path.rsplit('.', 1)[0]}_processed.{image_path.rsplit('.', 1)[1]}"
        
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Moderate enhancement of contrast and sharpness
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.2)  # Increase contrast by 20%
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(img)
            img = sharpness_enhancer.enhance(1.3)  # Increase sharpness by 30%
            
            # Enhance brightness slightly
            brightness_enhancer = ImageEnhance.Brightness(img)
            img = brightness_enhancer.enhance(1.1)  # Increase brightness by 10%
            
            # Resize if extremely large to improve processing speed
            max_dimension = 1800  # Maximum dimension in pixels
            if max(img.size) > max_dimension:
                # Maintain aspect ratio
                if img.width > img.height:
                    new_width = max_dimension
                    new_height = int(img.height * (max_dimension / img.width))
                else:
                    new_height = max_dimension
                    new_width = int(img.width * (max_dimension / img.height))
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save the processed image
            img.save(output_path, quality=95)
            
            logger.debug(f"Image preprocessed: {image_path} -> {output_path}")
            return output_path
            
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        # Return original image path if preprocessing fails
        return image_path

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_images(files: List[UploadFile] = File(...)):
    timestamp = str(datetime.now().timestamp())
    image_paths = []
    processed_image_paths = []
    analysis_results = []
    
    # Check and ensure both models are available
    llava_available = await ensure_model("llava")
    llava_llama3_available = await ensure_model("llava-llama3")
    
    if not (llava_available or llava_llama3_available):
        raise HTTPException(
            status_code=503, 
            detail="No Ollama models available. Please ensure Ollama is running with at least one model."
        )
    
    models_to_use = []
    if llava_available:
        models_to_use.append("llava")
    if llava_llama3_available:
        models_to_use.append("llava-llama3")
    
    logger.info(f"Using models: {', '.join(models_to_use)}")
    
    try:
        # Save and analyze each uploaded image
        for i, file in enumerate(files):
            try:
                # Save the uploaded file
                file_path = f"uploads/{timestamp}_{i}_{file.filename}"
                image_paths.append(file_path)
                
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Preprocess the image
                processed_path = await preprocess_image(file_path)
                processed_image_paths.append(processed_path)
                
                # Analyze the image with all available models
                all_model_results = []
                for model in models_to_use:
                    try:
                        model_result = await analyze_image_with_ollama(processed_path, model)
                        all_model_results.append(model_result)
                    except Exception as e:
                        logger.error(f"Error analyzing image {processed_path} with model {model}: {str(e)}")
                
                if not all_model_results:
                    # If all models failed, add an error message
                    analysis_results.append({
                        "error": "All models failed to analyze this image",
                        "model_results": []
                    })
                else:
                    # Combine results from all models
                    combined_analysis = combine_analysis_results(all_model_results)
                    analysis_results.append({
                        "model_results": all_model_results,
                        "combined_analysis": combined_analysis
                    })
                    
            except Exception as e:
                logger.error(f"Error processing image {file.filename}: {str(e)}")
                continue
        
        # Generate PDF report - pass original images for display
        try:
            report_path = await generate_pdf_report(analysis_results, image_paths, timestamp)
            
            # Prepare simplified analysis results for the frontend
            simplified_results = []
            for result in analysis_results:
                if "combined_analysis" in result:
                    simplified_results.append(result["combined_analysis"])
                elif "model_results" in result and result["model_results"]:
                    # Use the first model's result if there's no combined analysis
                    simplified_results.append(result["model_results"][0]["response"])
                else:
                    simplified_results.append("Analysis failed for this image.")
            
            return JSONResponse({
                "message": "Analysis completed successfully",
                "timestamp": timestamp,
                "analysis_results": simplified_results,
                "image_paths": [path.replace("uploads/", "") for path in image_paths],
                "report_path": report_path.replace("reports/", ""),
                "models_used": models_to_use,
                "full_analysis_results": analysis_results  # Include full results for comparison view
            })
        except Exception as e:
            # Return results even if PDF generation fails
            logger.error(f"PDF generation failed but returning analysis results: {str(e)}")
            return JSONResponse({
                "message": "Analysis completed but PDF generation failed",
                "timestamp": timestamp,
                "analysis_results": simplified_results,
                "image_paths": [path.replace("uploads/", "") for path in image_paths],
                "error": str(e),
                "models_used": models_to_use,
                "full_analysis_results": analysis_results  # Include full results for comparison view
            })
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{filename}")
async def get_report(filename: str):
    report_path = f"reports/{filename}"
    if os.path.exists(report_path):
        return FileResponse(report_path, media_type="application/pdf", filename=filename)
    raise HTTPException(status_code=404, detail="Report not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8029) 