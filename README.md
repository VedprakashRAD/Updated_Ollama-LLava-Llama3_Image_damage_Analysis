# Vehicle Damage Analysis

An AI-powered tool for analyzing vehicle damage from images using Ollama and LLaVA models.

## Features

- Upload and analyze vehicle images for damage assessment
- Advanced image preprocessing to enhance analysis quality
- Detailed vehicle damage reports with cost estimates
- Side-by-side model comparison for advanced users
- QR code generation for shareable reports
- PDF report generation with highlighted damage details
- Visual enhancements for color representation and damage highlighting

## Setup and Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running
- LLaVA model pulled into Ollama

### Installation

1. Clone the repository:
```
git clone <repository-url>
cd vehicle-damage-analysis
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Ensure Ollama is running with LLaVA model:
```
ollama pull llava
ollama pull llava-llama3  # Optional but recommended for comparison
```

4. Start the application:
```
cd vehicle_analysis
python main.py
```

5. Access the web interface at http://localhost:8029

## Usage

1. Open the web interface and upload vehicle images
2. Click "Analyze Images" to process the images
3. View the detailed damage analysis
4. Download the PDF report for documentation
5. Use the model comparison view for more detailed insights

## Project Structure

- `/vehicle_analysis`: Main application code
  - `main.py`: FastAPI application and core logic
  - `/templates`: HTML templates for the web interface
  - `/static`: Static assets
  - `/uploads`: Temporary storage for uploaded images
  - `/reports`: Generated PDF reports

## License

MIT 