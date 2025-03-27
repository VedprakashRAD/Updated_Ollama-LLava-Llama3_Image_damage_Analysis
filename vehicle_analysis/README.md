# Vehicle Damage Analysis Application

This application uses Ollama's AI models (LLaVA and LLaVA-Llama3) to analyze vehicle damage from uploaded images.

## Features

- Upload multiple vehicle images for analysis
- Automatically analyze images using two Ollama models:
  - **llava**: Standard LLaVA model for basic analysis
  - **llava-llama3**: Advanced model for more detailed analysis
- Combine results from both models for comprehensive damage assessment
- Generate detailed reports with damage analysis
- Download and print PDF reports
- Responsive web interface

## Requirements

- Python 3.9+
- Ollama (running locally with LLaVA and/or LLaVA-Llama3 models)
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/VedprakashRAD/Ollama_Image_damage_Analysis.git
   cd Ollama_Image_damage_Analysis
   ```

2. Install the required packages:
   ```
   pip install -r vehicle_analysis/requirements.txt
   ```

3. Ensure Ollama is installed and running:
   - Install Ollama from [https://ollama.com/](https://ollama.com/)
   - Pull the required models:
     ```
     ollama pull llava
     ollama pull llava-llama3
     ```

4. Start the application:
   ```
   cd vehicle_analysis
   python main.py
   ```

5. Open a web browser and navigate to http://localhost:8009

## Using the Application

1. Upload vehicle images using the 'Select Images' button
2. Click 'Analyze Images' to process the images
   - The application will use both models if available
   - If only one model is available, it will use that one
3. View the results on the web interface:
   - Each item will show which model provided the analysis
   - When models disagree, both assessments are shown
4. Download the PDF report for printing or sharing

## Analysis Comparison

The application combines results from both models to provide a more comprehensive analysis:

- **Standard LLaVA**: Good for general damage detection and basic cost estimates
- **LLaVA-Llama3**: Better at detailed analysis, color detection, and more accurate cost estimates

When models disagree on a particular assessment, both opinions are shown, allowing for a more informed decision.

## Troubleshooting

- If you encounter an error about the port being in use, you can change the port in the `main.py` file
- Ensure that Ollama is running before starting the application
- If a model is not available, the application will still run with any available model
- For best results, use clear, well-lit images of the vehicle damage

## License

MIT 