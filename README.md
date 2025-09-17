# Adaptive OCR - Batch Image Processing

A batch image processing tool that combines AWS Rekognition with AWS Bedrock for enhanced OCR accuracy. The system asynchronously processes multiple images from a folder, using Rekognition as the primary OCR engine and automatically triggering Bedrock multimodal models when confidence levels are low or no text is detected.

## Features

- **Async Batch Processing**: Process all images in a folder efficiently with asynchronous operations
- **Primary OCR**: AWS Rekognition DetectText API for fast, reliable text extraction
- **Adaptive Processing**: AWS Bedrock Mistral Pixtral multimodal model for low-confidence cases
- **Smart Adaptation**: Automatically uses Bedrock when Rekognition confidence < 70% or no text detected
- **Comprehensive Logging**: Detailed console output and structured error handling
- **JSON Export**: Results saved to `ocr_results.json` for further analysis

## Architecture

```
Images Folder → Rekognition OCR → Confidence Check → [Low Confidence/No Text] → Bedrock Fallback → JSON Results
                                        ↓
                                 [High Confidence] → Direct to Results
```

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd adaptive-ocr

# Install dependencies
pip install -r requirements.txt
```

### 2. AWS Configuration

Set up your AWS credentials using environment variables:

```bash
export AWS_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_SESSION_TOKEN=your_session_token_here  # Optional, for temporary credentials
```

Alternatively, you can create a `.env` file with the same variables if preferred.

**Required AWS Services:**
- AWS Rekognition (text detection permissions)
- AWS Bedrock (Mistral Pixtral model access in us-west-2)

### 3. Prepare Images

Place your images in the `images/` folder:
```bash
cp your_images.jpg images/
```

### 4. Run Batch Processing

```bash
# Process all images in the images/ folder
python main.py
```

## Usage Examples

### Basic Processing
```bash
$ python main.py
Found 3 image(s) to process:
  - bar.jpg
  - bus-station.jpg  
  - shell.jpg

Processing bar.jpg...
  Rekognition found: 0 text items
  Bedrock found: 1 text items
  Adaptive processing triggered: No text detected by Rekognition

Processing bus-station.jpg...
  Rekognition found: 6 text items

Processing shell.jpg...
  Rekognition found: 12 text items
  Bedrock found: 1 text items  
  Adaptive processing triggered: Rekognition confidence below 0.7 (lowest: 0.12)

Results saved to ocr_results.json
Summary: 3 successful, 0 failed
```

### Sample Output (ocr_results.json)
```json
{
  "bus-station.jpg": {
    "rekognition_texts": [
      {"text": "1:47", "confidence": 0.9613854217529297},
      {"text": "Grand", "confidence": 0.9963880157470704},
      {"text": "Central", "confidence": 0.9829401397705078},
      {"text": "6", "confidence": 0.9854002380371094},
      {"text": "Minutes", "confidence": 0.9601226806640625},
      {"text": "late.", "confidence": 0.9593543243408204}
    ],
    "bedrock_texts": [],
    "second_opinion_triggered": false,
    "second_opinion_reason": null
  },
  "shell.jpg": {
    "rekognition_texts": [
      {"text": "SHELL", "confidence": 0.5939225387573243},
      {"text": "BACKBAYSIGN", "confidence": 0.2433623504638672}
    ],
    "bedrock_texts": [
      {
        "text": "The text in the image reads \"SHELL.\" The neon sign is in the shape of a seashell, which is a well-known logo for the Shell Oil Company.",
        "confidence": 0.85
      }
    ],
    "second_opinion_triggered": true,
    "second_opinion_reason": "Rekognition confidence below 0.7 (lowest: 0.2433623504638672)"
  }
}
```

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.7 | Minimum confidence to avoid adaptive processing |
| `MAX_IMAGE_SIZE_MB` | 10 | Maximum image file size |
| `BEDROCK_MODEL_ID` | `us.mistral.pixtral-large-2502-v1:0` | Mistral Pixtral multimodal model |
| `BEDROCK_MAX_TOKENS` | 1000 | Max tokens for Bedrock responses |
| `SUPPORTED_IMAGE_FORMATS` | JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP | Accepted image formats |

## Adaptive Processing Logic

The system triggers Bedrock adaptive processing when:

1. **Low Confidence**: Any Rekognition result has confidence < 70%
2. **No Text Detected**: Rekognition finds no text in the image
3. **Processing Errors**: Rekognition fails to process the image

## Project Structure

```
adaptive-ocr/
├── main.py                    # Main batch processing entry point (async)
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── ocr_results.json         # Generated results file
├── images/                  # Input images folder
│   ├── bar.jpg
│   ├── bus-station.jpg
│   └── shell.jpg
├── models/
│   └── ocr_models.py        # Pydantic data models
└── services/
    ├── ocr_processor.py     # Main processing orchestration
    ├── rekognition_service.py # AWS Rekognition integration
    └── bedrock_service.py   # AWS Bedrock integration
```

## Error Handling

The system gracefully handles:
- Invalid or corrupted image files
- AWS API rate limits and errors
- Network connectivity issues
- Missing AWS credentials
- Unsupported image formats

Errors are logged to console and individual image failures don't stop batch processing.



### Common Issues

1. **No AWS credentials**: Set up AWS environment variables or credentials
2. **Bedrock access denied**: Ensure Mistral Pixtral model access in us-west-2
3. **No images found**: Check `images/` folder and file formats
4. **Large images**: Resize images if > 10MB