import os
from dotenv import load_dotenv

load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

# OCR Configuration
CONFIDENCE_THRESHOLD = 0.7
MAX_IMAGE_SIZE_MB = 10
SUPPORTED_IMAGE_FORMATS = {"image/jpeg", "image/png", "image/jpg"}

# Bedrock Configuration
BEDROCK_MODEL_ID = "us.mistral.pixtral-large-2502-v1:0"  # Mistral Pixtral multimodal model for OCR tasks
BEDROCK_MAX_TOKENS = 1000
BEDROCK_TEMPERATURE = 0.1

