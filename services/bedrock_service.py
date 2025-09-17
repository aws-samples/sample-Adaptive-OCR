import boto3
import json
import re
from typing import List, Optional
from botocore.exceptions import ClientError, BotoCoreError
from models.ocr_models import DetectedText
import config

class BedrockService:
    """Service for AWS Bedrock multimodal text extraction operations"""
    
    def __init__(self):
        """Initialize Bedrock Runtime client with configured credentials"""
        try:
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                aws_session_token=config.AWS_SESSION_TOKEN
            )
        except Exception as e:
            raise
    
    def extract_text_from_image(self, image_base64: str, rekognition_context: List[DetectedText] = None, 
                               target_categories: Optional[List[str]] = None) -> List[DetectedText]:
        """
        Extract text from image using AWS Bedrock multimodal model as a second opinion.
        
        Args:
            image_base64: Base64 encoded image data
            rekognition_context: Optional context from Rekognition results for comparison
            target_categories: Optional target categories to focus extraction on
            
        Returns:
            List of DetectedText objects with extracted text and confidence scores
            
        Raises:
            Exception: If Bedrock API call fails
        """
        try:
            
            
            # Prepare the message for Mistral Pixtral
            message = {
                "role": "user",
                "content": [
                    {
                        "text": "Please examine the provided image and identify the text shown. After the text, add your confidence level as 'Confidence: X.XX' where X.XX is a decimal between 0.0 and 1.0.",
                        "type": "text"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
            
            # Prepare the request body for Mistral
            body = {
                "messages": [message],
                "max_tokens": config.BEDROCK_MAX_TOKENS
            }
            
            # Make the API call
            response = self.client.invoke_model(
                modelId=config.BEDROCK_MODEL_ID,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            # Parse the Mistral response
            response_body = json.loads(response['body'].read())
            extracted_text = response_body['choices'][0]['message']['content']
            
            # Extract confidence from the response using regex
            confidence_match = re.search(r'Confidence:\s*([0-9]*\.?[0-9]+)', extracted_text)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
                    # Remove confidence text from extracted text
                    text_content = re.sub(r'\s*Confidence:\s*[0-9]*\.?[0-9]+.*$', '', extracted_text).strip()
                except ValueError:
                    confidence = 0.85
                    text_content = extracted_text.strip()
            else:
                confidence = 0.85
                text_content = extracted_text.strip()
            
            # Create DetectedText object from the response
            detected_text = DetectedText(
                text=text_content,
                confidence=confidence
            )
            
            return [detected_text]
            
        except ClientError as e:
            error_message = e.response['Error']['Message']
            raise Exception(f"Bedrock API error: {error_message}")
            
        except BotoCoreError as e:
            raise Exception(f"AWS connection error: {str(e)}")
            
        except Exception as e:
            raise Exception(f"Bedrock text extraction failed: {str(e)}")
    
