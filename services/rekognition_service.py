import boto3
from typing import List
from botocore.exceptions import ClientError, BotoCoreError
from models.ocr_models import DetectedText
import config


class RekognitionService:
    """Service for AWS Rekognition text detection operations"""
    
    def __init__(self):
        """Initialize Rekognition client with configured credentials"""
        try:
            self.client = boto3.client(
                'rekognition',
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                aws_session_token=config.AWS_SESSION_TOKEN
            )
        except Exception:
            raise
    
    def detect_text(self, image_bytes: bytes) -> List[DetectedText]:
        """
        Detect text in image using AWS Rekognition DetectText API.
        
        Args:
            image_bytes: Raw image data as bytes
            
        Returns:
            List of DetectedText objects with text and confidence scores
            
        Raises:
            Exception: If Rekognition API call fails
        """
        try:
            
            response = self.client.detect_text(
                Image={'Bytes': image_bytes}
            )
            
            detected_texts = []
            
            # Process text detections - only include WORD level detections to avoid duplicates
            for detection in response.get('TextDetections', []):
                if detection['Type'] == 'WORD':  # Skip LINE detections to avoid duplicates
                    detected_text = DetectedText(
                        text=detection['DetectedText'],
                        confidence=detection['Confidence'] / 100.0  # Convert to 0-1 scale
                    )
                    detected_texts.append(detected_text)
            
            return detected_texts
            
        except ClientError as e:
            error_message = e.response['Error']['Message']
            raise Exception(f"Rekognition API error: {error_message}")
            
        except BotoCoreError as e:
            raise Exception(f"AWS connection error: {str(e)}")
            
        except Exception as e:
            raise Exception(f"Text detection failed: {str(e)}")
    
    
    
