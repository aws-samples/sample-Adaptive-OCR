import base64
from typing import List
from models.ocr_models import DetectedText, SecondOpinion, OCRResponse
from services.rekognition_service import RekognitionService
from services.bedrock_service import BedrockService
import config


class OCRProcessor:
    """Main OCR processing orchestrator that handles the complete workflow"""
    
    def __init__(self):
        """Initialize OCR processor with required services"""
        self.rekognition_service = RekognitionService()
        self.bedrock_service = BedrockService()
    
    async def process_image(self, image_bytes: bytes) -> OCRResponse:
        """
        Process an image through the complete OCR pipeline with second opinion logic.
        
        Args:
            image_bytes: Raw image data as bytes
            
        Returns:
            OCRResponse with complete processing results
        """
        
        # Step 1: Process with Rekognition (all detected text)
        rekognition_results = self.rekognition_service.detect_text(image_bytes)
        
        # Step 2: Evaluate if second opinion is needed
        need_second_opinion = (
            not rekognition_results or 
            any(result.confidence < config.CONFIDENCE_THRESHOLD for result in rekognition_results)
        )
        
        bedrock_results = []
        second_opinion_data = SecondOpinion(triggered=False)
        
        if need_second_opinion:
            
            # Step 3: Get Bedrock second opinion
            bedrock_results = self.bedrock_service.extract_text_from_image(
                base64.b64encode(image_bytes).decode('utf-8'),
                rekognition_context=rekognition_results
            )
            
            # Step 4: Create simple second opinion data
            reason = self._determine_second_opinion_reason(rekognition_results)
            second_opinion_data = SecondOpinion(
                triggered=True,
                reason=reason
            )
        
        # Step 5: Build complete response
        response = OCRResponse(
            rekognition=rekognition_results,
            bedrock=bedrock_results,
            second_opinion=second_opinion_data
        )
        
        return response
    
    
    
    def _determine_second_opinion_reason(self, rekognition_results: List[DetectedText]) -> str:
        """Determine the reason why second opinion was triggered"""
        if not rekognition_results:
            return "No text detected by Rekognition"
        
        low_conf_items = [result for result in rekognition_results if result.confidence < config.CONFIDENCE_THRESHOLD]
        if low_conf_items:
            min_confidence = min(item.confidence for item in low_conf_items)
            return f"Rekognition confidence below 0.7 (lowest: {min_confidence:.2f})"
        
        return "Second opinion triggered for verification"
    
    
    
    
