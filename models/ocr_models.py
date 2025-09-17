from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DetectedText(BaseModel):
    """Represents a single detected text item"""
    text: str = Field(..., description="The detected text content")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")

class SecondOpinion(BaseModel):
    """Details about the second opinion from Bedrock"""
    triggered: bool = Field(..., description="Whether second opinion was triggered")
    reason: Optional[str] = Field(None, description="Reason for triggering second opinion")

class OCRResponse(BaseModel):
    """Complete OCR processing response"""
    rekognition: List[DetectedText] = Field(default_factory=list, description="Rekognition results")
    bedrock: List[DetectedText] = Field(default_factory=list, description="Bedrock results (if triggered)")
    second_opinion: SecondOpinion = Field(..., description="Second opinion details")