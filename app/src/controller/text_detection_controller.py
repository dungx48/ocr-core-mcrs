from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.service.text_detection_service import TextDetectionService

router = APIRouter(
    prefix="/text-detect",
    tags=["TextDetection"]
)

class TextDetectionController(BaseModel):
    def __init__(self):
        self.text_detection_service = TextDetectionService()
    