from fastapi import APIRouter, HTTPException, status, Request

from src.service.text_recognition_service import TextRecognitionService
from src.dto.request.text_detect_request import TextDetectRequest
from src.common.base.structural.response_object import ResponseObject

text_recognition_router = APIRouter(
    prefix="/text-recognition",
    tags=["TextRecognition"]
)
text_recognition_service = TextRecognitionService()

@text_recognition_router.post("/", response_model=dict, summary="OCR from image")
def recognize_text(request: Request, input: TextDetectRequest) -> ResponseObject:
    """
    POST /text-recognition/
    """
    try:
        data_result = text_recognition_service.text_recognition_by_vietocr(input.image_path)
        return ResponseObject().success(data_result=data_result, path=request.url.__str__())
    except Exception as e:
        return ResponseObject().error(error=str(e), path=request.url.__str__())