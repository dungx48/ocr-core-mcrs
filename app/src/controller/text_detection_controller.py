from fastapi import APIRouter, HTTPException, status, Request

from src.service.text_detection_service import TextDetectionService
from src.dto.request.text_detect_request import TextDetectRequest
from src.common.base.structural.response_object import ResponseObject

text_detection_router = APIRouter(
    prefix="/text-detect",
    tags=["TextDetection"]
)
text_detection_service = TextDetectionService()

@text_detection_router.post("/", response_model=dict, summary="Detect text from image")
def detect_text(request: Request, input: TextDetectRequest) -> ResponseObject:
    """
    POST /text-detect/
    Body JSON: { "image_path": "đường_dẫn_đến_ảnh.jpg" }
    """
    try:
        data_result = text_detection_service.text_detection_by_craft(input.image_path)
        return ResponseObject().success(data_result=data_result, path=request.url.__str__())
    except Exception as e:
        return ResponseObject().error(error=str(e), path=request.url.__str__())