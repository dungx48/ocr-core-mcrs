from fastapi import APIRouter, HTTPException, status

from src.service.text_detection_service import TextDetectionService
from src.dto.request.text_detect_request import TextDetectRequest
text_detection_router = APIRouter(
    prefix="/text-detect",
    tags=["TextDetection"]
)
text_detection_service = TextDetectionService()

@text_detection_router.post("/", response_model=dict, summary="Detect text from image")
def detect_text(request: TextDetectRequest):
    """
    POST /text-detect/
    Body JSON: { "image_path": "đường_dẫn_đến_ảnh.jpg" }
    """
    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))