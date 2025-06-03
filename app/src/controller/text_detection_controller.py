from src.service.text_detection_service import TextDetectionService



class TextDetectionController():
    def __init__(self):
        self.text_detection_service = TextDetectionService()
    