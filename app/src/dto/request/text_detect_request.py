from pydantic import BaseModel

class TextDetectRequest(BaseModel):
    image_path: str