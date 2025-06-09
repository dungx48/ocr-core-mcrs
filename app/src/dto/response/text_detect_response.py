from pydantic import BaseModel
from typing import List, Optional

class Polygon(BaseModel):
    # Định nghĩa một polygon 4 điểm luôn lưu thành list 2 chiều
    points: List[List[int]] # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confident_score: float
    text: Optional[str] = None

class TextDetectResponse(BaseModel):
    image_name: str
    polys: List[Polygon]
