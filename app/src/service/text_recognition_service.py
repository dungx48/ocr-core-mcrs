import os
import logging
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

from src.dto.response.text_detect_response import TextDetectResponse, Polygon
from src.service.text_detection_service import TextDetectionService

class TextRecognitionService():
    def __init__(self, model_name: str = 'vgg_seq2seq', device: str = 'cuda'):
        self.cfg = Cfg.load_config_from_name(model_name)
        self.cfg['device'] = device
        self.predictor = Predictor(self.cfg)
        self.detector = TextDetectionService()

    def recognize(self, image_np: np.ndarray, points: List[List[int]]) -> str:
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        crop = image_np[y_min:y_max, x_min:x_max]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return self.predictor.predict(pil_img)

    def text_recognition_by_vietocr(self, image_path: str) -> TextDetectResponse:
        logging.info("Starting OCR process")
        # Detect text regions
        detect_resp = self.detector.text_detection_by_craft(image_path)
        logging.info("Processing OCR Recognition")
        # Load image for recognition
        img = cv2.imread(image_path)

        # Recognize each region
        enriched = []
        for poly in detect_resp.polys:
            text = self.recognize(img, poly.points)
            enriched.append(Polygon(
                points=poly.points,
                confident_score=poly.confident_score,
                text=text
            ))
        
        # canvas = self.annotate_full_text_on_blank_canvas(
        #     img,
        #     enriched, 
        #     output_path='fulltext_' + os.path.basename(image_path),
        #     font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        #     font_size=10
        # )

        return TextDetectResponse(
            image_name=detect_resp.image_name,
            polys=enriched
        )

    # def annotate_full_text_on_blank_canvas(self,
    #                                        image_np: np.ndarray,
    #                                        polys: List[Polygon],
    #                                        output_path: Optional[str] = None,
    #                                        font_path: str = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    #                                        font_size: Optional[int] = None
    #                                       ) -> np.ndarray:
    #     """
    #     In full text (Unicode tiếng Việt) của từng bbox lên canvas trắng bên cạnh,
    #     theo đúng thứ tự và vị trí y (chuẩn dòng), không cắt chữ.
    #     """
    #     h, w = image_np.shape[:2]
    #     # 1) tạo canvas trắng cùng chiều cao, rộng đủ tối thiểu 2 lần ảnh (đỡ bị cắt)
    #     canvas_w = w * 2
    #     canvas = np.ones((h, canvas_w, 3), dtype=np.uint8) * 255
    #     # 2) đặt ảnh gốc bên trái
    #     canvas[:, :w] = image_np

    #     # 3) chuyển sang PIL để vẽ Unicode
    #     pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    #     draw = ImageDraw.Draw(pil)
    #     size = font_size or max(12, h // 60)
    #     try:
    #         font = ImageFont.truetype(font_path, size=size)
    #     except IOError:
    #         raise IOError(f"Không tìm thấy font tại {font_path}.")

    #     for poly in polys:
    #         text = poly.text or ""
    #         # tính x, y bắt đầu của dòng dựa trên bbox
    #         xs = [pt[0] for pt in poly.points]
    #         ys = [pt[1] for pt in poly.points]
    #         x0, y0 = min(xs), min(ys)

    #         # in text full (không cắt) tại (x0 + w, y0)
    #         draw.text((x0 + w, y0), text, font=font, fill=(0,0,0))

    #     # 4) convert ngược lại OpenCV BGR
    #     canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    #     if output_path:
    #         cv2.imwrite(output_path, canvas)
    #     return canvas