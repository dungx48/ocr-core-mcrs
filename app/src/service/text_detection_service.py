import os
import time

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict

from src.utils import imgproc, file_utils
from src.model.detect_text.craft import CRAFT
from src.model.detect_text.craft_test_net import test_net
from src.dto.response.text_detect_response import TextDetectResponse, Polygon

class TextDetectionService():
    def __init__(self, trained_model=None, result_folder='./result/detection/'):
        self.cuda = torch.cuda.is_available()
        self.refine_net = None
        self.text_threshold = 0.7
        self.link_threshold = 0.4
        self.low_text = 0.4
        self.poly = False
        # cuda
        if trained_model is None:
            self.trained_model = 'src/weights/craft_mlt_25k.pth'
        else:
            self.trained_model = trained_model
        # result_folder
        self.result_folder = result_folder
        if not os.path.isdir(self.result_folder):
            os.makedirs(self.result_folder, exist_ok=True)

    
    def compute_confidence_from_heatmap(self, score_text: np.ndarray, poly: np.ndarray) -> float:
        """
        Tính điểm tin cậy (confidence) trung bình bên trong polygon.
        - score_text: heatmap do CRAFT trả về, có thể là single‐channel (H, W) hoặc 3‐channel (H, W, 3).
        - poly: numpy array shape (4, 2) với 4 cặp (x, y).
        """
        # Nếu score_text có 3 kênh, chỉ lấy kênh đầu
        if score_text.ndim == 3:
            heatmap = score_text[:, :, 0]
        else:
            heatmap = score_text

        # Chuyển polygon sang dạng int32 (nếu chưa phải)
        pts = poly.reshape((-1, 2)).astype(int)
        h, w = heatmap.shape[:2]
        # Tạo mask cùng kích thước heatmap, giá trị 1 trong polygon, 0 ngoài
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        # Lấy giá trị heatmap chỉ ở vùng mask ==1
        values = heatmap[mask == 1]
        if values.size == 0:
            return 0.0

        # Nếu heatmap ở dạng uint8 ([0,255]), chuyển sang float [0,1]
        if values.dtype == np.uint8:
            values = values.astype(np.float32) / 255.0

        return float(np.mean(values))
    
    def text_detection_by_craft(self, image_path:str) -> TextDetectResponse:
        # 1. Khởi tạo model CRAFT
        net = CRAFT()
        print('Loading weights from checkpoint (' + self.trained_model + ')')
        if self.cuda:
            checkpoint = torch.load(self.trained_model)
            net.load_state_dict(self.copyStateDict(checkpoint))
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False
        else:
            net.load_state_dict(self.copyStateDict(torch.load(self.trained_model, map_location='cpu')))

        net.eval()
        
        # 2. Đọc ảnh đầu vào (BGR) và inference
        image = imgproc.loadImage(image_path)
        bboxes_raw, polys, score_text = test_net(
            net, image,
            self.text_threshold,
            self.link_threshold,
            self.low_text,
            self.cuda,
            self.poly,
            self.refine_net
        )

        # 3. Lưu heatmap (score_text) ra file để debug / visualize
        filename, _ = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(self.result_folder, f"res_{filename}_mask.jpg")
        cv2.imwrite(mask_file, score_text)

        # 4. Lưu kết quả polygon (vẽ lên ảnh + xuất .txt) vào thư mục result
        #    file_utils.saveResult sẽ tạo:
        #      - res_{filename}.jpg (vẽ polygon / bbox)
        #      - res_{filename}.txt (mỗi dòng “x1,y1,x2,y2,x3,y3,x4,y4”)
        file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=self.result_folder)

        # 5. Tạo danh sách các Polygon (theo model Pydantic) để trả về
        polys_list: list[Polygon] = []
        for poly in polys:
            # poly: numpy.ndarray shape (4, 2), dtype=int32 hoặc int
            pts = poly.reshape((-1, 2)).astype(int)  # đảm bảo int

            # Chuyển thành List[List[int]] kiểu [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            points_list: list[list[int]] = pts.tolist()

            # Tính confident_score bằng hàm tự viết
            confident_score = self.compute_confidence_from_heatmap(score_text, poly)

            # Tạo instance Polygon và thêm vào danh sách
            polygon_obj = Polygon(points=points_list, confident_score=confident_score)
            polys_list.append(polygon_obj)

        # 6. Trả về Pydantic model TextDetectResponse
        response = TextDetectResponse(
            image_name=filename,
            polys=polys_list
        )
        return response
    
    def copyStateDict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict