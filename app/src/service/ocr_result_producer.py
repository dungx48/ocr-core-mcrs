import json
import logging
from fastapi import status
from decouple import config
from datetime import datetime
from confluent_kafka import Producer

class OcrResultProducer:
    def __init__(self):
        conf = {'bootstrap.servers': config("KAFKA_BOOTSTRAP_SERVERS")}
        self.producer = Producer(conf)
        self.topic = config("KAFKA_OCR_RESPONSE_TOPIC")
        self.complete_time = datetime.now().isoformat()

    def send(self, json_message, result):
        """
        Nhận message gốc (dict), xử lý OCR, gửi kết quả lên Kafka.
        """
        try:
            # Build response (bạn có thể tuỳ chỉnh thêm các field nếu muốn)
            response = {
                "msg_id": json_message['msg_id'],
                "image_path": json_message['image_path'],
                "ocr_result": result.dict(),
                "status": status.HTTP_200_OK,
                "created_at": json_message["created_at"],
                "complete_at": self.complete_time,
                "error_message": None
            }
            # Gửi lên topic
            self.producer.produce(self.topic, json.dumps(response).encode('utf-8'))
            self.producer.flush()
            logging.info(f"Đã gửi kết quả OCR: {response}")
        except Exception as e:
            # Gửi lỗi (nếu muốn)
            error_response = {
                "msg_id": json_message['msg_id'],
                "image_path": json_message.get('image_path'),
                "ocr_result": None,
                "status": status.HTTP_400_BAD_REQUEST,
                "created_at": json_message["created_at"],
                "complete_at": self.complete_time,
                "error_message": str(e)
            }
            self.producer.produce(self.topic, json.dumps(error_response).encode('utf-8'))
            self.producer.flush()
            logging.error(f"Lỗi khi xử lý OCR: {str(e)}")