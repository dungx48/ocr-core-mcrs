import ast
import json
import logging
from time import sleep
from typing import List
from decouple import config
from threading import Thread
from confluent_kafka import Consumer, KafkaError

from src.service.text_recognition_service import TextRecognitionService
from src.service.ocr_result_producer import OcrResultProducer

class ConsumeRequestOcrController(Thread):
    consumer: Consumer = None
    application_enable = True
    def __init__(self):
        super(ConsumeRequestOcrController, self).__init__()

        conf = {
            'bootstrap.servers': config("KAFKA_BOOTSTRAP_SERVERS"),
            'group.id': config("KAFKA_GROUP_ID"),
            'auto.offset.reset': 'smallest',
            'partition.assignment.strategy': 'roundrobin'
        }
        self.consumer = Consumer(conf)
        self.topics = config("KAFKA_OCR_REQUEST_TOPIC").split(",")

        self.text_recog_service = TextRecognitionService()
        self.result_producer = OcrResultProducer()

    def run(self):
        try:
            self.consumer.subscribe(self.topics)
            while self.application_enable:
                msg = self.consumer.poll(timeout=1.0)
                if self.check_message(msg):
                    try:
                        json_message = self.get_json_message(msg)
                        # Xử lý logic OCR
                        result = self.text_recog_service.text_recognition_by_vietocr(json_message['image_path'])
                        logging.info(result)
                        # Produce msg lên Kafka
                        self.result_producer.send(json_message, result)
                    except Exception as e:
                        logging.error("[Controller] Error : {e}".format(e=e.__str__()))
        finally:
            self.consumer.close()

    def stop_application(self):
        self.application_enable = False

    def check_message(self, msg):
        if msg is None:
            return False
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition event
                logging.error('%% %s [%d] reached end at offset %d\n' %
                              (msg.topic(), msg.partition(), msg.offset()))
                return False
            else:
                logging.error(msg.error())
                sleep(5)
                return False
        return True
    
    def get_json_message(self, msg):
        try:
            json_message = ast.literal_eval(msg.value().decode('utf8'))
        except ValueError:
            json_message = json.loads(msg.value().decode('utf8'))
        except Exception as e:
            raise Exception(f"message {str(e)} error")
        if not isinstance(json_message, dict):
            raise Exception(f'message -{str(json_message)}-is not JSON ')
        return json_message