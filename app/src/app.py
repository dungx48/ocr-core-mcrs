import logging

from concurrent.futures.thread import ThreadPoolExecutor

from starlette.middleware.cors import CORSMiddleware
from starlette_prometheus import PrometheusMiddleware

from app.src.common.app import app
from app.src.controller.text_detection_controller import text_detection_router
from app.src.controller.text_recognition_cotroller import text_recognition_router
from app.src.controller.consume_request_ocr_controller import ConsumeRequestOcrController
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PrometheusMiddleware)

app.include_router(text_detection_router)
app.include_router(text_recognition_router)

ocr_process_consumer = ConsumeRequestOcrController()

@app.on_event("startup")
def startup_event():
    logging.info("[APP-WORKER] on startup event....")
    pool = ThreadPoolExecutor(max_workers=2)
    ocr_process_consumer.start()


@app.on_event("shutdown")
def shutdown_event():
    logging.warning("[APP] running on shutdown event....")
    ocr_process_consumer.stop_application()
