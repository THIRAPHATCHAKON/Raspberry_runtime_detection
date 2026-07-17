from ultralytics import YOLO
from pathlib import Path

MODEL = Path("model") / "round_3" / "best.onnx"
model = YOLO(str(MODEL))

def detect(image, conf=0.70):

    results = model(
        image,
        conf=conf
    )

    cars = []
    plates = []

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            class_name = model.names[cls]   # <-- ใช้ชื่อ class แทนเลข id
            score = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            crop = image[y1:y2, x1:x2]

            if class_name == "car":
                cars.append({
                    "image": crop,
                    "bbox": (x1, y1, x2, y2),
                    "conf": score
                })

            elif class_name == "license-plate":
                plates.append({
                    "image": crop,
                    "bbox": (x1, y1, x2, y2),
                    "conf": score
                })

    return cars, plates