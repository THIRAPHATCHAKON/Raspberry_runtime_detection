from ultralytics import YOLO
from pathlib import Path
import numpy as np
import time

MODEL = Path("model") / "round_3" / "best.onnx"
model = YOLO(str(MODEL))

# warm-up ครั้งเดียวตอนโหลดโมเดล
_dummy = np.zeros((640, 640, 3), dtype=np.uint8) 
model(_dummy, verbose=False)
# results = model(image, imgsz=320, verbose=False)  # จาก 640 → 320 ไว้ปรับความละเอียดเพื่อให้มันเร็ว
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import time

MODEL = Path("model") / "round_3" / "best.onnx"
model = YOLO(str(MODEL), task="detect")

# warm-up ครั้งเดียวตอนโหลดโมเดล
_dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model(_dummy, verbose=False)


def detect(image, conf=0.70, imgsz=640, verbose=False):

    t0 = time.time()

    results = model(
        image,
        conf=conf,
        imgsz=imgsz,
        verbose=verbose
    )

    t1 = time.time()
    print(f"[detect.py] YOLO inference: {(t1 - t0) * 1000:.2f} ms")

    cars = []
    plates = []

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])
            class_name = model.names[cls]
            score = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]

            if class_name == "car":
                cars.append({"image": crop, "bbox": (x1, y1, x2, y2), "conf": score})
            elif class_name == "license-plate":
                plates.append({"image": crop, "bbox": (x1, y1, x2, y2), "conf": score})

    t2 = time.time()
    print(f"[detect.py] Parse results (แยก cars/plates): {(t2 - t1) * 1000:.2f} ms")
    print(f"[detect.py] รวมทั้ง detect(): {(t2 - t0) * 1000:.2f} ms")

    return cars, plates