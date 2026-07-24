# from ultralytics import YOLO
# from pathlib import Path
# import numpy as np
# import time

# MODEL = Path("model") / "round_3" / "best.onnx"
# model = YOLO(str(MODEL))

# # warm-up ครั้งเดียวตอนโหลดโมเดล
# _dummy = np.zeros((640, 640, 3), dtype=np.uint8) 
# model(_dummy, verbose=False)
# # results = model(image, imgsz=320, verbose=False)  # จาก 640 → 320 ไว้ปรับความละเอียดเพื่อให้มันเร็ว
# from ultralytics import YOLO
# from pathlib import Path
# import numpy as np
# import time

# MODEL = Path("model") / "round_3" / "best.onnx"
# model = YOLO(str(MODEL), task="detect")

# # warm-up ครั้งเดียวตอนโหลดโมเดล
# _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
# model(_dummy, verbose=False)


# def detect(image, conf=0.70, imgsz=640, verbose=False):

#     # t0 = time.time()

#     results = model(
#         image,
#         conf=conf,
#         imgsz=imgsz,
#         verbose=verbose
#     )

#     # t1 = time.time()
#     # print(f"[detect.py] YOLO inference: {(t1 - t0) * 1000:.2f} ms")

#     cars = []
#     plates = []

#     for result in results:
#         for box in result.boxes:

#             cls = int(box.cls[0])
#             class_name = model.names[cls]
#             score = float(box.conf[0])

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             crop = image[y1:y2, x1:x2]

#             if class_name == "car":
#                 cars.append({"image": crop, "bbox": (x1, y1, x2, y2), "conf": score})
#             elif class_name == "license-plate":
#                 plates.append({"image": crop, "bbox": (x1, y1, x2, y2), "conf": score})

#     # t2 = time.time()
#     # print(f"[detect.py] Parse results (แยก cars/plates): {(t2 - t1) * 1000:.2f} ms")
#     # print(f"[detect.py] รวมทั้ง detect(): {(t2 - t0) * 1000:.2f} ms")

#     return cars, plates

from ultralytics import YOLO
from pathlib import Path
import numpy as np

MODEL = Path("model") / "round_3" / "best.onnx"

model = YOLO(str(MODEL), task="detect")

# Warm-up
_dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model(_dummy, verbose=False)

# -----------------------------
# Buffer
# -----------------------------
car_buffer = {}      # เก็บภาพรถที่ดีที่สุด
plate_buffer = {}    # เก็บภาพป้าย 5 ภาพ

MAX_PLATE_IMAGES = 5


def detect(image, conf=0.70, imgsz=640):

    results = model.track(
        image,
        persist=True,
        tracker="bytetrack.yaml",
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )

    cars = []
    plates = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls = int(box.cls[0])
            class_name = model.names[cls]

            score = float(box.conf[0])

            track_id = -1
            if box.id is not None:
                track_id = int(box.id[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = image[y1:y2, x1:x2]

            # -----------------------
            # CAR
            # -----------------------
            if class_name == "car":

                cars.append({
                    "track_id": track_id,
                    "image": crop,
                    "bbox": (x1, y1, x2, y2),
                    "conf": score
                })

                # เก็บเฉพาะภาพที่ confidence สูงสุด
                if track_id not in car_buffer:

                    car_buffer[track_id] = {
                        "image": crop,
                        "bbox": (x1, y1, x2, y2),
                        "conf": score
                    }

                elif score > car_buffer[track_id]["conf"]:

                    car_buffer[track_id] = {
                        "image": crop,
                        "bbox": (x1, y1, x2, y2),
                        "conf": score
                    }

            # -----------------------
            # LICENSE PLATE
            # -----------------------
            elif class_name == "license-plate":

                plates.append({
                    "track_id": track_id,
                    "image": crop,
                    "bbox": (x1, y1, x2, y2)
                })

                if track_id not in plate_buffer:
                    plate_buffer[track_id] = []

                plate_buffer[track_id].append(crop)

                # เก็บแค่ 5 ภาพล่าสุด
                if len(plate_buffer[track_id]) > MAX_PLATE_IMAGES:
                    plate_buffer[track_id].pop(0)

    return cars, plates