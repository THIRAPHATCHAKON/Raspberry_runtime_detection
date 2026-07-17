import cv2
from detect import detect
from ocr import read_plate
import time
t_start = time.time()
image = cv2.imread("r.jpg")  # อ่านภาพ
cars, plates = detect(image)  # ส่งภาพไป detection
# cars, plates เป็น list ของ dict: {"image": crop, "bbox": (x1,y1,x2,y2), "conf": score}

print("Cars:", len(cars))
print("Plates:", len(plates))

# ทำสำเนาภาพไว้วาดกรอบ (กันแก้ทับภาพต้นฉบับ)
image_draw = image.copy()

# แสดงพิกัดกรอบรถ
for i, car in enumerate(cars):
    print(f"Car {i}: bbox={car['bbox']} conf={car['conf']:.2f}")

# อ่าน OCR จากป้ายทะเบียนที่ crop มาแล้ว (ใช้ plate["image"] ที่ detect() crop ให้แล้ว)
for i, plate in enumerate(plates):
    plate_crop = plate["image"]  # ใช้ crop ที่ detect() ทำไว้แล้ว ไม่ต้อง crop ซ้ำ

    if plate_crop.size == 0:
        print(f"Plate {i}: crop ว่างเปล่า ข้ามไป")
        continue

    cv2.imwrite(f"plate_{i}.jpg", plate_crop)

    license_id, province = read_plate(plate_crop)
    print(f"Plate {i} -> {license_id}, {province}")

t3 = time.time()
print(f"[main.py] เวลารวมทั้งโปรแกรม (ไม่รวมการรอ imshow): {(t3 - t_start) * 1000:.2f} ms")