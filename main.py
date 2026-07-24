# import cv2
# from detect import detect
# from ocr import read_plate_sequence
# import time
# # t_start = time.time()
# image = cv2.imread("t.jpg")  # อ่านภาพ
# cars, plates = detect(image)  # ส่งภาพไป detection
# # cars, plates เป็น list ของ dict: {"image": crop, "bbox": (x1,y1,x2,y2), "conf": score}

# print("Cars:", len(cars))
# print("Plates:", len(plates))

# # ทำสำเนาภาพไว้วาดกรอบ (กันแก้ทับภาพต้นฉบับ)
# image_draw = image.copy()

# # แสดงพิกัดกรอบรถ
# for i, car in enumerate(cars):
#     print(f"Car {i}: bbox={car['bbox']} conf={car['conf']:.2f}")

# # อ่าน OCR จากป้ายทะเบียนที่ crop มาแล้ว (ใช้ plate["image"] ที่ detect() crop ให้แล้ว)
# for i, plate in enumerate(plates):
#     plate_crop = plate["image"]  # ใช้ crop ที่ detect() ทำไว้แล้ว ไม่ต้อง crop ซ้ำ

#     if plate_crop.size == 0:
#         print(f"Plate {i}: crop ว่างเปล่า ข้ามไป")
#         continue

#     cv2.imwrite(f"plate_{i}.jpg", plate_crop)

#     license_id, province = read_plate_sequence(plate_crop, split_ratio= 0.60,
#                 province_threshold= 70,
#                 conf_threshold= 0.6,)
#     print(f"Plate {i} -> {license_id}, {province}")

# # t3 = time.time()
# # print(f"[main.py] เวลารวมทั้งโปรแกรม (ไม่รวมการรอ imshow): {(t3 - t_start) * 1000:.2f} ms")

import cv2
import time
from detect import detect, car_buffer, plate_buffer
from ocr import read_plate_sequence

# =====================================================================
# จำลองว่ามีวิดีโอเข้ามา โดยใช้ภาพนิ่งภาพเดียว "ส่งซ้ำ" หลายรอบแทนหลายเฟรม
# เพื่อทดสอบว่าระบบ track_id + buffer (ใน detect.py) และการ vote หลายเฟรม
# (ใน ocr.py) ทำงานร่วมกันถูกต้องไหม ก่อนเอาไปต่อกับวิดีโอ/กล้องจริง
# =====================================================================

IMAGE_PATH = "t.jpg"
SIMULATED_FRAMES = 10  # จำลองว่ามี 10 เฟรมจากวิดีโอ

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {IMAGE_PATH}")

t_start = time.time()

print(f"[main.py] จำลอง {SIMULATED_FRAMES} เฟรมจากวิดีโอ (ใช้ภาพ '{IMAGE_PATH}' ซ้ำทุกเฟรม)")
print("[main.py] เป้าหมาย: เช็คว่า track_id คงที่ข้ามเฟรม และ plate_buffer สะสมภาพได้ถูกต้อง\n")

for frame_idx in range(SIMULATED_FRAMES):
    cars, plates = detect(image)
    print(f"  เฟรม {frame_idx}: cars={len(cars)} plates={len(plates)}")

t_detect_done = time.time()
print(f"\n[main.py] จบการจำลองเฟรมแล้ว ({(t_detect_done - t_start) * 1000:.1f} ms รวม {SIMULATED_FRAMES} เฟรม)")

# -----------------------------------------------------------------------
# หลังจากรับหลายเฟรมแล้ว car_buffer/plate_buffer (module-level dict ใน
# detect.py) จะสะสมข้อมูลไว้ตาม track_id:
#   - car_buffer[track_id]   = ภาพรถที่ confidence สูงสุด (ภาพเดียว)
#   - plate_buffer[track_id] = ภาพป้ายสูงสุด 5 ภาพล่าสุด (list)
# ตรงนี้คือจุดที่ main.py ควรเรียก OCR แบบ "vote หลายเฟรม" ไม่ใช่เรียก
# จากภาพเดี่ยวที่ได้จาก detect() ในแต่ละเฟรมตรงๆ
# -----------------------------------------------------------------------

print(f"[main.py] พบรถทั้งหมด {len(car_buffer)} คัน (track_id: {list(car_buffer.keys())})")
print(f"[main.py] พบป้ายทะเบียนทั้งหมด {len(plate_buffer)} ป้าย (track_id: {list(plate_buffer.keys())})\n")

for track_id, plate_images in plate_buffer.items():

    if not plate_images:
        print(f"Track {track_id}: buffer ว่างเปล่า ข้ามไป")
        continue

    # เซฟภาพตัวอย่างไว้ดู debug (เฉพาะภาพแรกใน buffer พอ)
    cv2.imwrite(f"plate_track{track_id}.jpg", plate_images[0])

    t0 = time.time()
    license_id, province = read_plate_sequence(
        plate_images,               # ส่ง "list ของภาพ" ไม่ใช่ภาพเดี่ยว
        split_ratio=0.60,
        province_threshold=70,
        conf_threshold=0.6,
    )
    t1 = time.time()

    print(
        f"Track {track_id} ({len(plate_images)} เฟรมสะสม, "
        f"vote OCR ใช้ {(t1 - t0) * 1000:.1f} ms) "
        f"-> license={license_id!r}, province={province!r}"
    )

t_end = time.time()
print(f"\n[main.py] เวลารวมทั้งโปรแกรม: {(t_end - t_start) * 1000:.1f} ms")