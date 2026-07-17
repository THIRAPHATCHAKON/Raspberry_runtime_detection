# import cv2
# from detect import detect
# from ocr import read_plate

# image = cv2.imread("r.jpg") # อ่านภาพ
# cars, plates = detect(image) # ส่งภาพไป detection
# print("Cars:", len(cars))
# print("Plates:", len(plates))


# while True:
#     cv2.imshow("test", cars)
    
#     cv2.waitKey(0)

# for i, car in enumerate(cars):

#     print(car)

# # อ่าน OCR
# for i, plate in enumerate(plates):
#     cv2.imwrite("h.jpg", plate)
#     license_id , province = read_plate(plate)
    
#     print(license_id)
#     print(province)
    
    
import cv2
from detect import detect
from ocr import read_plate

image = cv2.imread("r.jpg")  # อ่านภาพ
cars, plates = detect(image)  # ส่งภาพไป detection
# cars, plates เป็น list ของ dict: {"image": crop, "bbox": (x1,y1,x2,y2), "conf": score}

print("Cars:", len(cars))
print("Plates:", len(plates))

# ทำสำเนาภาพไว้วาดกรอบ (กันแก้ทับภาพต้นฉบับ)
image_draw = image.copy()

# วาดกรอบรถ (สีเขียว)
for i, car in enumerate(cars):
    x1, y1, x2, y2 = car["bbox"]
    cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_draw, f"car {i} {car['conf']:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# วาดกรอบป้ายทะเบียน (สีแดง)
for i, plate in enumerate(plates):
    x1, y1, x2, y2 = plate["bbox"]
    cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image_draw, f"plate {i} {plate['conf']:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# แสดงผลภาพที่วาดกรอบแล้ว
cv2.imshow("detections", image_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()

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


