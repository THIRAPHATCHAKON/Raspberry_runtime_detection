import re
import cv2
import numpy as np
from rapidfuzz import process, fuzz
from rapidocr_onnxruntime import RapidOCR
from huggingface_hub import hf_hub_download
import time

det_path = hf_hub_download("monkt/paddleocr-onnx", "detection/v5/det.onnx")
rec_path = hf_hub_download("monkt/paddleocr-onnx", "languages/thai/rec.onnx")
dict_path = hf_hub_download("monkt/paddleocr-onnx", "languages/thai/dict.txt")

ocr = RapidOCR(
    det_model_path=det_path,
    rec_model_path=rec_path,
    rec_keys_path=dict_path
)
_dummy = np.zeros((48, 320, 3), dtype=np.uint8)
ocr(_dummy)  # warm-up

PATTERNS = [
    {
        "name": "number_thai2_number4",
        "length": 7,
        "thai_index": [1,2],
        "number_index": [0,3,4,5,6]
    },

    {
        "name": "thai2_number4",
        "length": 6,
        "thai_index": [0,1],
        "number_index": [2,3,4,5]
    },

    {
        "name": "thai1_number4",
        "length": 5,
        "thai_index": [0],
        "number_index": [1,2,3,4]
    }
]

NUMBER_MAP = {
    "O": "0",
    "o": "0",
    "Q": "0",

    "I": "1",
    "l": "1",
    "|": "1",

    "Z": "2",

    "S": "5",

    "B": "8"
}

THAI_MAP = {
    "8": "ช",
    "6": "บ",
    "@": "อ"
}

THAI_PROVINCES = [ # ชุดข้อมูลจังหวัด
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท",
    "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง",
    "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม",
    "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส",
    "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
    "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา",
    "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
    "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
    "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
    "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี",
    "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
    "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
    "อุทัยธานี", "อุบลราชธานี", "เบตง",
]

def crop_border(image, margin=0.06): # ตัดขอบภาพทะเบียน

    h, w = image.shape[:2]

    # จำนวน pixel ที่ตัด
    mx = int(w * margin)
    my = int(h * margin)


    cropped = image[
        my:h-my,
        mx:w-mx
    ]

    return cropped

def character_constraint(license_id): # กฎป้ายทะเบียน

    chars = list(license_id)

    length = len(chars)


    for pattern in PATTERNS:

        if length == pattern["length"]:

            for i in pattern["thai_index"]:

                if chars[i] in THAI_MAP:
                    chars[i] = THAI_MAP[chars[i]]


            for i in pattern["number_index"]:

                if chars[i] in NUMBER_MAP:
                    chars[i] = NUMBER_MAP[chars[i]]


            return "".join(chars)


    return license_id

def fuzz_login(province): # ทำนายจังหวัด
    province = re.sub(r"[^\u0E00-\u0E7F]", "", province)
    province = province.strip()
    
    match = process.extractOne(
            province,
            THAI_PROVINCES,
            scorer=fuzz.ratio
        )

    if match is None:
        return province

    province, score, _ = match

    if score >= 90:
        return province
    
    return province
    
def split(img): # แบ่งรูปภาพ
    h,w = img.shape[:2]
    split_proportion = int(h * 0.65)
    top = img[:split_proportion+5, :]
    bottom = img[split_proportion-5:, :]
    return top , bottom


def order_points(pts):
    """
    เรียงมุมเป็น
    TL, TR, BR, BL
    """

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]   # Top Left
    rect[2] = pts[np.argmax(s)]   # Bottom Right

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]    # Top Right
    rect[3] = pts[np.argmax(diff)]    # Bottom Left

    return rect


def perspective_plate(plate): # หมุนภาพ

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    # ลด Noise
    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    # Threshold
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ถ้าป้ายเป็นตัวดำพื้นขาว
    # ลองเปิดบรรทัดนี้แทน
    # thresh = cv2.bitwise_not(thresh)

    # หา Contour
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("No contour")
        return plate

    # เลือก Contour ใหญ่สุด
    contour = max(
        contours,
        key=cv2.contourArea
    )

    # หา Min Area Rectangle
    rect = cv2.minAreaRect(contour)

    # 4 มุม
    box = cv2.boxPoints(rect)

    box = np.float32(box)

    # เรียงมุม
    box = order_points(box)

    # คำนวณความกว้าง
    widthA = np.linalg.norm(
        box[2] - box[3]
    )

    widthB = np.linalg.norm(
        box[1] - box[0]
    )

    maxWidth = int(max(widthA, widthB))

    # คำนวณความสูง
    heightA = np.linalg.norm(
        box[1] - box[2]
    )

    heightB = np.linalg.norm(
        box[0] - box[3]
    )

    maxHeight = int(max(heightA, heightB))

    # จุดปลายทาง
    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]
    ], dtype="float32")

    # Homography Matrix
    M = cv2.getPerspectiveTransform(
        box,
        dst
    )

    # Perspective Transform
    warp = cv2.warpPerspective(
        plate,
        M,
        (maxWidth, maxHeight)
    )

    return warp

def read_plate(folder_image):

    folder_image = perspective_plate(folder_image)
    folder_image = crop_border(folder_image, margin=0.07)
    folder_image = cv2.resize(folder_image, (320, 48))
    top, bottom = split(folder_image)

    t0 = time.time()
    top_result, _ = ocr(top, use_det=False, use_cls=False)
    t1 = time.time()
    bottom_result, _ = ocr(bottom, use_det=False, use_cls=False)
    t2 = time.time()

    print(f"OCR top: {(t1 - t0)*1000:.1f} ms")
    print(f"OCR bottom: {(t2 - t1)*1000:.1f} ms")
    print(f"OCR total: {(t2 - t0)*1000:.1f} ms")

    license_id = top_result[0][0] if top_result else ""
    province_raw = bottom_result[0][0] if bottom_result else ""

    license_id = re.sub(r"[\s\-]", "", license_id)
    province = fuzz_login(province_raw)

    return license_id, province
    

