import re
import cv2
import numpy as np
from pathlib import Path
from paddleocr import TextRecognition
from rapidfuzz import process, fuzz
from collections import Counter

ocr = TextRecognition(
    model_name="th_PP-OCRv5_mobile_rec"   # ตรงกับจุดที่เริ่ม fine-tune จริง
)

THAI_MAP = {
    "@": "ฮ",
    "&": "ฃ",
    "N": "ก",
    "n": "ก",
    "1": "ก",
    "0": "ค",
    "H": "ฬ",
    "W": "พ",
    "U": "ข",
    "A": "ฎ"
}

# Patterns
# PLATE_PATTERN = re.compile(r'^[0-9]?[ก-ฮ]{1,3}\s?[0-9]{1,4}$')

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# โฟลเดอร์เก็บภาพ debug (ภาพที่ผ่านการแปลงเป็นขาวดำ)
DEBUG_DIR = Path("debug_output")

THAI_PROVINCES = [
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
    "อุทัยธานี", "อุบลราชธานี","เบตง",
]

def read_plate_sequence(images,
                        split_ratio,
                        province_threshold,
                        conf_threshold):

    license_results = []
    province_results = []

    for img in images:

        license_id, province = read_plate(
            img,
            split_ratio,
            province_threshold,
            conf_threshold
        )

        if license_id:
            license_results.append(license_id)

        if province:
            province_results.append(province)

    final_license = character_vote(license_results)
    final_province = province_vote(province_results)

    return final_license, final_province

def read_plate(image_source, split_ratio, #ทำงาน 2
               province_threshold,
               conf_threshold,) -> dict:
    
    if isinstance(image_source, (str, Path)):
        img = cv2.imread(str(image_source))
        if img is None:
            raise FileNotFoundError(f"ไม่พบไฟล์: {image_source}")
    elif isinstance(image_source, np.ndarray):
        img = image_source
    else:
        raise TypeError(
            f"image_source ต้องเป็น str/Path (path ไฟล์) หรือ np.ndarray (ภาพที่โหลดแล้ว) "
            f"แต่ได้รับ {type(image_source)}"
        )
    stem = Path(image_source).stem if isinstance(image_source, (str, Path)) else "crop"
    img = perspective_plate(img)
    img = crop_border(
        img,
        margin=0.04
    )
    top_img, bottom_img = split_plate_image(img, split_ratio)

    # OCR พร้อม confidence threshold ตามงานวิจัยโรมาเนีย
    top_results      = ocr_image(top_img, conf_threshold)
    plate_number_raw = " ".join(r[0] for r in top_results)
    license_id = clean_plate_number(plate_number_raw)
    bottom_results       = ocr_image(bottom_img, conf_threshold)
    province_raw         = " ".join(r[0] for r in bottom_results)
    province, prov_score = match_province(province_raw, threshold=province_threshold)

    return license_id ,province

def order_points(pts): #ทำงาน 3
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
    widthA = np.linalg.norm(box[2] - box[3])

    widthB = np.linalg.norm(box[1] - box[0])

    maxWidth = int(max(widthA, widthB))

    # คำนวณความสูง
    heightA = np.linalg.norm( box[1] - box[2] )

    heightB = np.linalg.norm( box[0] - box[3] )

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

def crop_border(image, margin): # ตัดขอบภาพทะเบียน ทำงาน 4

    h, w = image.shape[:2]

    # จำนวน pixel ที่ตัด
    mx = int(w * margin)
    my = int(h * margin)

    cropped = image[
        my:h-my,
        mx:w-mx
    ]

    return cropped

def split_plate_image(image: np.ndarray, split_ratio: float): # ทำงาน 5
    h, w = image.shape[:2]
    cut = int(h * split_ratio)
    return image[:cut, :], image[cut:, :]

# Helpers
def parse_ocr_result(res): #ทำงาน 6.5
    """รองรับทั้ง dict และ object แบบ PaddleOCR v3"""
    if isinstance(res, dict):
        data = res.get("res", res)
        return data.get("rec_text", ""), data.get("rec_score", 0.0)
    if hasattr(res, "rec_text"):
        return res.rec_text, getattr(res, "rec_score", 0.0)
    if hasattr(res, "__dict__"):
        d = vars(res)
        return d.get("rec_text", ""), d.get("rec_score", 0.0)
    return str(res), 0.0

def ocr_image(image: np.ndarray, conf_threshold: float = 0.6): #ทำงาน 6
    """อ่าน OCR และกรองผลลัพธ์ที่ confidence ต่ำกว่า threshold ออก (ตามงานวิจัยโรมาเนีย)"""
    output = []
    for res in ocr.predict(image):
        text, score = parse_ocr_result(res)
        if score >= conf_threshold:
            output.append((text, score))
    return output

def clean_plate_number(text: str):
    text = text.replace(" ", "")
    length = len(text)

    if length == 6:
        text = "" + repair_thai(text[0:2]) + text[2:6]

    elif length == 7:
        text = text[0] + repair_thai(text[1:3]) + text[3:7]

    return text

def repair_thai(text):

    for old, new in THAI_MAP.items():
        text = text.replace(old, new)

    return text

def clean_province_text(text: str) -> str: #ทำงาน 8
    text = text.strip()
    for old, new in THAI_MAP.items():
        text = text.replace(old, new)
    text = re.sub(r'[^ก-๙]', '', text)
    return text

def match_province(text: str, threshold: int): #ทำงาน 9
    if not text:
        return None, 0
    result = process.extractOne(
        text,
        THAI_PROVINCES,
        scorer=fuzz.token_set_ratio,
    )
    if result is None:
        return None, 0
    match, score, _ = result
    if score >= threshold:
        return match, score
    return None, score

def character_vote(texts):

    texts = [t for t in texts if t]

    if not texts:
        return ""

    max_len = max(len(t) for t in texts)

    result = ""

    for i in range(max_len):

        chars = []

        for t in texts:

            if i < len(t):
                chars.append(t[i])

        if chars:
            result += Counter(chars).most_common(1)[0][0]

    return result

def province_vote(provinces):

    provinces = [p for p in provinces if p]

    if not provinces:
        return None

    return Counter(provinces).most_common(1)[0][0]


