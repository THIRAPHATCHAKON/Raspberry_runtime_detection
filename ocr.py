import re
import cv2
import numpy as np
from pathlib import Path
from paddleocr import TextRecognition
from rapidfuzz import process, fuzz
from collections import Counter
from itertools import combinations

ocr = TextRecognition(
    model_name="th_PP-OCRv5_mobile_rec"   # ตรงกับจุดที่เริ่ม fine-tune จริง
)

# ---------------------------------------------------------------------------
# Confusion maps
# ---------------------------------------------------------------------------
# แผนที่แก้ตัวอักษรที่ OCR อ่านผิดในโซน "พยัญชนะไทย" (ตัวละติน/สัญลักษณ์ -> ไทย)
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
    "A": "ฎ",
    "า": "ว",
    "ฤ": "ฎ",
}

# แผนที่แก้ตัวอักษรที่ OCR อ่านผิดในโซน "ตัวเลข" (ตัวละติน -> เลข)
# ปรับตัวเลข/ตัวอักษรใน map นี้ตามสถิติ error จริงของ engine ที่ใช้อยู่
DIGIT_MAP = {
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "i": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
    "m": "1",   # เคสจากตัวอย่างจริง: "1กคm456" -> m ควรเป็นเลขในโซนนี้
}

VALID_THAI_CONSONANTS = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")

# Patterns สำหรับยืนยันผลลัพธ์สุดท้าย
NEW_PLATE_RE = re.compile(r'^[1-9][ก-ฮ]{2}\d{1,4}$')
OLD_PLATE_RE = re.compile(r'^[ก-ฮ]{2}\d{1,4}$')
RED_PLATE_RE = re.compile(r'^[ก-ฮ]-\d{1,4}$')

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
    "อุทัยธานี", "อุบลราชธานี", "เบตง",
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


def read_plate(image_source, split_ratio,  # ทำงาน 2
                province_threshold,
                conf_threshold) -> dict:

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
    top_results = ocr_image(top_img, conf_threshold)
    plate_number_raw = " ".join(r[0] for r in top_results)
    license_id = clean_plate_number(plate_number_raw)
    bottom_results = ocr_image(bottom_img, conf_threshold)
    province_raw = " ".join(r[0] for r in bottom_results)
    province, prov_score = match_province(province_raw, threshold=province_threshold)

    return license_id, province


def order_points(pts):  # ทำงาน 3
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


def perspective_plate(plate):  # หมุนภาพ

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    # ลด Noise
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
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
    heightA = np.linalg.norm(box[1] - box[2])

    heightB = np.linalg.norm(box[0] - box[3])

    maxHeight = int(max(heightA, heightB))

    # จุดปลายทาง
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
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


def crop_border(image, margin):  # ตัดขอบภาพทะเบียน ทำงาน 4

    h, w = image.shape[:2]

    # จำนวน pixel ที่ตัด
    mx = int(w * margin)
    my = int(h * margin)

    cropped = image[
        my:h - my,
        mx:w - mx
    ]

    return cropped


def split_plate_image(image: np.ndarray, split_ratio: float):  # ทำงาน 5
    h, w = image.shape[:2]
    cut = int(h * split_ratio)
    return image[:cut, :], image[cut:, :]


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------
def parse_ocr_result(res):  # ทำงาน 6.5
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


def ocr_image(image: np.ndarray, conf_threshold: float = 0.6):  # ทำงาน 6
    """อ่าน OCR และกรองผลลัพธ์ที่ confidence ต่ำกว่า threshold ออก (ตามงานวิจัยโรมาเนีย)"""
    output = []
    for res in ocr.predict(image):
        text, score = parse_ocr_result(res)
        if score >= conf_threshold:
            output.append((text, score))
    return output


# ---------------------------------------------------------------------------
# Plate number cleaning / repair
# ---------------------------------------------------------------------------
def clean_plate_number(text: str):
    """
    ทำความสะอาดและแก้ไขเลขทะเบียนที่ OCR อ่านมา

    รองรับ 3 ปัญหาหลักที่พบบ่อย:
      1) ตัวอักษรละตินหลุดเข้ามาแทนพยัญชนะไทย (ผ่าน THAI_MAP)
      2) ตัวอักษรละตินหลุดเข้ามาแทนตัวเลข เช่น "1กคm456" (ผ่าน DIGIT_MAP)
      3) พยัญชนะเกินมา 1 ตัวจาก noise/เงา เช่น "1กคค457" (ผ่าน _reduce_extra_letters)

    ถ้าซ่อมแล้วยังไม่ตรงกับรูปแบบป้ายที่รู้จัก จะคืนค่าที่ clean เบื้องต้น
    (ตัดช่องว่างออก) พร้อม print แจ้งเตือนให้ตรวจสอบด้วยคน
    """
    raw = text.replace(" ", "").strip()
    if not raw:
        return ""

    fixed = _try_fix_plate(raw)
    if fixed:
        return fixed

    print(f"[WARN] ไม่สามารถยืนยันรูปแบบป้ายทะเบียนได้ ควรตรวจสอบด้วยคน: '{raw}'")
    return raw


def _try_fix_plate(raw: str):
    # กรณีมีขีด -> ป้ายแดง เช่น "ก-1234"
    if "-" in raw:
        return _fix_red_plate(raw)

    # แยกโซน "เลขท้าย" ออกจากโซน "พยัญชนะ (+เลขนำ)" ก่อน
    number_zone, letter_zone = _split_number_letter_zone(raw)
    if number_zone is None:
        return None

    fixed_number = _repair_digits(number_zone)
    if not fixed_number or not fixed_number.isdigit():
        return None

    # แยก prefix ตัวเลขจังหวัด/หมวด (ถ้ามี) ออกจากโซนพยัญชนะ
    prefix = ""
    letters = letter_zone
    if letters:
        first = letters[0]
        if first.isdigit() and first != "0":
            prefix = first
            letters = letters[1:]
        elif first in DIGIT_MAP and DIGIT_MAP[first] != "0":
            prefix = DIGIT_MAP[first]
            letters = letters[1:]

    fixed_letters = _repair_letters(letters)

    # ถ้าพยัญชนะเกิน 2 ตัว (noise ทำให้เห็นตัวเกิน) ให้ลองลดเหลือ 2 ตัวที่ valid
    if len(fixed_letters) > 2:
        fixed_letters = _reduce_extra_letters(fixed_letters)

    if len(fixed_letters) != 2 or not all(c in VALID_THAI_CONSONANTS for c in fixed_letters):
        return None

    plate = prefix + fixed_letters + fixed_number

    if NEW_PLATE_RE.match(plate) or OLD_PLATE_RE.match(plate):
        return plate

    return None


def _split_number_letter_zone(raw: str):
    """
    ดึงเลขรันท้ายสุดออกมาเป็นโซนตัวเลข ส่วนที่เหลือคือโซนพยัญชนะ (+prefix)
    นับตัวอักษรละตินที่ควรเป็นเลข (ตาม DIGIT_MAP) เป็นส่วนหนึ่งของโซนตัวเลขด้วย
    เพื่อจัดการเคสแบบ "1กคm456" ที่ m หลุดเข้ามาแทนเลข
    """
    idx = len(raw)
    for i in range(len(raw) - 1, -1, -1):
        ch = raw[i]
        if ch.isdigit() or ch in DIGIT_MAP:
            idx = i
        else:
            break

    if idx == len(raw):
        return None, raw  # ไม่มีโซนตัวเลขท้ายเลย ซ่อมไม่ได้

    return raw[idx:], raw[:idx]


def _repair_digits(text: str) -> str:
    return "".join(DIGIT_MAP.get(c, c) for c in text)


def _repair_letters(text: str) -> str:
    fixed = "".join(THAI_MAP.get(c, c) for c in text)
    # ตัดอักขระที่ไม่ใช่พยัญชนะไทยทิ้ง (สระ/วรรณยุกต์/สัญลักษณ์แปลกปลอมที่หลงเหลือ)
    return "".join(c for c in fixed if c in VALID_THAI_CONSONANTS)


def _reduce_extra_letters(letters: str) -> str:
    """
    เมื่อโซนพยัญชนะมีมากกว่า 2 ตัว เช่น "กคค" จาก noise/เงาบนป้าย:
      1) ยุบตัวอักษรที่ซ้ำติดกันเหลือตัวเดียวก่อน (แก้เคสอ่านตัวเดิมซ้ำ)
      2) ถ้ายังเกิน 2 ตัว ลองตัดออกทีละตัว/หลายตัว จนเหลือ 2 ตัวที่ valid
    """
    deduped = letters[0]
    for c in letters[1:]:
        if c != deduped[-1]:
            deduped += c

    if len(deduped) <= 2:
        return deduped

    for drop_count in range(1, len(deduped) - 1):
        for positions in combinations(range(len(deduped)), drop_count):
            candidate = "".join(c for i, c in enumerate(deduped) if i not in positions)
            if len(candidate) == 2 and all(c in VALID_THAI_CONSONANTS for c in candidate):
                return candidate

    return deduped[:2]  # fallback: เอา 2 ตัวแรกไปก่อน (จะถูกดักด้วย regex ท้ายทางอยู่ดี)


def _fix_red_plate(raw: str):
    parts = raw.split("-", 1)
    if len(parts) != 2:
        return None
    letter, number = parts
    letter = _repair_letters(letter)
    number = _repair_digits(number)
    if len(letter) != 1 or not number.isdigit():
        return None
    plate = f"{letter}-{number}"
    return plate if RED_PLATE_RE.match(plate) else None


# ---------------------------------------------------------------------------
# Province matching
# ---------------------------------------------------------------------------
def clean_province_text(text: str) -> str:  # ทำงาน 8
    text = text.strip()
    for old, new in THAI_MAP.items():
        text = text.replace(old, new)
    text = re.sub(r'[^ก-๙]', '', text)
    return text


def match_province(text: str, threshold: int):  # ทำงาน 9
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


# ---------------------------------------------------------------------------
# Voting across multiple frames
# ---------------------------------------------------------------------------
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