#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
identify_person_face_and_clothes.py
───────────────────────────────────
Find one person by **both**
  1. Face similarity (Haar/MTCNN + FaceNet)     ← higher priority
  2. Shirt-colour match (YOLO + Pose + HSV/Lab)

Bounding-box colours
────────────────────
🟥 RED     – Face **and** clothing match
🟧 ORANGE  – Face match only
🟩 GREEN   – Clothing match only

Updates in this version
──────────────────────
• Larger YOLO inference size (better tiny-person recall)
• Upscale crops < 120 px before analysis
• Adaptive colour-ratio threshold for small ROIs
• Wider “baby-pink” hue band (150–179 °)
• Optional YOLO tracking (ID-stable boxes)
• **NEW:** clean real-time CSV logging of RED / ORANGE boxes
"""

from __future__ import annotations
import os, sys, cv2, numpy as np, torch, logging
from datetime import datetime
from types import SimpleNamespace

# ───────────────────────────── LOGGING ────────────────────────────────────────
LOG_CSV = "detections.csv"                       # grows while video runs

# console logger
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("detector")

# dedicated CSV logger (no prefixes)
csv_logger = logging.getLogger("detector.csv")
csv_logger.setLevel(logging.INFO)
csv_handler = logging.FileHandler(LOG_CSV, mode="w", encoding="utf-8")
csv_handler.setFormatter(logging.Formatter("%(message)s"))
csv_logger.addHandler(csv_handler)
csv_logger.propagate = False
csv_logger.info("timestamp,bbox_colour,overall_score,cloth_score")

logger.info("Script started — detections will be stored in %s", LOG_CSV)

# ─────────────────────────── USER CONFIG ──────────────────────────────────────
INPUT_PATH       = r"C:\Users\Pc\Downloads\dataset_cv.mp4"
REFERENCE_IMAGE  = r"C:\Users\Pc\Pictures\Screenshots\Screenshot 2025-05-14 103002.png"
QUERY            = "baby pink shirt"
YOLO_WEIGHTS     = "yolov8n.pt"                   # yolov8m.pt / yolov8l.pt for better recall
HAAR_XML         = r"C:/Users/Pc/Documents/hass.xml"

CONF_THRES, IOU_THRES = 0.25, 0.5
SHOW_SCALE            = (1280, 720)

# weights (must sum 1)
FACE_WT, CLOTH_WT = 0.7, 0.3

# colour-scorer blend (≈1)
RATIO_WT, COS_WT, KNN_WT = 0.40, 0.30, 0.30
K_VAL = 3

# robustness tweaks
YOLO_IMG_SIZE  = 960
MIN_CROP_SIDE  = 120
UPSCALE_FACTOR = 2.0
USE_TRACKING   = False

# ───────────────────── 3rd-PARTY IMPORTS ──────────────────────────────────────
try:
    from ultralytics import YOLO
    import mediapipe as mp
    from sklearn.neighbors import KNeighborsClassifier
    from facenet_pytorch import InceptionResnetV1
except ImportError as e:
    sys.exit(f"❌  {e.name} not installed — please pip install the listed dependencies")

# ──────────────────────── COLOUR TABLES ───────────────────────────────────────
HSV_BANDS = {
    "black":  [([0,   0,   0],  [180, 255,  40])],
    "white":  [([0,   0, 200],  [180,  60, 255])],
    "red":    [([0, 120,  60],  [ 10, 255, 255]),
               ([170,120, 60],  [180, 255, 255])],
    "blue":   [([100,120, 60],  [130, 255, 255])],
    "green":  [([40,  70, 70],  [ 80, 255, 255])],
    "yellow": [([20,100,100],   [ 35, 255, 255])],
    "orange": [([10,100,100],   [ 20, 255, 255])],
    "brown":  [([ 5,  50, 40],  [ 20, 255, 200])],
    "grey":   [([ 0,   0, 50],  [180,  40, 220])],
    "baby pink": [([150, 20, 150], [179, 120, 255])]
}

REF_LAB = {
    "black":     (20,  0,   0),
    "white":     (95,  0,   0),
    "red":       (53, 80,  67),
    "blue":      (32, 79,-108),
    "green":     (46,-51,  49),
    "yellow":    (93,-15,  94),
    "orange":    (65, 45,  65),
    "brown":     (46, 23,  18),
    "grey":      (60,  0,   0),
    "baby pink": (82, 18,   5)
}

# ─────────────────────────── HELPER FUNCS ─────────────────────────────────────
def build_masker(query: str):
    colour = next((c for c in HSV_BANDS if c in query.lower()), None)
    if not colour:
        sys.exit(f"Colour not recognised — supported: {', '.join(HSV_BANDS)}")

    bands, ref_lab = HSV_BANDS[colour], np.array(REF_LAB[colour])
    logger.info("Building HSV masker for '%s'", colour)

    def mask_fn(bgr_roi: np.ndarray) -> np.ndarray:
        hsv   = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
        v_med = np.median(hsv[..., 2])
        adj   = []
        for lo, hi in bands:
            lo, hi = lo.copy(), hi.copy()
            if colour == "black":
                hi[2] = int(max(30,  v_med*0.45))
            elif colour == "white":
                lo[2] = int(max(180, v_med*0.85))
            elif colour in ("brown", "grey"):
                hi[2] = int(min(hi[2], v_med*1.4))
            adj.append((np.array(lo), np.array(hi)))
        mask = sum(cv2.inRange(hsv, *r) for r in adj)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        return mask

    return colour, ref_lab, mask_fn


def lab_histogram(bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    lab     = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    samples = lab.reshape(-1, 3)
    hist, _ = np.histogramdd(samples, bins=(bins,bins,bins), range=((0,255),)*3)
    hist = hist.astype(np.float32).flatten()
    if hist.sum():
        hist /= hist.sum()
    return hist


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u.dot(v)/d) if d else 0.0


def lab_to_bgr(lab_triplet):
    return cv2.cvtColor(np.uint8([[lab_triplet]]), cv2.COLOR_Lab2BGR)[0,0]


def adaptive_ratio(px: int) -> float:
    if px > 40_000: return 0.60
    if px > 15_000: return 0.50
    if px >  5_000: return 0.40
    return 0.30

# ──────────────── FACE RECOGNISER INITIALISATION ─────────────────────────────
logger.info("Initialising face recogniser …")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_cascade = cv2.CascadeClassifier(HAAR_XML)
facenet      = InceptionResnetV1(pretrained="vggface2").eval().to(device)
logger.info("FaceNet loaded on %s", device)

def get_embedding(face_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img_160 = cv2.resize(img_rgb, (160,160))
    tensor  = torch.from_numpy(img_160).permute(2,0,1).float().unsqueeze(0)/255.0
    return facenet(tensor.to(device)).detach().cpu()

# ─────────────────── LOAD & EMBED REFERENCE FACE ─────────────────────────────
logger.info("Loading reference image %s", REFERENCE_IMAGE)
ref_img = cv2.imread(REFERENCE_IMAGE)
if ref_img is None:
    sys.exit("❌  Cannot read reference image")
faces_ref = face_cascade.detectMultiScale(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY),
                                          scaleFactor=1.1, minNeighbors=5)
if len(faces_ref) == 0:
    sys.exit("❌  No face detected in reference image")

x,y,w,h            = faces_ref[0]
reference_embedding = get_embedding(ref_img[y:y+h, x:x+w])
logger.info("Reference face embedding computed")

def face_match(face_bgr: np.ndarray, threshold: float = 0.8):
    emb  = get_embedding(face_bgr)
    dist = torch.norm(reference_embedding - emb).item()
    return dist < threshold, dist

# ───────────── CLOTHING SCORER (HSV+LAB+KNN) SETUP ───────────────────────────
colour_word, ref_lab, mask_fn = build_masker(QUERY)

hists, labels = [], []
for cname, lab in REF_LAB.items():
    patch = np.full((32,32,3), lab_to_bgr(lab), dtype=np.uint8)
    hists.append(lab_histogram(patch))
    labels.append(cname)

knn = KNeighborsClassifier(n_neighbors=K_VAL, metric="cosine")
knn.fit(hists, labels)
logger.info("KNN trained for colour classification (k=%d)", K_VAL)

target_idx = knn.classes_.tolist().index(colour_word)
ref_hist   = lab_histogram(np.full((32,32,3), lab_to_bgr(ref_lab), dtype=np.uint8))

# ──────────────────── LOAD YOLO + MEDIA-PIPE POSE ────────────────────────────
print("📦  Loading YOLOv8 and MediaPipe …")
logger.info("Loading YOLO weights %s", YOLO_WEIGHTS)
yolo  = YOLO(YOLO_WEIGHTS)
pose  = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)

# ───────────────────────── PERSON ANALYSIS ───────────────────────────────────
def analyse_person(crop: np.ndarray) -> SimpleNamespace:
    h, w = crop.shape[:2]
    if min(h, w) < MIN_CROP_SIDE:
        crop = cv2.resize(crop,
                          (int(w*UPSCALE_FACTOR), int(h*UPSCALE_FACTOR)),
                          interpolation=cv2.INTER_CUBIC)
        h, w = crop.shape[:2]

    # face test
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    face_boxes = face_cascade.detectMultiScale(gray, 1.1, 5)
    face_ok, face_dist = False, None
    if len(face_boxes):
        fx,fy,fw,fh = max(face_boxes, key=lambda b: b[2]*b[3])
        face_ok, face_dist = face_match(crop[fy:fy+fh, fx:fx+fw])

    face_score = 1.0 if face_ok else 0.0

    # torso ROI via pose
    res = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if res.pose_landmarks:
        lm   = res.pose_landmarks.landmark
        xs   = [p.x*w for p in (lm[11], lm[12], lm[23], lm[24])]
        ys   = [p.y*h for p in (lm[11], lm[12], lm[23], lm[24])]
        rx1, rx2 = int(max(0,min(xs)-0.05*w)), int(min(w,max(xs)+0.05*w))
        ry1, ry2 = int(max(0,min(ys)-0.05*h)), int(min(h,max(ys)+0.05*h))
        roi = crop[ry1:ry2, rx1:rx2]
    else:
        roi = crop[:h//2, :]

    cloth_score, cloth_ok = 0.0, False
    if roi.size:
        mask  = mask_fn(roi)
        ratio = cv2.countNonZero(mask) / mask.size
        if ratio >= adaptive_ratio(mask.size):
            hist = lab_histogram(roi)
            cos  = cosine(hist, ref_hist)
            prob = knn.predict_proba([hist])[0][target_idx]
            cloth_score = RATIO_WT*ratio + COS_WT*cos + KNN_WT*prob
            cloth_ok    = True

    total_score = FACE_WT*face_score + CLOTH_WT*cloth_score
    return SimpleNamespace(face=face_ok,
                           cloth=cloth_ok,
                           score=total_score,
                           cloth_score=cloth_score)

# ───────────────────────── FRAME PROCESSING ──────────────────────────────────
def detect_frame(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    if USE_TRACKING:
        det = yolo.track(img, persist=True,
                         conf=CONF_THRES, iou=IOU_THRES,
                         imgsz=YOLO_IMG_SIZE, verbose=False)[0]
    else:
        det = yolo(img, conf=CONF_THRES, iou=IOU_THRES,
                   imgsz=YOLO_IMG_SIZE, verbose=False)[0]

    people = [b for b in det.boxes.data.cpu().numpy() if int(b[5]) == 0]

    results = []
    for (x1, y1, x2, y2, conf, _) in people:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        crop = img[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
        if crop.size == 0:  continue
        stats = analyse_person(crop)
        results.append((stats, (x1,y1,x2,y2)))

    # draw & CSV-log
    for stats, (x1,y1,x2,y2) in results:
        if stats.face and stats.cloth:
            colour, tag = (0,0,255), "RED"
        elif stats.face:
            colour, tag = (0,140,255), "ORANGE"
        else:
            continue  # we draw only matched cases

        cv2.rectangle(img, (x1,y1), (x2,y2), colour, 2)
        cv2.putText(img, tag, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cloth_val = f"{stats.cloth_score:.2f}" if stats.cloth else ""
        csv_logger.info(f"{ts},{tag},{stats.score:.2f},{cloth_val}")

    return img

# ────────────────────────────── MAIN LOOP ─────────────────────────────────────
ext = os.path.splitext(INPUT_PATH)[1].lower()
if ext in {".jpg", ".jpeg", ".png"}:
    frame = cv2.imread(INPUT_PATH)
    if frame is None:
        sys.exit("❌  Could not read image")
    out = detect_frame(frame)
    h, w = out.shape[:2]
    s    = min(SHOW_SCALE[0]/w, SHOW_SCALE[1]/h, 1.0)
    cv2.imshow("Result", cv2.resize(out, (int(w*s), int(h*s))) if s < 1 else out)
    cv2.waitKey(0); cv2.destroyAllWindows()

elif ext in {".mp4", ".avi", ".mov", ".mkv"}:
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        sys.exit("❌  Could not open video")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out  = detect_frame(frame)
        h, w = out.shape[:2]
        s    = min(SHOW_SCALE[0]/w, SHOW_SCALE[1]/h, 1.0)
        disp = cv2.resize(out, (int(w*s), int(h*s))) if s < 1 else out
        cv2.imshow("Result", disp)
        if cv2.waitKey(delay) & 0xFF in (27, ord('q')):
            break
        frame_idx += 1
        if frame_idx % 30 == 0:
            logger.info("Processed %d frames", frame_idx)

    cap.release(); cv2.destroyAllWindows()
    logger.info("Video processing complete — processed %d frames", frame_idx)

else:
    sys.exit("❌  Unsupported file type")
