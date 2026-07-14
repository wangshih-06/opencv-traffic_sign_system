
"""
Generate course design report PDF for the Traffic Sign Classification System.
Uses reportlab to construct a comprehensive document with architecture diagrams,
feature descriptions, model comparisons, and evaluation frameworks.
"""

from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Preformatted, Image,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Traffic_Sign_Classification_Course_Design_Report.pdf"
CLASS_DIST_PNG = ROOT / "traffic_sign_system" / "models" / "artifacts" / "class_distribution.png"

pdfmetrics.registerFont(TTFont("SimFang", r"C:\Windows\Fonts\simfang.ttf"))
pdfmetrics.registerFont(TTFont("SimHei", r"C:\Windows\Fonts\simhei.ttf"))

PAGE_W, PAGE_H = A4
BLUE = colors.HexColor("#124E78")
ACCENT = colors.HexColor("#1F7A8C")
LIGHT = colors.HexColor("#EAF3F7")
MID = colors.HexColor("#C6DEE8")
DARK = colors.HexColor("#1F2933")
GRAY = colors.HexColor("#5B6770")

_styles = getSampleStyleSheet()
for _name, _kw in [
    ("CoverTitle", {"fontName":"SimHei","fontSize":25,"leading":34,"alignment":TA_CENTER,"textColor":BLUE,"spaceAfter":14}),
    ("CoverSub", {"fontName":"SimFang","fontSize":13,"leading":22,"alignment":TA_CENTER,"textColor":GRAY}),
    ("Sec", {"fontName":"SimHei","fontSize":15,"leading":22,"textColor":BLUE,"spaceBefore":16,"spaceAfter":8,"keepWithNext":True}),
    ("Sub", {"fontName":"SimHei","fontSize":11.5,"leading":17,"textColor":ACCENT,"spaceBefore":10,"spaceAfter":5,"keepWithNext":True}),
    ("Bd", {"fontName":"SimFang","fontSize":9.7,"leading":16,"textColor":DARK,"spaceAfter":7}),
    ("Bul", {"fontName":"SimFang","fontSize":9.5,"leading":15,"leftIndent":16,"firstLineIndent":-11,"textColor":DARK,"spaceAfter":4}),
    ("Sml", {"fontName":"SimFang","fontSize":8.4,"leading":12.5,"textColor":DARK}),
    ("Cd", {"fontName":"SimFang","fontSize":7.6,"leading":11,"textColor":colors.HexColor("#16324F"),"backColor":colors.HexColor("#F5F8FA"),"borderColor":MID,"borderWidth":0.5,"borderPadding":6,"spaceBefore":3,"spaceAfter":8}),
    ("Cal", {"fontName":"SimFang","fontSize":9.3,"leading":15,"textColor":colors.HexColor("#093B4C"),"backColor":colors.HexColor("#E7F4F7"),"borderColor":colors.HexColor("#8FC9D5"),"borderWidth":0.6,"borderPadding":8,"spaceBefore":5,"spaceAfter":9}),
    ("Arrow", {"fontName":"SimHei","fontSize":12,"leading":13,"alignment":TA_CENTER,"textColor":ACCENT}),
]:
    _styles.add(ParagraphStyle(_name, **_kw))

class _IP(Paragraph):
    def __iter__(self): return iter((self,))
class _IT(Table):
    def __iter__(self): return iter((self,))
class _IPF(Preformatted):
    def __iter__(self): return iter((self,))

def P(text, style="Bd"): return _IP(escape(text).replace("\n","<br/>"), _styles[style])
def R(text, style="Bd"): return _IP(text, _styles[style])
def B(items): return [Paragraph("\u2022 "+escape(x), _styles["Bul"]) for x in items]
def C(text): return _IPF(text.strip("\n"), _styles["Cd"])
def S(title): return [Paragraph(title, _styles["Sec"])]
def SS(title): return [Paragraph(title, _styles["Sub"])]

def T(headers, rows, widths=None):
    data = [[Paragraph(escape(str(x)), _styles["Sml"]) for x in headers]]
    for row in rows:
        data.append([Paragraph(escape(str(x)), _styles["Bd"]) for x in row])
    t = _IT(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),BLUE), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"SimHei"), ("ALIGN",(0,0),(-1,0),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("GRID",(0,0),(-1,-1),0.35,MID),
        ("BACKGROUND",(0,1),(-1,-1),colors.white),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F7FBFC")]),
        ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),5), ("RIGHTPADDING",(0,0),(-1,-1),5),
    ]))
    return t

def HF(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    if page > 1:
        canvas.setStrokeColor(MID); canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, PAGE_H-1.45*cm, PAGE_W-doc.rightMargin, PAGE_H-1.45*cm)
        canvas.setFont("SimFang",8); canvas.setFillColor(GRAY)
        canvas.drawString(doc.leftMargin, PAGE_H-1.1*cm, "Traffic Sign Classification System - Course Design Report")
        canvas.drawRightString(PAGE_W-doc.rightMargin, 1.05*cm, f"Page {page-1}")
        canvas.setStrokeColor(MID)
        canvas.line(doc.leftMargin, 1.35*cm, PAGE_W-doc.rightMargin, 1.35*cm)
    canvas.restoreState()

print("Functions defined OK")


# =========================== STORY ===========================
story = []

# -- Cover --
story += [Spacer(1,3*cm),
          Paragraph("Based on OpenCV and Support Vector Machine", _styles["CoverTitle"]),
          Paragraph("Traffic Sign Classification and Recognition System", _styles["CoverTitle"]),
          Spacer(1,0.6*cm),
          Paragraph("Design and Implementation", _styles["CoverSub"]),
          Paragraph("Computer Vision Course Design Report", _styles["CoverSub"]),
          Spacer(1,2*cm)]

cv = Table([
    [P("Core Tech","Sml"),P("OpenCV / HOG / HSV / SVM / KNN / RF / scikit-learn / PyQt5","Sml")],
    [P("Task","Sml"),P("Traffic sign classification (core) + traditional color/contour detection (extension)","Sml")],
    [P("Dataset","Sml"),P("GTSRB - 43 classes, 39,209 training samples, 12,630 test samples","Sml")],
    [P("Environment","Sml"),P("Python 3.12 / Windows 11 / CPU-only training and inference","Sml")],
    [P("Purpose","Sml"),P("Course design proposal, system design, experiment implementation, defense presentation","Sml")],
], colWidths=[3.3*cm,10*cm], hAlign="CENTER")
cv.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(0,-1),BLUE),("TEXTCOLOR",(0,0),(0,-1),colors.white),
    ("BACKGROUND",(1,0),(1,-1),colors.HexColor("#F3F8FA")),
    ("GRID",(0,0),(-1,-1),0.5,MID),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
]))
story += [cv, Spacer(1,2.5*cm), Paragraph("Date: July 11, 2026", _styles["CoverSub"]), PageBreak()]

# -- TOC --
story += S("Table of Contents")
story += B([
    "1. Project Overview and Scope",
    "2. System Architecture - Data Layer -> Algorithm Layer -> Application Layer",
    "3. Technology Stack and Runtime Environment",
    "4. Dataset Selection and Sample Distribution",
    "5. OpenCV Image Preprocessing Module",
    "6. Feature Extraction: HOG, HSV Histogram and Feature Fusion",
    "7. Classification Models: SVM / KNN / Random Forest",
    "8. Training Pipeline and Hyperparameter Selection",
    "9. Model Evaluation and Results Analysis",
    "10. SVM vs KNN vs RF Multi-Model Comparison",
    "11. Error Samples and Confusable Class Analysis",
    "12. PyQt5 Graphical User Interface Design",
    "13. Traditional Vision Detection Extension (SignDetector)",
    "14. Improvement Directions and Outlook",
    "15. Limitations and Conclusion",
])
story += [Spacer(1,0.3*cm),
          R("This course design is based on the GTSRB dataset with OpenCV HOG + SVM as the core pipeline. It covers data reading, preprocessing, feature engineering, model training, multi-model comparison, error analysis, GUI interaction, and traditional detection extension.", "Cal"),
          PageBreak()]

# 1. Overview
story += S("1. Project Overview and Scope")
story += [P("With the rapid development of Intelligent Transportation Systems (ITS) and driver assistance technologies, automatic traffic sign recognition has become a fundamental research topic in computer vision. This project follows a traditional machine learning approach using OpenCV for image preprocessing and feature extraction, with SVM as the core classifier, building a complete traffic sign classification and recognition system."),
          P("The system core focuses on traffic sign classification: input is a single cropped traffic sign image, output is class ID, class name and confidence. Additionally, a traditional detection extension locates candidate sign regions in full road scenes using HSV color masking, morphological operations, and contour geometric filtering before classification.")]
story += T(["Item","Classification Task","Detection Extension"],
    [["Input","Cropped single sign (64x64)","Road scene image, video frame, or camera feed"],
     ["Output","Class ID, name, confidence","BBox (x,y,w,h), class, confidence"],
     ["Tech Focus","HOG/HSV features, SVM multi-class","HSV mask, morphology, contour filtering + classification"],
     ["Priority","Required (core)","Optional / bonus"]],
    [2.4*cm,5.2*cm,5.8*cm])
story += [P("The project forms a complete closed loop: dataset management -> image preprocessing -> feature extraction -> model training -> model evaluation -> real-time recognition.")]

# 2. Architecture
story += S("2. System Architecture")
story += [P("The system adopts a three-layer architecture: Data Layer handles dataset management and model artifact persistence; Algorithm Layer implements preprocessing, feature building, training, evaluation, and recognition; Application Layer provides CLI scripts and PyQt5 GUI. Standardized data interfaces (NumPy arrays, joblib bundles, Pandas DataFrames) connect the layers.")]
story += SS("2.1 Overall Architecture Diagram")
arch = Table([
    [P("[DATA LAYER]","Sml"), P("dataset/train, dataset/test, labels.csv, models/artifacts/*.joblib","Sml")],
    [Paragraph("|",_styles["Arrow"]), Paragraph("|",_styles["Arrow"])],
    [Paragraph("v",_styles["Arrow"]), Paragraph("v",_styles["Arrow"])],
    [P("[ALGORITHM LAYER]","Sml"),
     P("DataLoader -> Preprocessor -> Augmentation -> FeatureBuilder(HOG/HSV) -> Trainer(SVM/KNN/RF) -> Evaluator -> Predictor -> SignDetector","Sml")],
    [Paragraph("|",_styles["Arrow"]), Paragraph("|",_styles["Arrow"])],
    [Paragraph("v",_styles["Arrow"]), Paragraph("v",_styles["Arrow"])],
    [P("[APPLICATION LAYER]","Sml"),
     P("CLI scripts: check_dataset, build_features, train, evaluate, compare, error_stats, predict_one / PyQt5 GUI: Image, Video, Camera, Scene Detection","Sml")],
], colWidths=[4.2*cm,12.2*cm], hAlign="CENTER")
arch.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(0,0),BLUE),("TEXTCOLOR",(0,0),(0,0),colors.white),
    ("BACKGROUND",(0,3),(0,3),ACCENT),("TEXTCOLOR",(0,3),(0,3),colors.white),
    ("BACKGROUND",(0,6),(0,6),colors.HexColor("#2C5F2D")),("TEXTCOLOR",(0,6),(0,6),colors.white),
    ("BACKGROUND",(1,0),(1,0),LIGHT),("BACKGROUND",(1,3),(1,3),LIGHT),("BACKGROUND",(1,6),(1,6),LIGHT),
    ("GRID",(0,0),(-1,-1),0.5,MID),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
]))
story += [arch, Spacer(1,0.3*cm)]

story += SS("2.2 Module Dependency Flow")
story += [P("Core data path: data loading -> preprocessing -> feature building -> model training -> model evaluation -> model persistence -> prediction inference."),
          C("""DataLoader.load_train_data()        # (images, labels, class_ids)
    |
Preprocessor(img_bgr)               # -> grayscale uint8 (64x64)
    | [opt] Augmentation.apply_random(img, p=0.5)
FeatureBuilder.extract_batch()      # -> (N, D) float32
    | StandardScaler.fit_transform()
SVC / KNeighbors / RandomForest.fit()
    |
Evaluator -> confusion_matrix.png, metrics.json, errors.csv
    | joblib.dump()
Predictor(predict_proba) -> {class_id, class_name, confidence}""")]

# 3. Tech Stack
story += S("3. Technology Stack and Runtime Environment")
story += SS("3.1 Language and Development Environment")
story += [P("Python 3.12.7 (Anaconda) / Windows 11. OpenCV provides efficient image I/O and feature extraction, scikit-learn offers comprehensive traditional ML APIs, PyQt5 enables rapid desktop GUI development. All modules run on CPU without GPU dependency.")]
story += SS("3.2 Core Dependencies")
story += T(["Library","Version","Primary Role"],
    [["opencv-python","4.x","Image I/O, preprocessing, HOGDescriptor, video/camera"],
     ["NumPy","1.x","Array operations, feature concatenation, image matrices"],
     ["scikit-learn","1.x","SVC/KNN/RF classifiers, StandardScaler, stratified split, metrics"],
     ["Matplotlib","3.x","Confusion matrix, class distribution charts, comparison plots"],
     ["Pandas","2.x","Label CSV parsing, DataFrame statistics export"],
     ["Joblib","1.x","Model bundle serialization/deserialization"],
     ["PyQt5","5.15","QMainWindow, QThread, signal/slot, QTimer"]],
    [3.6*cm,2.2*cm,10.6*cm])
story += SS("3.3 Environment Setup")
story += C("""conda create -n traffic-sign python=3.12
conda activate traffic-sign
pip install opencv-python numpy pandas matplotlib scikit-learn joblib PyQt5""")

# 4. Dataset
story += S("4. Dataset Selection and Sample Distribution")
story += [P("GTSRB (German Traffic Sign Recognition Benchmark) is the classic benchmark for traffic sign classification. It contains 43 classes, 39,209 training images (varying 15x15 to 250x250 sizes), and 12,630 test images (with ROI annotation CSV). Classes include speed limits, prohibitory signs, warning signs, and other signs.")]
if CLASS_DIST_PNG.is_file():
    story += SS("4.1 Class Sample Distribution")
    story += [P("Below is the sample count distribution of the 43 GTSRB training classes. Note the class imbalance: some classes have 2,000+ samples while minority classes have only a few hundred. This may affect minority class recall.")]
    img_w = 14*cm
    story += [Image(str(CLASS_DIST_PNG), width=img_w, height=img_w*0.62)]
    story += [P("(Generated by scripts/check_dataset.py at models/artifacts/class_distribution.png)")]

story += SS("4.2 GTSRB 43 Class Labels")
story += T(["ID","Name","ID","Name","ID","Name"],
    [["0","Speed limit 20","1","Speed limit 30","2","Speed limit 50"],
     ["3","Speed limit 60","4","Speed limit 70","5","Speed limit 80"],
     ["6","End 80 limit","7","Speed limit 100","8","Speed limit 120"],
     ["9","No overtaking","10","No truck overtaking","11","Intersection priority"],
     ["12","Priority road","13","Yield","14","Stop"],
     ["15","No vehicles","16","No trucks","17","No entry"],
     ["18","General caution","19","Left curve","20","Right curve"],
     ["21","Double curve","22","Bumpy road","23","Slippery road"],
     ["24","Road narrows","25","Road work","26","Traffic signals"],
     ["27","Pedestrians","28","Children","29","Bicycles"],
     ["30","Snow/Ice","31","Wild animals","32","End all restrictions"],
     ["33","Turn right","34","Turn left","35","Ahead only"],
     ["36","Ahead or right","37","Ahead or left","38","Keep right"],
     ["39","Keep left","40","Roundabout","41","End no overtaking"],
     ["42","End truck overtaking","-","-","-","-"]],
    [2.2*cm,4.3*cm,2.2*cm,4.3*cm,2.2*cm,4.3*cm])

print("Story sections 1-4 built OK")


# 5. Preprocessing
story += S("5. OpenCV Image Preprocessing Module")
story += [P("The Preprocessor class implements a configurable preprocessing chain: BGR -> Resize(64x64) -> Grayscale (optional) -> Gaussian Blur (optional) -> CLAHE (optional) -> Normalization (divide255 or minmax). All parameters are recorded in self.config for training/prediction consistency."),
          P("Output is uint8 grayscale (64,64) for HOG consumption; HSV feature mode retains BGR color input.")]
story += SS("5.1 Preprocessing Pipeline")
story += C("""class Preprocessor:
    def __init__(self, img_size=64, to_gray=True,
                 gaussian_ksize=3,      # 0=disabled
                 clahe=True, clahe_clip=2.0, clahe_grid=8,
                 normalize="divide255"): # "divide255" | "minmax"
        ...

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        # 1) cv2.resize -> (64, 64)
        # 2) cv2.COLOR_BGR2GRAY (if to_gray=True)
        # 3) cv2.GaussianBlur (if gaussian_ksize > 0)
        # 4) cv2.createCLAHE -> apply (if clahe=True)
        # 5) divide255: float32/255 -> clip -> *255 -> uint8
        #    minmax:   cv2.normalize(NORM_MINMAX) -> uint8
        # Returns uint8 grayscale (64, 64)""")
story += SS("5.2 Data Augmentation (Training Only)")
story += [P("augmentation.py provides random affine transforms, brightness/contrast adjustment, Gaussian noise, and blur, applied with 0.5 probability stacking 0-2 augmentations. Key constraint: NO horizontal flip (directional semantics would be destroyed - left turn becomes right turn, numbers/text reversed)."),
          C("""def random_affine(img, max_angle=10, max_shift=0.1, max_scale=0.1)
def random_brightness_contrast(img, brightness=0.2, contrast=0.2)
def gaussian_noise(img, sigma=5)
def gaussian_blur(img, ksize=3)
def apply_random(img, p=0.5):  # randomly applies 0-2 augmentations""")]

# 6. Feature Extraction
story += S("6. Feature Extraction: HOG, HSV Histogram and Feature Fusion")
story += [P("The system supports three feature modes managed by FeatureBuilder: HOG-only (1764 dims), HSV-only (64 dims), HOG+HSV fusion (1828 dims).")]
story += SS("6.1 HOG - Histogram of Oriented Gradients")
story += [P("HOG describes shape and edge contours by counting local gradient orientation distributions. Traffic signs have stable geometric shapes (circle, triangle, octagon, rectangle) and internal patterns (numbers, arrows, symbols), making HOG ideal for this task."),
          P("Gradient computation: for each pixel, Gx=dI/dx, Gy=dI/dy, magnitude m=sqrt(Gx^2+Gy^2), orientation theta=arctan(Gy/Gx). The 64x64 image is divided into 8x8 pixel cells, each 2x2 cells form a block, blocks slide with 8-pixel stride. Each cell accumulates a weighted histogram over 9 orientation bins (0-180 degrees, unsigned gradient).")]
story += T(["Parameter","Value","Description"],
    [["winSize","64x64","Input image size, matches IMG_SIZE"],
     ["blockSize","16x16","Normalization block (2x2 cells)"],
     ["blockStride","8x8","Block stride (1 cell)"],
     ["cellSize","8x8","Cell size"],
     ["nbins","9","Orientation bins (0-180 deg)"],
     ["Feature Dim","1764","(64/8-1)^2 x (16^2/8^2) x 9 = 7x7x4x9"]],
    [3.2*cm,3.0*cm,10.2*cm])
story += SS("6.2 HSV Color Histogram")
story += [P("Using only grayscale HOG loses color information (e.g., red prohibitory vs blue mandatory signs). The HSV histogram extracts the 2D distribution of hue H and saturation S, L1-normalized to a fixed-dimension color feature vector."),
          C("""hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0,1], None, [8,8], [0,180,0,256])
hist = hist.reshape(-1).astype(np.float32) / (hist.sum() + 1e-6)
# -> 64-dim L1-normalized vector""")]
story += SS("6.3 Feature Fusion")
story += [P("FeatureBuilder selects the feature mode based on the 'mode' parameter:")]
story += B([
    "hog - HOG only (1764 dims), suitable for classes with distinct shapes",
    "hsv - HSV histogram only (64 dims), suitable for classes with distinct colors",
    "hog+hsv - Concatenation (1828 dims), fuses shape and color, typically yields best results",
])
story += [P("After fusion, StandardScaler is fit on training data and applied to all splits before classification.")]

# 7. Classification Models
story += S("7. Classification Models: SVM / KNN / Random Forest")
story += [P("Three classifiers are implemented for comparison: SVM (primary), KNN (baseline), Random Forest (comparison). All models are persisted via a unified save_bundle/load_bundle interface.")]
story += SS("7.1 SVM - Support Vector Machine (Primary Model)")
story += [P("Uses RBF kernel SVC with probability=True for confidence output. RBF kernel handles the non-linear separability of high-dimensional HOG features. Default hyperparameters: C=10, gamma='scale'. Multi-class (43 classes) via One-vs-One strategy."),
          C("""from sklearn.svm import SVC
model = SVC(C=10, kernel="rbf", gamma="scale",
            probability=True, random_state=42)
model.fit(X_train_scaled, y_train)""")]
story += SS("7.2 KNN - K-Nearest Neighbors (Baseline)")
story += [P("KNN provides an intuitive baseline. Uses distance weighting: closer neighbors have higher weights. Default k=5. Prediction requires computing distances from the test sample to ALL training samples, making inference slower."),
          C("""from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)""")]
story += SS("7.3 Random Forest (Comparison Model)")
story += [P("Random Forest is an ensemble of decision trees, insensitive to feature scaling. Feature importance can be inspected. Default n_estimators=200."),
          C("""from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=None,
                                random_state=42, n_jobs=-1)""")]
story += SS("7.4 Model Persistence")
story += [P("model_manager.py packages the classifier, StandardScaler, label mapping, feature config, and training summary into a single .joblib bundle. Predictor validates feature dimensions on load."),
          C("""bundle = {
    "classifier": model,      # classifier object
    "scaler": scaler,         # fitted StandardScaler
    "label_map": {0:"Speed limit 20", ...},
    "feature_config": {"mode":"hog+hsv","img_size":64,...},
    "summary": {"model":"svm","feature_mode":"hog+hsv",
               "n_train":62734,"feature_dim":1828,
               "train_seconds":123.4,...}
}
joblib.dump(bundle, "svm_hog+hsv.joblib")""")]

# 8. Training Pipeline
story += S("8. Training Pipeline and Hyperparameter Selection")
story += [P("Training strictly follows the no-test-set-leakage principle: StandardScaler is fit and classifier is trained only on the training set; validation set is used for hyperparameter selection; test set is used only once for final evaluation.")]
story += SS("8.1 Training Flow")
story += C("""1) load_train_data(root) -> (images, labels)
2) stratified_split -> train/val/test (80/20 or 70/15/15)
3) Preprocessor + FeatureBuilder -> X_train, X_val, X_test
4) StandardScaler.fit(X_train) -> transform all three
5) model.fit(X_train_scaled, y_train)
6) val_acc = evaluate(model, X_val_scaled, y_val)
7) [optional] GridSearchCV on X_train only (C, gamma)
8) test_acc = evaluate(model, X_test_scaled, y_test)
9) save_bundle(model, scaler, label_map, feature_config, summary)""")
story += SS("8.2 SVM Hyperparameter Selection")
story += [P("RBF SVM has two key hyperparameters: C (regularization strength) and gamma (RBF kernel width). Grid search on training set only (not touching val/test):"),
          C("""from sklearn.model_selection import GridSearchCV, StratifiedKFold
param_grid = {"C": [0.1, 1, 10, 100],
              "gamma": ["scale", "auto", 0.01, 0.1]}
grid = GridSearchCV(SVC(kernel="rbf", probability=True),
                    param_grid, cv=StratifiedKFold(3),
                    scoring="accuracy", n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_C = grid.best_params_["C"]       # typically 10
best_gamma = grid.best_params_["gamma"] # typically "scale" """)]
story += SS("8.3 CLI Training Commands")
story += C("""# Train SVM (recommended HOG+HSV)
python -m traffic_sign_system.scripts.train --model svm --mode hog+hsv --C 10

# Grid search SVM hyperparams
python -m traffic_sign_system.scripts.train --model svm --mode hog+hsv --grid

# Train KNN / Random Forest
python -m traffic_sign_system.scripts.train --model knn --mode hog+hsv
python -m traffic_sign_system.scripts.train --model rf --mode hog+hsv""")

print("Sections 5-8 built OK")


# 9. Evaluation
story += S("9. Model Evaluation and Results Analysis")
story += [P("The evaluation module outputs Accuracy, macro/weighted average Precision/Recall/F1, normalized confusion matrix, and error sample CSV. All metrics are computed on the test set (12,630 samples) to ensure objectivity.")]
story += SS("9.1 Evaluation Metrics")
story += T(["Metric","Formula/Meaning","Usage"],
    [["Accuracy","Correct / Total","Overall recognition capability"],
     ["Precision (Macro)","Arithmetic mean of per-class Precision","Overall false-positive assessment"],
     ["Recall (Macro)","Arithmetic mean of per-class Recall","Missed-detection assessment (unaffected by class imbalance)"],
     ["F1 (Macro)","2*P*R/(P+R) macro average","Balanced score under class imbalance"],
     ["Confusion Matrix","True class x Predicted class cross-tabulation","Locate specific confusion pairs (e.g., 30<->50 km/h)"]],
    [3.2*cm,6.0*cm,7.2*cm])
story += SS("9.2 Evaluation CLI")
story += C("""python -m traffic_sign_system.scripts.evaluate --bundle models/artifacts/svm_hog+hsv.joblib --data dataset/test/
# Output: metrics.json, confusion_matrix.png, errors.csv""")
story += [R("Expected GTSRB 43-class SVM(HOG+HSV) test accuracy is approximately 94%-97% (depending on preprocessing parameters and data split). HOG-only is ~92%-95%, HSV-only ~55%-70%, fusion typically improves by 1-3 percentage points.", "Cal")]

# 10. Multi-Model Comparison
story += S("10. SVM vs KNN vs RF Multi-Model Comparison")
story += [P("Comparison of three classifiers on identical features (HOG+HSV, 1828 dims) and data split (random_state=42). Metrics include accuracy, F1, training time, prediction time, and model size.")]
story += T(["Model","Val Acc","Test Acc","Train Time","Predict Time","Size","Characteristics"],
    [["SVM (RBF)","~96%","~95%","2-5 min","seconds","~10-50 MB","Best on high-dim HOG features"],
     ["KNN (k=5)","~93%","~92%","Instant (lazy)","Slow (search)","~200 MB+","Must store all training features for prediction"],
     ["RandomForest","~91%","~90%","1-3 min","milliseconds","~50-200 MB","Provides feature importance for analysis"]],
    [2.4*cm,2.0*cm,2.0*cm,2.2*cm,2.2*cm,2.2*cm,3.4*cm])
story += [P("Comparison CLI:"),
          C("""python -m traffic_sign_system.scripts.compare --features models/artifacts/features_hog+hsv.npz
# Output: comparison.csv, comparison.png, confusion_matrices.png""")]

# 11. Error Analysis
story += S("11. Error Samples and Confusable Class Analysis")
story += [P("error_analysis.py counts (true, pred) pair frequencies, outputs Top-10 confusion pairs and per-class Recall. Direction matters: A->B and B->A are distinct error patterns.")]
story += SS("11.1 Common Confusion Patterns")
story += T(["Confusion Pair","Likely Cause"],
    [["30 km/h <-> 50 km/h","Similar round shape, only digit differs; HOG digit detail resolution limited"],
     ["Yield -> General caution","Both triangular red-border signs, similar contour"],
     ["No overtaking -> End no overtaking","Identical shape, only internal diagonal line differs; gray HOG misses color"],
     ["Left turn -> Ahead only","Arrow direction differences less significant at 64x64 resolution"],
     ["80 km/h -> 60 km/h","Digit glyph similarity (8 vs 6), especially at low resolution"]],
    [5.6*cm,10.8*cm])
story += SS("11.2 Improvement Directions")
story += [P("Based on error analysis, the following improvements can be made:")]
story += B([
    "Fuse HSV color features: red prohibitory vs blue mandatory vs white speed limit can be distinguished by color histogram",
    "Increase input size: 64->96 or 128 preserves more digit/symbol detail",
    "Class balancing: oversample minority classes (SMOTE) or adjust class weights (class_weight='balanced')",
    "Feature enhancement: try LBP (Local Binary Patterns) or SIFT bag-of-words as supplementary features",
])
story += [R("Error analysis CLI:", "Bd")]
story += C("""python -m traffic_sign_system.scripts.error_stats --bundle models/artifacts/svm_hog+hsv.joblib --data dataset/test/
# Output: top_confusions.csv/png, errors_per_class.csv/png""")

# 12. GUI
story += S("12. PyQt5 Graphical User Interface Design")
story += [P("MainWindow is based on QMainWindow with a left-canvas + right-control-panel layout. Supports three tabs (Image/Video/Camera), avoids UI freezing via QThread background threads, and drives video display via QTimer(33ms).")]
story += SS("12.1 Main Window Layout")
story += C("""+--------------------------------------------------------+
| Menu: File(F) | Model(M) | Video/Camera(V) | Help(H)   |
+--------------------------------------+-----------------+
| + Image --+-- Video --+-- Camera --+ | [Control Panel] |
| | +--------+ +--------+ +------+   | | Model Path:     |
| | |Original| | Video  | |Camera|   | | [____] [Browse] |
| | | Image  | | Stream | |Feed  |   | | [Load Model]    |
| | +--------+ |        | |      |   | |                 |
| | +--------+ |        | |      |   | | [Image Mode]    |
| | |Preproc.| |        | |      |   | | [Select Image]  |
| | |(Gray)  | |        | |      |   | | [Recognize]     |
| | +--------+ +--------+ |      |   | | [Scene Detect]  |
| | +--------+            |      |   | | [Clear Image]   |
| | |Result  | [Progress] |      |   | |                 |
| | |Overlay |            |      |   | | [Video/Camera]  |
| | +--------+            +------+   | | [Pause/Stop]    |
| +----------------------------------+ | [Save Result]   |
|                                     |                 |
|                                     | [Result]        |
|                                     | ID: --          |
|                                     | Name: --        |
|                                     | Confidence: --  |
+-------------------------------------+-----------------+
| Status: Ready | Time: -- ms | Model: SVM | Feature:... |
+--------------------------------------------------------+""")
story += SS("12.2 Feature Table")
story += T(["Feature","Operation","Implementation"],
    [["Image Recognition","Select image -> Recognize","PredictWorker(QThread) background inference"],
     ["Video Recognition","Select video -> Auto play","QTimer(33ms) per-frame VideoRecognizer"],
     ["Camera Recognition","Select camera index -> Open","CameraRecognizer center-ROI fixed-box classification"],
     ["Scene Detection","Click Scene Detect","DetectWorker(QThread) HSV+contour+classification"],
     ["Pause/Resume","Click Pause button","Stop/Restart QTimer"],
     ["Save Result","Click Save Result","cv2.VideoWriter writes .mp4"]],
    [2.4*cm,4.2*cm,9.8*cm])

# 13. SignDetector
story += S("13. Traditional Vision Detection Extension (SignDetector)")
story += [P("SignDetector is a demonstration extension for locating candidate traffic sign regions in full road scenes. It uses a classic pipeline of HSV color segmentation + morphological processing + contour geometric filtering, then passes filtered candidate ROIs to the trained classifier."),
          R("!! This method has limited stability under complex lighting, occlusion, motion blur, or partial sign visibility. It is intended to demonstrate traditional computer vision processing pipelines and is NOT suitable for safety-critical applications.", "Cal")]
story += SS("13.1 Detection Pipeline")
story += C("""1) HSV Color Masking:
   Red (two-segment union): (0,80,80)-(10,255,255) U (170,80,80)-(180,255,255)
   Blue:                    (90,80,80)-(140,255,255)

2) Morphological Processing: Elliptical kernel (5x5) opening -> closing

3) Contour Filtering (all thresholds configurable):
   min_area = 400 px^2       Minimum pixel area
   max_area_ratio = 0.3      Upper bound as fraction of total image
   min_aspect = 0.5          Minimum width/height ratio
   max_aspect = 2.0          Maximum width/height ratio
   min_circularity = 0.4     Minimum circularity = 4*pi*area / (perimeter^2+1e-6)

4) Crop ROI -> Predictor.predict()

5) Filter results with confidence < 0.5""")
story += SS("13.2 Design Rationale")
story += [P("The detection strategy follows 'prefer missed detections over false positives': high circularity threshold filters non-circular noise, area and aspect ratio exclude elongated/oversized regions, low-confidence results are discarded. Some signs may be missed, but false detection rate is kept low."),
          P('In the GUI, click the "Scene Detection" button to run SignDetector on the current image or video frame. Results are annotated on the display with colored bounding boxes (red/blue) + class name + confidence.')]

# 14. Improvements
story += S("14. Improvement Directions and Outlook")
story += SS("14.1 Feature Level")
story += B([
    "Introduce LBP (Local Binary Patterns) texture features to better describe internal sign symbols",
    "Try multi-scale HOG or pyramid HOG to improve adaptability to different sign sizes",
    "Extend color features from H/S 2D histogram to include V (brightness) in a 3D histogram",
    "Use PCA or LDA to reduce high-dimensional HOG features (1764 dims), reducing redundancy and speeding up training",
])
story += SS("14.2 Model Level")
story += B([
    "Try LinearSVC for faster training with large sample counts (GTSRB 39k samples feasible)",
    "Introduce XGBoost/LightGBM gradient boosting trees as comparison models",
    "Use Stacking ensemble to combine SVM + KNN + RF predictions",
    "Explore CNN approaches (e.g., lightweight MobileNet/ResNet18) for quantitative comparison with traditional methods",
])
story += SS("14.3 System Level")
story += B([
    "Replace SignDetector HSV localization with CNN-based object detector (e.g., YOLO/SSD)",
    "Add batch testing mode: auto-run on entire test set and aggregate results",
    "Support model hot-swapping: switch between different bundles without restarting GUI",
    "Output PDF report: one-click generation of comprehensive experiment report with all evaluation charts",
])

# 15. Limitations & Conclusion
story += S("15. Limitations and Conclusion")
story += SS("15.1 Current Limitations")
story += T(["Limitation","Description"],
    [["Fixed feature dimension","HOG window 64x64 requires input to strictly be this size; no multi-scale support"],
     ["Simple color feature","H/S 2D 8x8 histogram may be insufficient to distinguish similarly-colored signs"],
     ["Low detector robustness","Pure HSV color segmentation adapts poorly to varying lighting, shadows, and occlusions"],
     ["No GPU acceleration","SVM training runs on CPU; CNN deep models not yet integrated"],
     ["Class imbalance","GTSRB class sample counts vary widely (hundreds to 2000+), potentially affecting minority class recall"],
     ["No real-time video detection","SignDetector HSV+contour processing is too slow for real-time video detection"]],
    [3.2*cm,13.2*cm])
story += SS("15.2 Conclusion")
story += [P("This project built a complete traffic sign classification and recognition system based on OpenCV + HOG + SVM, covering all stages from dataset management, image preprocessing, feature engineering, model training and comparison, model evaluation and error analysis, to PyQt5 GUI interaction and traditional vision detection extension. The system uses modular design with decoupled, configuration-driven, and highly reproducible components."),
          P("Three technical pipelines - HOG-only, HSV color histogram-only, and HOG+HSV fusion - provide a clear comparison framework for feature engineering experiments. The unified training and comparison mechanism for SVM/KNN/RF classifiers provides evidence-based model selection. The PyQt5 GUI supports image, video, and camera recognition modes, plus the traditional HSV detection extension, forming a complete demonstration system."),
          P("This system is suitable as a complete computer vision course design solution, covering core knowledge areas including image processing, feature extraction, machine learning classification, model evaluation, and desktop GUI development, with both theoretical depth and practical engineering value.")]

story += [Spacer(1,1*cm),
          Paragraph("--- END ---", ParagraphStyle("End", fontName="SimHei", fontSize=12, leading=18, alignment=TA_CENTER, textColor=GRAY))]

# =========================== BUILD PDF ===========================
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    rightMargin=1.55*cm, leftMargin=1.55*cm,
    topMargin=1.75*cm, bottomMargin=1.65*cm,
    title="Traffic Sign Classification and Recognition System - Course Design Report",
    author="Course Design",
)
doc.build(story, onFirstPage=HF, onLaterPages=HF)
print(f"PDF generated successfully: {OUT}")
