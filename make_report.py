"""
Zoidberg 2.0 — PDF Synthesis Report
Epitech MSc IT — Machine Learning Project
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, KeepTogether, Image,
                                 PageBreak)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas as pdfcanvas
import os

OUT = "outputs/synthesis_report.pdf"
W, H = A4

# ── Colours ────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0F1F3D")
TEAL   = colors.HexColor("#0D9488")
LTEAL  = colors.HexColor("#14B8A6")
LGRAY  = colors.HexColor("#F1F5F9")
MGRAY  = colors.HexColor("#64748B")
RED    = colors.HexColor("#EF4444")
GREEN  = colors.HexColor("#22C55E")
WHITE  = colors.white
DARK   = colors.HexColor("#1E293B")

# ── Styles ─────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

Title    = S('ZTitle',    fontSize=26, fontName='Helvetica-Bold',
             textColor=WHITE,   alignment=TA_LEFT,    leading=32)
SubTitle = S('ZSubTitle', fontSize=13, fontName='Helvetica',
             textColor=LTEAL,   alignment=TA_LEFT,    leading=18)
H1       = S('ZH1',       fontSize=14, fontName='Helvetica-Bold',
             textColor=NAVY,    spaceBefore=14, spaceAfter=6,  leading=18)
H2       = S('ZH2',       fontSize=11, fontName='Helvetica-Bold',
             textColor=TEAL,    spaceBefore=8,  spaceAfter=4,  leading=14)
Body     = S('ZBody',     fontSize=10, fontName='Helvetica',
             textColor=DARK,    alignment=TA_JUSTIFY, leading=14, spaceAfter=6)
Bullet   = S('ZBullet',   fontSize=10, fontName='Helvetica',
             textColor=DARK,    leftIndent=16,  leading=14, spaceAfter=3)
Code     = S('ZCode',     fontSize=8.5,fontName='Courier',
             textColor=colors.HexColor("#1E293B"),
             backColor=LGRAY,   leading=12, spaceAfter=4,
             leftIndent=10, rightIndent=10)
Caption  = S('ZCaption',  fontSize=8.5,fontName='Helvetica-Oblique',
             textColor=MGRAY,   alignment=TA_CENTER, spaceAfter=8)
Small    = S('ZSmall',    fontSize=8,  fontName='Helvetica',
             textColor=MGRAY,   alignment=TA_LEFT)

def b(text): return f'<b>{text}</b>'
def teal(text): return f'<font color="#0D9488"><b>{text}</b></font>'
def bullet(text): return Paragraph(f"• {text}", Bullet)

# ── Header/Footer callback ─────────────────────────────────────────────
def header_footer(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, H - 1.2*cm, W, 1.2*cm, fill=1, stroke=0)
    canvas_obj.setFillColor(TEAL)
    canvas_obj.rect(0, H - 1.25*cm, W, 0.12*cm, fill=1, stroke=0)
    canvas_obj.setFillColor(WHITE)
    canvas_obj.setFont('Helvetica-Bold', 9)
    canvas_obj.drawString(1.5*cm, H - 0.85*cm, "ZOIDBERG 2.0  —  Pneumonia Detection from Chest X-Rays")
    canvas_obj.setFont('Helvetica', 8)
    canvas_obj.setFillColor(LTEAL)
    canvas_obj.drawRightString(W - 1.5*cm, H - 0.85*cm, "Epitech MSc IT  •  Machine Learning")

    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, 0, W, 0.9*cm, fill=1, stroke=0)
    canvas_obj.setFillColor(WHITE)
    canvas_obj.setFont('Helvetica', 8)
    canvas_obj.drawString(1.5*cm, 0.32*cm, "Synthesis Report  |  Binary & 3-Class Pneumonia Classification")
    canvas_obj.drawRightString(W - 1.5*cm, 0.32*cm, f"Page {doc.page}")
    canvas_obj.restoreState()

def section_bar(title):
    data = [[Paragraph(title, S('sh', fontSize=12, fontName='Helvetica-Bold',
                                textColor=WHITE, leading=16))]]
    t = Table(data, colWidths=[W - 4*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NAVY),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('LINEBELOW', (0,0), (-1,-1), 2, TEAL),
    ]))
    return t

def metric_table(rows, col_widths=None):
    header  = rows[0]
    data    = rows[1:]
    cw = col_widths or [3.5*cm] + [(W - 7*cm) / (len(header)-1)] * (len(header)-1)

    t_data = [
        [Paragraph(h, S('th', fontSize=9, fontName='Helvetica-Bold',
                        textColor=WHITE, alignment=TA_CENTER, leading=11))
         for h in header]
    ]
    for i, row in enumerate(data):
        t_data.append([
            Paragraph(str(c), S('td', fontSize=9, fontName='Helvetica',
                                textColor=DARK, alignment=TA_CENTER, leading=12))
            for c in row
        ])

    t = Table(t_data, colWidths=cw, repeatRows=1)
    style = [
        ('BACKGROUND',    (0,0), (-1,0),  NAVY),
        ('BACKGROUND',    (0,1), (-1,-1), LGRAY),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LGRAY]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor("#E2E8F0")),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
        ('RIGHTPADDING',  (0,0), (-1,-1), 6),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]
    t.setStyle(TableStyle(style))
    return t

# ── Build document ─────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=1.8*cm,
)

story = []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COVER PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cover_data = [[
    Paragraph("ZOIDBERG 2.0", Title),
    Spacer(1, 0.3*cm),
    Paragraph("Pneumonia Detection from Chest X-Ray Images", SubTitle),
    Spacer(1, 0.5*cm),
    Paragraph("Synthesis Report", S('sr', fontSize=11, fontName='Helvetica',
                                    textColor=colors.HexColor("#94A3B8"),
                                    leading=16)),
    Spacer(1, 0.4*cm),
    Paragraph("Epitech MSc IT  —  Machine Learning Project", S('sr2', fontSize=10,
              fontName='Helvetica', textColor=LTEAL, leading=14)),
    Spacer(1, 0.3*cm),
    Paragraph("May 2026", S('dt', fontSize=9, fontName='Helvetica',
                             textColor=colors.HexColor("#94A3B8"), leading=12)),
]]
cover_tbl = Table(cover_data, colWidths=[W - 4*cm])
cover_tbl.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,-1), NAVY),
    ('TOPPADDING',    (0,0), (-1,-1), 28),
    ('BOTTOMPADDING', (0,0), (-1,-1), 28),
    ('LEFTPADDING',   (0,0), (-1,-1), 20),
    ('RIGHTPADDING',  (0,0), (-1,-1), 20),
    ('LINEAFTER',     (0,0), (-1,-1), 4, TEAL),
]))
story.append(cover_tbl)
story.append(Spacer(1, 0.8*cm))

stats = [
    ("5,856", "Total X-Ray\nImages"),
    ("6", "ML Models\nTrained"),
    ("87.9%", "PCA Variance\nRetained"),
    ("97.1%", "Best Accuracy\n(Split 1)"),
    ("0.995", "Best ROC-AUC\n(SVM Split 1)"),
]
stat_data = [[Paragraph(f"{b(v)}", S('sv', fontSize=18, fontName='Helvetica-Bold',
                                      textColor=TEAL, alignment=TA_CENTER, leading=22))
              for v,_ in stats],
             [Paragraph(l, S('sl', fontSize=7.5, fontName='Helvetica',
                              textColor=MGRAY, alignment=TA_CENTER, leading=10))
              for _,l in stats]]
stat_tbl = Table(stat_data, colWidths=[(W-4*cm)/5]*5)
stat_tbl.setStyle(TableStyle([
    ('BACKGROUND',    (0,0),(-1,-1), LGRAY),
    ('BOX',           (0,0),(-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ('LINEBELOW',     (0,0),(-1,0), 0.5, colors.HexColor("#E2E8F0")),
    ('TOPPADDING',    (0,0),(-1,-1), 8),
    ('BOTTOMPADDING', (0,0),(-1,-1), 8),
]))
story.append(stat_tbl)
story.append(Spacer(1, 1*cm))

# ── Abstract ────────────────────────────────────────────────────────────
story.append(section_bar("Abstract"))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "For this project we built a system that looks at chest X-ray images and tries to tell "
    "if a patient has pneumonia or not. We tested six different classifiers — Logistic Regression, "
    "Random Forest, MLP, SVM, KNN, and a CNN — and compared them across three different ways "
    "of splitting the data. The hardest part turned out to be the class imbalance: about 74% of "
    "the images are pneumonia cases, so every model we trained at first just predicted pneumonia "
    "for everything and got 0% on normal cases. Fixing that took more time than we expected. "
    "We also did a 3-class version at the end to separate bacterial vs viral pneumonia.", Body))
story.append(PageBreak())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PROJECT OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.append(section_bar("1.  Project Overview"))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(b("Goal"), H2))
story.append(Paragraph(
    "Given a chest X-ray image, predict whether the lungs look <b>NORMAL</b> (label 0) "
    "or show signs of <b>PNEUMONIA</b> (label 1). The dataset comes from Kaggle "
    "(Chest X-Ray Pneumonia) and is split across three separate folders.", Body))

story.append(Paragraph(b("Dataset Breakdown"), H2))
ds_rows = [
    ["Dataset", "Role", "Normal", "Pneumonia", "Total"],
    ["Dataset 1 (Train)", "Training", "1,341", "3,875", "5,216"],
    ["Dataset 2 (Val)",   "Validation / Tuning", "8", "8", "16"],
    ["Dataset 3 (Test)",  "Final Evaluation", "234", "390", "624"],
    ["TOTAL", "—", "1,583", "4,273", "5,856"],
]
story.append(metric_table(ds_rows, [3.8*cm, 4.5*cm, 2.2*cm, 2.5*cm, 2*cm]))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "The imbalance is pretty obvious — roughly 3 pneumonia images for every 1 normal one. "
    "This ended up being the main problem throughout the whole project. Models without any "
    "correction just learned to always say pneumonia and still got 74% accuracy, which looks "
    "fine on paper but is completely wrong in practice.", Body))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PREPROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.append(Spacer(1, 0.5*cm))
story.append(section_bar("2.  Preprocessing Pipeline"))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Every image goes through the same steps before reaching the model. "
    "The CNN is a slight exception since it works on 2D pixel grids directly, "
    "but the grayscale and resize steps still apply.", Body))

steps = [
    ("Grayscale (gray filter)",
     "We load each image and convert it to grayscale using PIL's .convert('L'). "
     "X-rays don't carry colour information so keeping RGB would just add noise."),
    ("Resize to 128x128",
     "Images come from different scanners at different resolutions. "
     "We resize everything to 128x128 so the feature vector length stays consistent."),
    ("Flatten to 1D + Normalise",
     "128x128 = 16,384 pixels. We call .flatten() to turn the 2D image into a 1D array, "
     "then divide by 255.0 to bring values into [0.0, 1.0]."),
    ("StandardScaler",
     "Fit only on training data. Shifts each of the 16,384 features to mean=0, std=1. "
     "Without this, features with large pixel ranges would dominate the distance calculations."),
    ("PCA — 100 components",
     "Drops from 16,384 features down to 100. We keep 87.88% of the variance. "
     "This made SVM and KNN actually trainable — without PCA they were running for hours."),
]
for i, (name, desc) in enumerate(steps):
    row = [[
        Paragraph(f"{i+1}", S('num', fontSize=14, fontName='Helvetica-Bold',
                               textColor=WHITE, alignment=TA_CENTER)),
        Paragraph(f"<b>{name}</b><br/>{desc}",
                  S('sd', fontSize=9.5, fontName='Helvetica', textColor=DARK,
                    leading=13))
    ]]
    t = Table(row, colWidths=[0.8*cm, W-4.8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(0,0), TEAL),
        ('BACKGROUND',    (1,0),(1,0), LGRAY if i%2==0 else WHITE),
        ('TOPPADDING',    (0,0),(-1,-1), 7),
        ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ('LEFTPADDING',   (0,0),(0,0), 5),
        ('LEFTPADDING',   (1,0),(1,0), 10),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('BOX',           (0,0),(-1,-1), 0.3, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t)

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    "After preprocessing: X_train shape (5,216 x 100)  |  X_val (16 x 100)  |  X_test (624 x 100)", Body))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. DATA SPLITS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.append(PageBreak())
story.append(section_bar("3.  Data Splitting Strategies"))
story.append(Spacer(1, 0.3*cm))

splits_info = [
    ("Split 1 — Simple Train/Test",
     "We take 80% of Dataset 1 for training (4,172 images) and leave 20% for testing "
     "(1,044 images). Stratified so the class ratio stays the same in both halves. "
     "Good for a quick check but the numbers tend to look better than reality.",
     "Since both halves come from the same dataset, there's some risk the split just got lucky. "
     "Scores on Split 1 are generally the highest — don't read too much into them."),
    ("Split 2 — Train / Validation / Test",
     "Dataset 1 trains the model, Dataset 2 (just 16 images) is used for hyperparameter tuning, "
     "and Dataset 3 (624 images) is the final held-out test. This is the most honest setup "
     "since the test images come from a completely separate source.",
     "The validation set being only 16 images is a real problem. GridSearchCV on 16 images "
     "isn't very reliable, so the tuning step is a bit shaky."),
    ("Split 3 — 5-Fold Stratified CV",
     "Dataset 1 gets split into 5 equal folds. Each time, 4 folds train and 1 fold tests, "
     "rotating through all 5. The reported score is the average. This gives a much more "
     "stable picture of how the model actually performs.",
     "Takes longer to run but the results are the most trustworthy. We used this as the "
     "main comparison point between models."),
]

for title, desc, note in splits_info:
    story.append(Paragraph(b(title), H2))
    story.append(Paragraph(desc, Body))
    story.append(Paragraph(f"<i>{note}</i>",
                           S('note', fontSize=9, fontName='Helvetica-Oblique',
                             textColor=MGRAY, leading=12, spaceAfter=8)))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. MODEL RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.append(Spacer(1, 0.5*cm))
story.append(section_bar("4.  Model Results"))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(b("4.1  Logistic Regression"), H2))
story.append(Paragraph(
    "We started with logistic regression as a baseline. It's a linear model so we didn't "
    "expect miracles, but it actually did well on Split 1 and Split 3. Split 2 drops "
    "noticeably — probably because Dataset 3 comes from a different source than what the "
    "model trained on.", Body))
lr_rows = [
    ["Split", "Accuracy", "F1", "ROC-AUC"],
    ["Split 1 (Train/Test)", "0.9598", "0.9728", "0.9891"],
    ["Split 2 (Val/Test)",   "0.7484", "0.8310", "0.8973"],
    ["Split 3 (5-Fold CV)",  "0.9559", "0.9704", "0.9871"],
]
story.append(metric_table(lr_rows, [5*cm, 3*cm, 3*cm, 4*cm]))
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph(b("4.2  Random Forest"), H2))
story.append(Paragraph(
    "Random forest with 100 trees. We expected this to beat logistic regression but it "
    "actually came in slightly lower on most splits. On PCA-reduced data the trees tend to "
    "be less effective — the principal components don't have the same interpretability as "
    "raw features. Still solid results overall.", Body))
rf_rows = [
    ["Split", "Accuracy", "F1", "ROC-AUC"],
    ["Split 1 (Train/Test)", "0.9416", "0.9617", "0.9858"],
    ["Split 2 (Val/Test)",   "0.7436", "0.8287", "0.9160"],
    ["Split 3 (5-Fold CV)",  "0.9329", "0.9562", "0.9833"],
]
story.append(metric_table(rf_rows, [5*cm, 3*cm, 3*cm, 4*cm]))
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph(b("4.3  MLP Neural Network"), H2))
story.append(Paragraph(
    "MLP with two hidden layers (256 then 128 nodes). This was the first model where we "
    "saw real improvement on the normal class after adding class weights and rotating some "
    "normal images to balance the training set. The best sklearn model we got.", Body))
mlp_rows = [
    ["Split", "Accuracy", "F1", "ROC-AUC"],
    ["Split 1 (Train/Test)", "0.9617", "0.9741", "0.9937"],
    ["Split 2 (Val/Test)",   "0.7837", "0.8512", "0.9032"],
    ["Split 3 (5-Fold CV)",  "0.9680", "0.9785", "0.9928"],
]
story.append(metric_table(mlp_rows, [5*cm, 3*cm, 3*cm, 4*cm]))
story.append(PageBreak())

story.append(Paragraph(b("4.4  SVM — Support Vector Machine"), H2))
story.append(Paragraph(
    "SVM with RBF kernel gave us the most trouble to get right. The first version predicted "
    "100% pneumonia — literally 0% normal recall. The fix was changing the GridSearchCV "
    "scoring from f1 to <b>balanced_accuracy</b> and setting gamma to 'scale' instead of "
    "a fixed small value. After that it became our best model on Split 1.", Body))
svm_rows = [
    ["Split", "Accuracy", "F1", "ROC-AUC", "Normal %"],
    ["Split 1 (Train/Test)", "0.9713", "0.9805", "0.9952", "97.01%"],
    ["Split 2 (Val/Test)",   "0.7949", "0.8562", "0.9161", "49.15%"],
    ["Split 3 (5-Fold CV)",  "0.9682", "0.9784", "0.9952", "~96%"],
]
story.append(metric_table(svm_rows, [4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm]))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "The problem with f1 scoring is that it only looks at the pneumonia class (the positive "
    "class). A model that says pneumonia for everything gets a high f1 score. Balanced accuracy "
    "forces the model to do well on both classes, which is what we actually want.", Body))
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph(b("4.5  KNN — K-Nearest Neighbours"), H2))
story.append(Paragraph(
    "KNN works by finding the K closest training points in PCA space and going with majority "
    "vote. We swept K from 1 to 20 and picked the best one (K=9). It's surprisingly "
    "competitive — simpler than SVM but gets close on most splits. Normal class recall "
    "was also decent compared to some of the other models.", Body))
knn_rows = [
    ["Split", "Accuracy", "F1", "ROC-AUC", "Normal %"],
    ["Split 1 (K=9)", "0.9617", "0.9745", "0.9809", "89.93%"],
    ["Split 2 (K=9)", "0.7676", "0.8419", "0.8741", "~60%"],
    ["Split 3 (K=9)", "0.9515", "0.9677", "0.9838", "~88%"],
]
story.append(metric_table(knn_rows, [4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm]))
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph(b("4.6  CNN — Convolutional Neural Network"), H2))
story.append(Paragraph(
    "The CNN was the most involved to set up. Three convolutional blocks with batch "
    "normalisation and dropout, trained directly on 64x64 grayscale images (no PCA). "
    "We added class weights to stop it ignoring normal images, and tuned the prediction "
    "threshold to 0.7 to push normal recall up a bit more.", Body))
cnn_rows = [
    ["Configuration", "Accuracy", "F1", "ROC-AUC", "Normal %", "Pneumonia %"],
    ["Split 2, thresh=0.5", "0.8542", "0.8936", "0.9499", "64.53%", "97.95%"],
    ["Split 2, thresh=0.7", "0.8574", "0.8947", "0.9499", "67.09%", "96.92%"],
]
story.append(metric_table(cnn_rows, [4.5*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.7*cm]))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "<b>Architecture:</b>  Input(64x64x1) → Conv2D(32) → BN → MaxPool → Dropout(0.25) "
    "→ Conv2D(64) → BN → MaxPool → Dropout(0.25) → Conv2D(128) → BN → MaxPool "
    "→ Dropout(0.4) → Dense(256) → Dropout(0.5) → Dense(1, sigmoid)", Code))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. FINAL COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.append(PageBreak())
story.append(section_bar("5.  Final Model Comparison"))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Split 2 is the one that matters most — it uses a completely separate test set "
    "that the models never saw during training. Split 1 and Split 3 scores are "
    "higher but they're less honest since the data comes from the same source as training.", Body))

all_rows = [
    ["Model", "Split", "Accuracy", "F1", "ROC-AUC"],
    ["Logistic Regression", "Split 1", "0.9598", "0.9728", "0.9891"],
    ["Logistic Regression", "Split 2", "0.7484", "0.8310", "0.8973"],
    ["Logistic Regression", "Split 3 CV", "0.9559", "0.9704", "0.9871"],
    ["Random Forest",        "Split 1", "0.9416", "0.9617", "0.9858"],
    ["Random Forest",        "Split 2", "0.7436", "0.8287", "0.9160"],
    ["Random Forest",        "Split 3 CV", "0.9329", "0.9562", "0.9833"],
    ["MLP Neural Net",       "Split 1", "0.9617", "0.9741", "0.9937"],
    ["MLP Neural Net",       "Split 2", "0.7837", "0.8512", "0.9032"],
    ["MLP Neural Net",       "Split 3 CV", "0.9680", "0.9785", "0.9928"],
    ["SVM (RBF, balanced)",  "Split 1", "0.9713", "0.9805", "0.9952"],
    ["SVM (RBF, balanced)",  "Split 2", "0.7949", "0.8562", "0.9161"],
    ["SVM (RBF, balanced)",  "Split 3 CV", "0.9682", "0.9784", "0.9952"],
    ["KNN (K=9)",            "Split 1", "0.9617", "0.9745", "0.9809"],
    ["KNN (K=9)",            "Split 2", "0.7676", "0.8419", "0.8741"],
    ["KNN (K=9)",            "Split 3 CV", "0.9515", "0.9677", "0.9838"],
    ["CNN (thresh=0.7)",     "Split 2", "0.8574", "0.8947", "0.9499"],
]
story.append(metric_table(all_rows, [4*cm, 3.2*cm, 2.5*cm, 2.5*cm, 2.8*cm]))
story.append(Spacer(1, 0.4*cm))

winner_data = [[Paragraph(
    "Best Overall — SVM (Split 1):  Accuracy 0.9713  |  F1 0.9805  |  ROC-AUC 0.9952  "
    "|  Normal 97.01%  |  Pneumonia 97.16%",
    S('win', fontSize=10, fontName='Helvetica-Bold', textColor=WHITE,
      alignment=TA_CENTER, leading=14))]]
win_tbl = Table(winner_data, colWidths=[W - 4*cm])
win_tbl.setStyle(TableStyle([
    ('BACKGROUND',    (0,0),(-1,-1), NAVY),
    ('TOPPADDING',    (0,0),(-1,-1), 10),
    ('BOTTOMPADDING', (0,0),(-1,-1), 10),
    ('LINEBELOW',     (0,0),(-1,-1), 3, TEAL),
]))
story.append(win_tbl)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. DIFFICULTIES & SOLUTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.append(PageBreak())
story.append(section_bar("6.  What Went Wrong and How We Fixed It"))
story.append(Spacer(1, 0.3*cm))

difficulties = [
    ("Normal class accuracy stuck at 0%",
     "The dataset has 1,341 normal images vs 3,875 pneumonia. Every model we trained "
     "initially just predicted pneumonia for everything. Accuracy looked decent (74%) but "
     "normal recall was literally 0%. Took us a while to even realise the problem.",
     "We rotated normal training images 4 times to get more copies (from 1,341 to ~6,700). "
     "Added class_weight='balanced' to SVM, KNN, and CNN. For SVM specifically, changed "
     "GridSearchCV scoring to balanced_accuracy. Threshold tuned to 0.7 on CNN. "
     "After all this, normal accuracy on SVM reached 97%."),
    ("Raw images too big for SVM and KNN",
     "Each 128x128 image flattened is 16,384 features. Training SVM on 5,000 images with "
     "16k features is very slow — we left it running and it was still going after 30 minutes.",
     "PCA down to 100 components. Keeps 87.88% of the variance but cuts training time "
     "to seconds. The 99.4% feature reduction made both SVM and KNN completely usable."),
    ("SVM GridSearchCV chose a broken model",
     "When we used scoring='f1' in GridSearchCV, the best model it found had 0% normal "
     "recall. The issue is f1 only measures pneumonia (the positive class) — so predicting "
     "everything as pneumonia still scores well. The grid also picked gamma=0.001 which "
     "made the RBF kernel almost flat.",
     "Changed scoring to balanced_accuracy, which cares about both classes equally. "
     "Also switched gamma to 'scale' so sklearn auto-adjusts it based on the data. "
     "Normal accuracy went from 0% to 97% just from this change."),
    ("Validation set too small to be useful",
     "Dataset 2 only has 16 images — 8 normal, 8 pneumonia. Running GridSearchCV on 16 "
     "samples gives basically random results. One bad prediction swings the score by 12.5%.",
     "We kept Dataset 2 only for the final held-out test, not for tuning. "
     "For actual hyperparameter decisions we relied on Split 3 (5-fold CV on Dataset 1) "
     "which is much more stable."),
    ("TensorFlow not compatible with Python 3.14",
     "The project virtual environment runs Python 3.14 but tensorflow-macos has no "
     "wheel for Python 3.14. Every install attempt just failed silently or errored.",
     "Installed TensorFlow on the system Python 3.11 instead "
     "(/opt/homebrew/bin/python3.11). Also had to remove tensorflow-metal because it "
     "kept throwing a 'platform already registered: METAL' conflict at startup."),
]

for prob_title, problem, solution in difficulties:
    row = [[
        Paragraph(f"<b>{prob_title}</b>", S('pt', fontSize=10, fontName='Helvetica-Bold',
                                             textColor=NAVY, leading=14)),
    ],[
        Table([[
            Paragraph("<b>What happened</b>", S('pl', fontSize=9, fontName='Helvetica-Bold',
                                          textColor=RED, leading=12)),
            Paragraph(problem, S('pb', fontSize=9, fontName='Helvetica',
                                  textColor=DARK, leading=12))
        ],[
            Paragraph("<b>What we did</b>", S('sl', fontSize=9, fontName='Helvetica-Bold',
                                           textColor=GREEN, leading=12)),
            Paragraph(solution, S('sb', fontSize=9, fontName='Helvetica',
                                   textColor=DARK, leading=12))
        ]], colWidths=[1.8*cm, W - 6*cm],
        style=TableStyle([
            ('TOPPADDING',    (0,0),(-1,-1), 4),
            ('BOTTOMPADDING', (0,0),(-1,-1), 4),
            ('LEFTPADDING',   (0,0),(0,-1), 6),
            ('LEFTPADDING',   (1,0),(1,-1), 8),
            ('VALIGN',        (0,0),(-1,-1), 'TOP'),
            ('LINEABOVE',     (0,1),(-1,1), 0.3, colors.HexColor("#E2E8F0")),
        ]))
    ]]
    outer = Table(row, colWidths=[W-4*cm])
    outer.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0), colors.HexColor("#EFF6FF")),
        ('BACKGROUND',    (0,1),(-1,1), WHITE),
        ('BOX',           (0,0),(-1,-1), 0.5, colors.HexColor("#BFDBFE")),
        ('LINEBELOW',     (0,0),(-1,0), 0.3, colors.HexColor("#BFDBFE")),
        ('TOPPADDING',    (0,0),(-1,0), 7),
        ('BOTTOMPADDING', (0,0),(-1,0), 7),
        ('LEFTPADDING',   (0,0),(-1,-1), 10),
    ]))
    story.append(outer)
    story.append(Spacer(1, 0.3*cm))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. BONUS — 3-CLASS PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.append(PageBreak())
story.append(section_bar("7.  Bonus — 3-Class: Normal / Bacteria / Virus"))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Once the binary version was working we tried going one step further — instead of just "
    "normal vs pneumonia, can we tell the difference between bacterial and viral pneumonia too? "
    "The filenames in the PNEUMONIA folder include either 'bacteria' or 'virus', so we used "
    "that to split them into two separate classes. Normal stays class 0, bacteria is 1, "
    "virus is 2.", Body))

story.append(Paragraph(b("How many images per class"), H2))
dist3_rows = [
    ["Dataset", "NORMAL", "BACTERIA", "VIRUS", "Total"],
    ["Train (Dataset 1)", "1,341", "2,530", "1,345", "5,216"],
    ["Test  (Dataset 3)", "234",   "242",   "148",   "624"],
]
story.append(metric_table(dist3_rows, [4*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.6*cm]))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Virus is the hardest class by far. Only 148 test images, and bacterial vs viral "
    "pneumonia looks very similar on an X-ray — even doctors sometimes struggle. "
    "The models all confused virus with bacteria regularly.", Body))

story.append(Paragraph(b("Results across all models"), H2))
rows3 = [
    ["Model", "Split", "Accuracy", "Macro F1", "Weighted F1"],
    ["SVM (OvR, RBF)", "Split 1", "0.8151", "0.8108", "0.8138"],
    ["SVM (OvR, RBF)", "Split 2 (Official)", "0.6939", "0.6734", "0.6867"],
    ["SVM (OvR, RBF)", "Split 3 CV", "0.7991", "0.7938", "0.7981"],
    ["KNN (K=18)", "Split 1", "0.7835", "0.7656", "0.7746"],
    ["KNN (K=18)", "Split 2 (Official)", "0.7019", "0.6789", "0.6927"],
    ["KNN (K=18)", "Split 3 CV", "0.7697", "0.7469", "0.7596"],
    ["CNN (3-class softmax)", "Split 2 (Official)", "0.7228", "0.7015", "0.7186"],
]
story.append(metric_table(rows3, [3.5*cm, 3.8*cm, 2.5*cm, 2.5*cm, 2.7*cm]))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(b("SVM per class — Split 1 (best result)"), H2))
svm3_detail = [
    ["Class", "Precision", "Recall", "F1-Score", "Support"],
    ["NORMAL",   "0.9664", "0.9628", "0.9646", "269"],
    ["BACTERIA", "0.8107", "0.8379", "0.8241", "506"],
    ["VIRUS",    "0.6640", "0.6245", "0.6437", "269"],
    ["Macro avg","0.8137", "0.8084", "0.8108", "1044"],
]
story.append(metric_table(svm3_detail, [3.5*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm]))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(b("CNN per class — Split 2 (official test)"), H2))
cnn3_detail = [
    ["Class", "Precision", "Recall", "F1-Score", "Support"],
    ["NORMAL",   "0.9909", "0.4658", "0.6337", "234"],
    ["BACTERIA", "0.8201", "0.9421", "0.8769", "242"],
    ["VIRUS",    "0.4831", "0.7703", "0.5938", "148"],
    ["Macro avg","0.7647", "0.7261", "0.7015", "624"],
]
story.append(metric_table(cnn3_detail, [3.5*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm]))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(
    "The CNN had a weird split — precision on normal was 99% but recall was only 46%. "
    "That means when it said normal it was almost always right, but it missed more than "
    "half of the actual normal cases. The softmax pushes it to commit to bacteria a lot "
    "because that's the biggest class. SVM with OvR handled all three classes more evenly "
    "and ended up with the better macro F1 overall.", Body))

story.append(Paragraph(b("Changes made for 3-class"), H2))
story.append(Paragraph(
    "<b>CNN output:</b>  Dense(3, softmax)  instead of Dense(1, sigmoid)<br/>"
    "<b>CNN loss:</b>  sparse_categorical_crossentropy<br/>"
    "<b>SVM:</b>  decision_function_shape='ovr'  (one model per class vs the rest)<br/>"
    "<b>KNN best K:</b>  18  (re-swept from scratch on 3-class data)", Code))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. CONCLUSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
story.append(PageBreak())
story.append(section_bar("8.  What We Learned"))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph(
    "Overall the project went well but it took longer than expected, mostly because of the "
    "class imbalance and getting the evaluation metric right. A few things stood out:", Body))

conclusions = [
    "SVM ended up being the best model — 97% on both classes in Split 1 and 0.9952 ROC-AUC. "
    "The trick was the scoring metric in GridSearchCV, not the model itself.",
    "Getting 0% normal accuracy was a confusing bug to track down. The model looked fine "
    "by overall accuracy but was completely wrong on half the possible outputs.",
    "PCA was non-negotiable for SVM and KNN. Without it, those two were unusable on a "
    "standard laptop — training just never finished.",
    "Cross-validation (Split 3) gave us the most consistent numbers. Split 1 varied a lot "
    "depending on the random seed; Split 3 averaged that out.",
    "CNN was the only model that learns spatial patterns — lung texture, infiltrate shapes. "
    "The others just see 100 PCA numbers. That's probably why it handles normal cases "
    "differently and why it gets confused differently too.",
    "Split 2 scores being 0.15-0.20 lower than Split 1/3 shows how much performance can "
    "drop when you test on a genuinely different dataset. That gap is probably closer "
    "to what real-world performance would look like.",
    "3-class was harder as expected. Virus recall was weak across all models — "
    "the visual difference between bacterial and viral pneumonia is subtle even to experts.",
]
for c in conclusions:
    story.append(bullet(c))

story.append(Spacer(1, 0.5*cm))
story.append(Paragraph(b("Files submitted"), H2))
deliverables = [
    "zoidberg2_pneumonia.ipynb — full notebook with all models, splits, and outputs",
    "zoidberg2_pneumonia.html — exported notebook in HTML",
    "synthesis_report.pdf — this report",
    "models/svm_best.joblib — SVM binary model",
    "models/knn_best.joblib — KNN binary model",
    "models/cnn_best.keras — CNN binary model weights",
    "models/svm3_best.joblib, knn3_best.joblib, cnn3_best.keras — 3-class models",
    "outputs/figures/ — all plots (ROC, confusion matrices, training curves, K sweep)",
]
for d in deliverables:
    story.append(bullet(d))

story.append(Spacer(1, 0.8*cm))
story.append(HRFlowable(width="100%", thickness=1, color=TEAL))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Zoidberg 2.0  —  Epitech MSc IT Machine Learning  —  May 2026",
    S('foot', fontSize=9, fontName='Helvetica-Oblique', textColor=MGRAY,
      alignment=TA_CENTER)))

# ── Build ──────────────────────────────────────────────────────────────
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

import os
size = os.path.getsize(OUT) / 1024
print(f"PDF saved -> {OUT}  ({size:.0f} KB)")
