import os
import cv2
from flask import Flask, render_template_string, request, send_from_directory
from ultralytics import YOLO

# ---------------- SETUP ----------------

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO("ppe.pt")

# Normalize class names
class_names = {
    k: v.lower().replace("-", "").replace(" ", "")
    for k, v in model.names.items()
}

PPE_CLASSES = {"hardhat", "safetyvest", "mask"}
VIOLATION_CLASSES = {"nomask"}

# ---------------- UTILS ----------------

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter + 1e-6)

# ---------------- IMAGE PROCESSING ----------------

def process_image(image_path):
    image = cv2.imread(image_path)
    results = model(image, conf=0.35, verbose=False)[0]

    persons, ppe_items, violations = [], [], []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = class_names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "person":
            persons.append((x1, y1, x2, y2))
        elif label in PPE_CLASSES:
            ppe_items.append((label, (x1, y1, x2, y2)))
        elif label in VIOLATION_CLASSES:
            violations.append((label, (x1, y1, x2, y2)))

    helmet_yes = helmet_no = 0
    vest_yes = vest_no = 0
    mask_yes = mask_no = 0

    mask_items = [box for lbl, box in ppe_items if lbl == "mask"]

    for px1, py1, px2, py2 in persons:
        # -------- Helmet
        has_helmet = False
        for lbl, (hx1, hy1, hx2, hy2) in ppe_items:
            if lbl != "hardhat":
                continue
            hc_x = (hx1 + hx2) // 2
            hc_y = (hy1 + hy2) // 2
            head_limit = py1 + int(0.35 * (py2 - py1))
            if px1 < hc_x < px2 and py1 < hc_y < head_limit:
                has_helmet = True
                break

        # -------- Vest
        has_vest = any(
            lbl == "safetyvest" and iou((px1, py1, px2, py2), box) > 0.1
            for lbl, box in ppe_items
        )

        # -------- Mask (CORRECT LOGIC)
        has_mask = any(
            iou((px1, py1, px2, py2), box) > 0.1
            for box in mask_items
        )

        has_nomask = any(
            lbl == "nomask" and iou((px1, py1, px2, py2), box) > 0.1
            for lbl, box in violations
        )

        if has_nomask:
            mask_state = "NO_MASK"
        elif has_mask:
            mask_state = "MASK"
        else:
            mask_state = "UNKNOWN"  # treated as unsafe

        # -------- Counters
        helmet_yes += has_helmet
        helmet_no += not has_helmet

        vest_yes += has_vest
        vest_no += not has_vest

        if mask_state == "MASK":
            mask_yes += 1
        else:
            mask_no += 1

        # -------- AI CONFIDENCE
        confidence = (
            (1 if has_helmet else 0) +
            (1 if has_vest else 0) +
            (1 if mask_state == "MASK" else 0)
        )
        confidence = int((confidence / 3) * 100)

        unsafe = not has_helmet or not has_vest or mask_state != "MASK"

        if unsafe:
            color = (0, 0, 255)
            label_text = f"UNSAFE | {confidence}%"
        else:
            color = (0, 255, 0)
            label_text = f"SAFE | {confidence}%"

        cv2.rectangle(image, (px1, py1), (px2, py2), color, 2)
        cv2.putText(image, label_text, (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ---------------- AI DECISION ENGINE ----------------

    compliance_score = 0
    if persons:
        compliance_score = int(
            ((helmet_yes + vest_yes + mask_yes) / (3 * len(persons))) * 100
        )

    if compliance_score >= 85:
        risk_level = "LOW"
        recommendation = "Site is compliant. Maintain existing safety protocols."
    elif compliance_score >= 60:
        risk_level = "MEDIUM"
        recommendation = "Partial compliance detected. Increase supervision and PPE enforcement."
    else:
        risk_level = "HIGH"
        recommendation = "Critical safety risk identified. Immediate corrective action required."

    output_filename = "output.jpg"
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, output_filename), image)

    summary = {
        "total": len(persons),
        "helmet": (helmet_yes, helmet_no),
        "vest": (vest_yes, vest_no),
        "mask": (mask_yes, mask_no),
        "compliance": compliance_score,
        "risk": risk_level,
        "recommendation": recommendation
    }

    return output_filename, summary

# ---------------- ROUTES ----------------

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>AI Safety Sentinel</title>
<style>
body {
  background: radial-gradient(circle at top, #020617, #000000);
  color:white;
  font-family:Arial;
  text-align:center
}

.card {
  margin:auto;
  width:90%;
  background:#020617;
  padding:25px;
  border-radius:14px;
  box-shadow:0 0 40px rgba(56,189,248,0.15)
}

h1 { color:#38bdf8 }
.subtitle { opacity:0.7 }

.stat-grid {
  display:grid;
  grid-template-columns:repeat(5,1fr);
  gap:18px;
  margin-top:25px;
}

.stat {
  background:#020617;
  border:1px solid #1e293b;
  padding:15px;
  border-radius:12px;
  font-size:17px;
}

.risk-low { color:#22c55e }
.risk-medium { color:#facc15 }
.risk-high { color:#ef4444 }

.reco {
  margin-top:20px;
  padding:18px;
  border-left:5px solid #38bdf8;
  background:#020617;
  text-align:left;
}

button {
  padding:12px 30px;
  font-size:17px;
  border:none;
  border-radius:10px;
  background:#38bdf8;
  color:black;
  cursor:pointer;
}
</style>
</head>
<body>

<h1>AI Safety Sentinel™</h1>
<p class="subtitle">Autonomous PPE Compliance & Risk Intelligence System</p>

<div class="stat" style="margin:auto;width:70%">
🧠 AI Engine: <b style="color:#22c55e">ACTIVE</b> |
📡 Vision Model: <b>YOLOv8</b> |
⚙ Decision Engine: <b>Spatial + Rule-Based AI</b>
</div>

<br>

<div class="card">
<form method="POST" enctype="multipart/form-data">
<input type="file" name="image" required>
<br><br>
<button>Run AI Safety Analysis</button>
</form>
</div>

{% if summary %}
<br>
<div class="card">
<img src="/outputs/{{ image_name }}" width="900"><br>

<div class="stat-grid">
  <div class="stat">👷 Workforce<br><b>{{ summary.total }}</b></div>
  <div class="stat">🪖 Helmet OK<br><b>{{ summary.helmet[0] }}</b></div>
  <div class="stat">🦺 Vest OK<br><b>{{ summary.vest[0] }}</b></div>
  <div class="stat">😷 Mask OK<br><b>{{ summary.mask[0] }}</b></div>
  <div class="stat">📊 Compliance<br><b>{{ summary.compliance }}%</b></div>
</div>

<div class="stat" style="font-size:22px;margin-top:20px">
🚦 Site Safety Index:
<b class="
{% if summary.risk=='LOW' %}risk-low
{% elif summary.risk=='MEDIUM' %}risk-medium
{% else %}risk-high{% endif %}
">
{{ summary.compliance }}% — {{ summary.risk }}
</b>
</div>

<div class="reco">
<b>🤖 Autonomous Safety Advisor Output</b><br>
<span style="opacity:0.6">Generated by AI Safety Sentinel™</span><br><br>
{{ summary.recommendation }}
</div>

</div>
{% endif %}

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    image_name = None

    if request.method == "POST":
        file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        image_name, summary = process_image(image_path)

    return render_template_string(HTML, summary=summary, image_name=image_name)

# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)
