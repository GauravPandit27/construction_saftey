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

# normalize class names
class_names = {
    k: v.lower().replace("-", "").replace(" ", "")
    for k, v in model.names.items()
}

PPE_CLASSES = {"hardhat", "safetyvest"}
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

    persons = []
    ppe_items = []
    violations = []

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

    for px1, py1, px2, py2 in persons:
        # -------- Helmet (head region)
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

        # -------- Vest (IoU)
        has_vest = any(
            lbl == "safetyvest" and iou((px1, py1, px2, py2), box) > 0.1
            for lbl, box in ppe_items
        )

        # -------- Mask (negative class)
        has_nomask = any(
            lbl == "nomask" and iou((px1, py1, px2, py2), box) > 0.1
            for lbl, box in violations
        )

        helmet_yes += has_helmet
        helmet_no += not has_helmet
        vest_yes += has_vest
        vest_no += not has_vest
        mask_no += has_nomask
        mask_yes += not has_nomask

        if not has_helmet or not has_vest or has_nomask:
            color = (0, 0, 255)
            label_text = "VIOLATION"
        else:
            color = (0, 255, 0)
            label_text = "SAFE"

        cv2.rectangle(image, (px1, py1), (px2, py2), color, 2)
        cv2.putText(
            image,
            label_text,
            (px1, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    output_filename = "output.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    summary = {
        "total": len(persons),
        "helmet": (helmet_yes, helmet_no),
        "vest": (vest_yes, vest_no),
        "mask": (mask_yes, mask_no),
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
<title>PPE Image Compliance</title>
<style>
body { background:#0f172a; color:white; font-family:Arial; text-align:center }
.card { margin:auto; width:80%; background:#020617; padding:20px; border-radius:12px }
h1 { color:#38bdf8 }
.bad { color:#ef4444 }
.good { color:#22c55e }
button { padding:10px 20px; font-size:16px }
</style>
</head>
<body>

<h1>PPE Compliance – Image Analysis</h1>

<div class="card">
<form method="POST" enctype="multipart/form-data">
<input type="file" name="image" required>
<br><br>
<button>Analyze Image</button>
</form>
</div>

{% if summary %}
<br>
<div class="card">
<img src="/outputs/{{ image_name }}" width="900"><br><br>

<p>Total Persons: {{ summary.total }}</p>

<p>Helmet → 
<span class="good">{{ summary.helmet[0] }}</span> |
<span class="bad">{{ summary.helmet[1] }}</span></p>

<p>Vest → 
<span class="good">{{ summary.vest[0] }}</span> |
<span class="bad">{{ summary.vest[1] }}</span></p>


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

    return render_template_string(
        HTML,
        summary=summary,
        image_name=image_name
    )

# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)
