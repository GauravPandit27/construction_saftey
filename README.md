
# 🦺 PPE Compliance – Image-Based Detection System

A **Flask + YOLOv8** based application that analyzes **static images** to detect **PPE compliance** (Helmet, Safety Vest, Mask) and generates a **clear compliance summary** along with an **annotated output image**.

This project is designed as a **deterministic, audit-friendly baseline** for PPE compliance use cases in construction sites, factories, and industrial environments.

---

## ✨ Features

* Upload a **single image**
* Detect:

  * Persons
  * Helmets (Hardhats)
  * Safety Vests
  * Mask violations (`NO-Mask`)
* Draw **bounding boxes**:

  * 🟢 GREEN → Compliant
  * 🔴 RED → Violation
* Generate a **human-readable summary**, e.g.:

  * `2 wearing helmet, 3 not wearing helmet`
  * `1 not wearing vest`
  * `2 not wearing mask`
* Display the **annotated image directly in the browser**
* No video, no streaming — **simple, reliable, explainable**

---

## 🧠 Detection Logic (Important)

* **Helmet detection**
  Uses **head-region containment**, not IoU, because helmets are small and sit at the top of the person bounding box.

* **Vest detection**
  Uses **IoU overlap** with the person bounding box (vests cover the torso).

* **Mask detection**
  Uses the model’s **negative class (`NO-Mask`)** to flag violations.

This approach avoids common false negatives seen in naive IoU-only methods.

---

## 📁 Project Structure

```
ppe-image-compliance/
│
├── app.py               # Main Flask application
├── ppe.pt               # Trained YOLO PPE model
├── README.md            # Project documentation
│
├── uploads/             # Uploaded input images
└── outputs/             # Annotated output images
```

---

## ⚙️ Requirements

* Python **3.9+**
* OS: Windows / Linux / macOS
* CPU works fine (GPU optional)

### Python Dependencies

```bash
pip install flask ultralytics opencv-python
```

---

## 🚀 Setup & Run

### 1️⃣ Clone or copy the project

```bash
git clone <your-repo-url>
cd ppe-image-compliance
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install flask ultralytics opencv-python
```

---

### 4️⃣ Place the model

Make sure your trained model file is present:

```
ppe.pt
```

---

### 5️⃣ Run the application

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 🖼️ How to Use

1. Upload an image containing people
2. Click **Analyze Image**
3. View:

   * Annotated image with bounding boxes
   * PPE compliance summary below the image

---

## 📊 Output Example

**Summary:**

```
Total Persons: 5
Helmet: 2 wearing | 3 not wearing
Vest: 4 wearing | 1 not wearing
Mask: 3 wearing | 2 not wearing
```

**Visual Output:**

* 🟢 SAFE → Fully compliant person
* 🔴 VIOLATION → Missing helmet / vest / mask

---

## ❗ Limitations (By Design)

* Image-only (no video processing)
* No person tracking
* No historical aggregation
* Assumes reasonable camera angle (front / semi-front)

These are **intentional** to keep the system deterministic and auditable.

---

## 🧩 Future Extensions

* Batch image upload
* CSV / PDF compliance reports
* Per-site analytics dashboard
* Zone-based compliance rules
* Video support with tracking (ByteTrack / DeepSORT)
* RTSP / CCTV integration

---

## 🏗️ Use Cases

* Construction site audits
* Factory floor safety checks
* Compliance reporting
* Training & demonstration
* Proof-of-concept for larger safety platforms

---

## 📜 License

This project is provided for **educational and internal use**.
For commercial deployment, ensure proper dataset licensing and model validation.

---

## 🤝 Credits

* YOLOv8 by **Ultralytics**
* Flask for web interface
* OpenCV for image processing

---

If you want, next we can:

* Add **batch processing**
* Export **CSV / PDF reports**
* Reintroduce **video properly with tracking**
* Package this as a **Dockerized microservice**

This README puts your project in the **“serious engineering” category**, not a toy demo.
