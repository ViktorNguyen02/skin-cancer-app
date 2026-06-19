from flask import Flask, request, jsonify, render_template, redirect
from keras.models import load_model
import numpy as np
from PIL import Image
import os
from datetime import datetime
import csv
from io import BytesIO  
from xhtml2pdf import pisa
from flask import make_response
from flask import render_template
from datetime import datetime
from gradcam_utils import generate_and_save_gradcam
from PIL import Image
import uuid  # for unique filenames



app = Flask(__name__)

# Load the model
model = load_model("skin_cancer_model_25.h5")
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

# Descriptions and links
descriptions = {
    "nv": {
        "name": "Melanocytic Nevus",
        "desc": "Benign proliferations of melanocytes, often called moles. They are usually uniform in color, symmetrical, and stable over time. While most are harmless, changes in size, shape, or color may warrant further evaluation to rule out melanoma."
    },
    "mel": {
        "name": "Melanoma",
        "desc": "A serious and potentially life-threatening form of skin cancer originating from melanocytes. Melanoma may appear as an irregular, multicolored lesion and can spread rapidly if not detected early. Early diagnosis and treatment are critical to improving prognosis."
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "desc": "A group of non-cancerous skin growths that includes seborrheic keratosis, solar lentigo, and lichen-planus-like keratosis. These lesions are generally related to aging or sun exposure and are often mistaken for malignancies due to their variable appearance."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "desc": "The most common type of skin cancer. BCC arises from the basal cells in the epidermis and typically presents as a pearly bump, often with visible blood vessels. It grows slowly and rarely metastasizes but can cause local tissue damage if untreated."
    },
    "akiec": {
        "name": "Actinic Keratosis / Bowen's Disease",
        "desc": "A precancerous skin lesion caused by long-term sun exposure. Actinic keratoses appear as rough, scaly patches and can progress to squamous cell carcinoma. Bowen’s disease is a form of in situ squamous cell carcinoma and appears as a red, scaly plaque."
    },
    "vasc": {
        "name": "Vascular Lesion",
        "desc": "Includes angiomas, hemangiomas, and pyogenic granulomas — benign growths made up of blood vessels. These lesions can appear red or purple and may be congenital or acquired. They are usually harmless but may bleed or become irritated."
    },
    "df": {
        "name": "Dermatofibroma",
        "desc": "A common, benign fibrous skin nodule that typically develops in response to minor skin injuries like insect bites or folliculitis. Dermatofibromas are firm, dome-shaped, and often hyperpigmented. They are usually asymptomatic and non-cancerous."
    }
}

resource_links = {
    "nv": "https://dermnetnz.org/topics/melanocytic-naevus",
    "mel": "https://dermnetnz.org/topics/melanoma",
    "bkl": "https://dermnetnz.org/topics/seborrhoeic-keratosis",
    "bcc": "https://dermnetnz.org/topics/basal-cell-carcinoma",
    "akiec": "https://dermnetnz.org/topics/actinic-keratosis",
    "vasc": "https://dermnetnz.org/topics/facial-red-vein-and-vascular-birthmark-treatments",
    "df": "https://dermnetnz.org/topics/dermatofibroma"
}

differential_features = {
    "nv": {
        "name": "Melanocytic Nevus",
        "appearance": "Symmetrical, uniform color",
        "growth": "Stable or slow",
        "borders": "Well-defined",
        "risk": "Low – benign"
    },
    "mel": {
        "name": "Melanoma",
        "appearance": "Asymmetrical, multicolored",
        "growth": "Rapid and irregular",
        "borders": "Notched or uneven",
        "risk": "High – malignant"
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "appearance": "Warty or scaly texture",
        "growth": "Slow, sun-related",
        "borders": "Irregular but non-invasive",
        "risk": "Low"
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "appearance": "Pearly or waxy bump",
        "growth": "Slow and localized",
        "borders": "Rolled edges",
        "risk": "Low – rarely metastasizes"
    },
    "akiec": {
        "name": "Actinic Keratosis / Bowen's Disease",
        "appearance": "Rough, scaly patch or red plaque",
        "growth": "Slow with potential to become cancerous",
        "borders": "Ill-defined",
        "risk": "Moderate – precancerous"
    },
    "vasc": {
        "name": "Vascular Lesion",
        "appearance": "Red or purple, sometimes raised",
        "growth": "Varies – congenital or acquired",
        "borders": "Well or poorly defined",
        "risk": "Low – benign"
    },
    "df": {
        "name": "Dermatofibroma",
        "appearance": "Firm, dome-shaped nodule",
        "growth": "Stable, post-injury",
        "borders": "Well-circumscribed",
        "risk": "Low – benign"
    }
}

IMG_SIZE = 224

def preprocess_image(file_path):
    image = Image.open(file_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    patient_id = request.form.get('patient_id', '')

    results = []
    for i, file in enumerate(files):
        if file.filename == '':
            continue

        filename = f"uploaded_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        filepath = os.path.join("static", filename)
        file.save(filepath)

        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        class_name = class_labels[predicted_class]

        prediction_info = descriptions[class_name]
        prediction_info["link"] = resource_links[class_name]

        top3 = sorted([
            {"label": label, "confidence": float(conf) * 100}
            for label, conf in zip(class_labels, prediction[0])
        ], key=lambda x: x["confidence"], reverse=True)[:3]

        result = {
            "image_path": filepath,
            "class_name": class_name,
            "confidence": confidence,
            "prediction_info": prediction_info,
            "probabilities": [float(p) for p in prediction[0]],
            "top3": top3
        }

        if len(files) == 1:
            # Generate Grad-CAM and save
            original_img = Image.open(filepath).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            generate_and_save_gradcam(img_array, original_img, model, save_path='static/gradcam.jpg')

            low_confidence = confidence < 70

            return render_template("result.html",
                class_name=class_name,
                confidence=confidence,
                image_path=filepath,
                prediction_info=prediction_info,
                probabilities=result["probabilities"],
                class_labels=class_labels,
                top3=top3,
                patient_id=patient_id,
                differential_features=differential_features,
                gradcam_path="static/gradcam.jpg",
                low_confidence=low_confidence  # ✅ NEW FLAG
            )

        results.append(result)

    return render_template("multi_result.html", results=results, class_labels=class_labels)

@app.route('/detailed-view')
def detailed_view():
    image_path = request.args.get('image_path')
    patient_id = request.args.get('patient_id', '')

    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    class_name = class_labels[predicted_class]

    prediction_info = descriptions[class_name]
    prediction_info["link"] = resource_links[class_name]

    top3 = sorted([
        {"label": label, "confidence": float(conf) * 100}
        for label, conf in zip(class_labels, prediction[0])
    ], key=lambda x: x["confidence"], reverse=True)[:3]

    # Generate Grad-CAM with unique filename
    original_img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    gradcam_filename = f"gradcam_{uuid.uuid4().hex}.jpg"
    gradcam_path = os.path.join("static", gradcam_filename)
    generate_and_save_gradcam(img_array, original_img, model, save_path=gradcam_path)

    return render_template("result.html",
        class_name=class_name,
        confidence=confidence,
        image_path=image_path,
        prediction_info=prediction_info,
        probabilities=[float(p) for p in prediction[0]],
        class_labels=class_labels,
        top3=top3,
        patient_id=patient_id,
        differential_features=differential_features,
        gradcam_path=gradcam_path
    )

@app.route('/save-batch-notes', methods=['POST'])
def save_batch_notes():
    try:
        image_paths = request.form.getlist('image_paths')
        class_names = request.form.getlist('class_names')
        confidences = request.form.getlist('confidences')
        notes_list = request.form.getlist('notes')
        patient_ids = request.form.getlist('patient_ids')

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_exists = os.path.isfile('clinical_notes.csv')

        with open('clinical_notes.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['timestamp', 'patient_id', 'image_path', 'class_name', 'confidence', 'notes'])

            for pid, img, cls, conf, note in zip(patient_ids, image_paths, class_names, confidences, notes_list):
                writer.writerow([timestamp, pid, img, cls, conf, note.strip()])

        return render_template("notes_saved.html")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save-notes', methods=['POST'])
def save_notes():
    try:
        data = request.form
        patient_id = data.get('patient_id')
        notes = data.get('notes', '')
        image_path = data.get('image_path', '')
        class_name = data.get('class_name', '')
        confidence = data.get('confidence', '')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        file_exists = os.path.isfile('clinical_notes.csv')
        with open('clinical_notes.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Patient ID', 'Image Path', 'Predicted Class', 'Confidence', 'Clinical Notes'])
            writer.writerow([timestamp, patient_id, image_path, class_name, confidence, notes.strip()])

        return jsonify({"success": True, "message": "Notes saved successfully", "notes": notes.strip()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/download-result-pdf')
def download_result_pdf():
    try:
        from flask import request
        import json

        # Receive parameters from the form
        image_path = request.args.get("image_path", "")
        class_name = request.args.get("class_name", "")
        confidence = float(request.args.get("confidence", "0"))
        patient_id = request.args.get("patient_id", "")
        notes = request.args.get("notes", "")
        top3 = json.loads(request.args.get("top3", "[]"))

        # Prepare the differential features dictionary for the top 3 items
        diff_features = {
            item["label"]: differential_features[item["label"]]
            for item in top3
        }
        # Load and preprocess the image
        original_img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = preprocess_image(image_path)

        # Generate Grad-CAM and save it to static/
        gradcam_filename = generate_and_save_gradcam(img_array, original_img, model)
        gradcam_path = os.path.join("static", gradcam_filename)

        # Render the pdf_template.html with correct values
        html_content = render_template(
            "pdf_template.html",
            patient_id=patient_id,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            class_name=class_name,
            confidence=confidence,
            image_path=image_path,
            descriptions=descriptions,
            top3=top3,
            diff_features=diff_features,
            notes=notes,
            gradcam_path=gradcam_path
        )

        # Generate PDF
        pdf_buffer = BytesIO()
        pisa.CreatePDF(BytesIO(html_content.encode("utf-8")), dest=pdf_buffer)

        # Send PDF as a file response
        response = make_response(pdf_buffer.getvalue())
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = f"attachment; filename=diagnostic_report_{patient_id or 'report'}.pdf"
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()  # 👈 Shows full error in console
        app.logger.error(f"PDF generation failed: {str(e)}")
        return f"PDF generation failed: {str(e)}", 500


@app.route('/patient-history', methods=['GET', 'POST'])
def patient_history():
    results = []
    patient_id = ""

    if request.method == 'POST':
        patient_id = request.form.get('patient_id', '').strip()
        if os.path.exists('clinical_notes.csv'):
            with open('clinical_notes.csv', 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    clean_row = {k.strip(): v.strip() if v else '' for k, v in row.items() if k}
                    if clean_row.get('Patient ID') == patient_id:
                        results.append({
                            'timestamp': clean_row.get('Timestamp'),
                            'image_path': clean_row.get('Image Path'),
                            'class_name': clean_row.get('Predicted Class'),
                            'confidence': clean_row.get('Confidence'),
                            'notes': clean_row.get('Clinical Notes'),
                            'patient_id': clean_row.get('Patient ID')
                        })

    return render_template("patient_history.html", results=results, patient_id=patient_id)

if __name__ == '__main__':
    app.run(debug=True) 