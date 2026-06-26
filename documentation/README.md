# Skin Cancer Classification App

This is a Flask web app for classifying skin lesion images with a trained deep learning model. It includes prediction results, top class probabilities, Grad-CAM heatmaps, clinical note/report pages, and patient history views.

Important: this project is for learning and research. It is not a medical device and should not be used as a replacement for diagnosis by a qualified clinician.

## What Is Included

- Flask app source code in `app/`
- HTML templates in `app/templates/`
- Model training/evaluation code in `model.py`
- Python dependencies in `requirements.txt`

## What Is Not Included

These files stay on your computer and are intentionally ignored by Git:

- virtual environments: `myenv/`, `myenv311/`
- HAM10000 dataset files in `data/`
- trained model files such as `.h5` and `.keras`
- uploaded images and generated Grad-CAM images
- `app/clinical_notes.csv`

This keeps the GitHub repository smaller and avoids uploading generated or private data.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place your trained model file in the `app/` folder. The app currently expects:

```text
app/skin_cancer_model_25.h5
```

## Run The App

From the project folder:

```bash
cd app
python app.py
```

Then open the local Flask URL shown in the terminal.

## Training Data

The training script expects the HAM10000 dataset locally in the `data/` folder. The dataset is not committed to GitHub because it is large.

## Beginner GitHub Notes

After this project is committed locally, create a new private repository on GitHub. Then connect this folder to GitHub with:

```bash
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
git branch -M main
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY-NAME` with your real GitHub username and repository name.
