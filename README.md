# Deepfake Image Detector (Streamlit)

A web app that estimates whether a face photo is real or AI-generated / AI-modified.
Model: ResNet50 fine-tuned on face images (see the "Model & results" tab in the app).

## Folder layout
```
app.py              the Streamlit app
pipeline.py         image loading, face detection and cropping, heat-map overlay
model_utils.py      model loading, prediction and Grad-CAM (PyTorch)
check_setup.py      self-test
requirements.txt
model/              put best.pt and final_settings.json here
samples/            optional sample images
.streamlit/config.toml
```

## Run it on your computer
1. Install Python 3.10 or newer.
2. Copy `best.pt` and `final_settings.json` from Google Drive (`Deepfake_Project/final_model/`) into `model/`.
3. In a terminal, inside this folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)   or   source .venv/bin/activate   (macOS/Linux)
   pip install -r requirements.txt
   python check_setup.py
   streamlit run app.py
   ```
4. Your browser opens at http://localhost:8501. Runs on CPU; each photo takes about a second or less.

## Share it online (optional)
Streamlit Community Cloud can host the app from a GitHub repository (free, public link).
Check the current instructions at docs.streamlit.io before you start, because the details change.
Points to watch:
- `best.pt` is about 95 MB. GitHub rejects single files over 100 MB, so it fits, but it is close.
  Check GitHub's current limits, and use Git LFS if you add larger files.
- Community Cloud has limited memory (documented as roughly 2.7 GB maximum as of February 2024; limits can change).
  The app loads one model and caches at most 64 results, which should fit.
- Apps without traffic for 12 hours go to sleep and wake up when someone visits.
- Pin the Streamlit version in `requirements.txt` (run `pip show streamlit` and write `streamlit==<version>`).
- If installing PyTorch fails there, remove the first line of `requirements.txt`.
- Uploaded photos are processed on the host's server. Do not upload private photos to a public deployment.

## Troubleshooting
- "Model file not found": `model/best.pt` is missing.
- `ModuleNotFoundError`: the virtual environment is not active, or `pip install -r requirements.txt` was not run.
- "No face found" for a normal photo: the face detector is basic. Use a larger, front-facing face, or
  untick "Detect and crop the face" if the image is already a tight face crop.
