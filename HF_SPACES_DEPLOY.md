# Deploying the Streamlit dashboard to HuggingFace Spaces

HuggingFace Spaces provides a free, always-on public URL for the Streamlit demo.
You only need to do this once; every subsequent `git push` redeploys.

## 1. Create the Space

1. Create a free account at https://huggingface.co (if you don't have one).
2. Click **New** → **Space**.
3. Fill in:
   - **Owner**: your username
   - **Space name**: `readmission-risk` (or similar — this becomes the URL)
   - **License**: MIT
   - **SDK**: **Streamlit**
   - **Hardware**: CPU basic (free)
   - **Visibility**: Public
4. Click **Create Space**. You will be given a URL like `https://huggingface.co/spaces/<you>/readmission-risk` and a git remote.

## 2. Prepare the deployment folder

The Space needs specific files at its repository root. The cleanest approach is a
separate clone of the Space repo:

```cmd
cd c:\Users\shrut\OneDrive\Desktop
git clone https://huggingface.co/spaces/<you>/readmission-risk hf-space
cd hf-space
```

Copy these files from this project into the Space clone (drag-and-drop in
Explorer is fine):

```
streamlit_app.py
src\               (whole folder)
artifacts\         (whole folder — contains champion.pkl and metadata)
reports\           (whole folder — contains fairness/drift JSONs)
docs\screenshots\  (whole folder — used by Performance page)
```

Then create an `app.py` shim (Spaces Streamlit SDK expects this entry point) and
a deploy-specific `requirements.txt`.

### `app.py` (at repo root of the Space)
```python
from streamlit_app import main
main()
```

### `requirements.txt` (at repo root of the Space)
```
streamlit==1.40.1
plotly==5.24.1
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
xgboost==2.1.3
shap==0.46.0
joblib==1.4.2
matplotlib==3.9.2
```

### `README.md` (at repo root of the Space)
```markdown
---
title: Readmission Risk
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.1
app_file: app.py
pinned: false
---

# Readmission Risk MLOps Demo

Live dashboard for the 30-day hospital readmission risk model.
Source code: https://github.com/<you>/readmission-mlops
```

## 3. Push to deploy

```cmd
cd c:\Users\shrut\OneDrive\Desktop\hf-space
git add .
git commit -m "Initial deploy"
git push
```

HuggingFace builds and serves the Streamlit app automatically. Build logs are
visible on the Space's page; first build takes 3–5 minutes.

Your public URL will be `https://huggingface.co/spaces/<you>/readmission-risk`.
Link this in your resume and the main GitHub README.

## 4. Optional — deploy the FastAPI service as a second Docker Space

If you want a public `/docs` endpoint alongside the Streamlit UI, create a
second Space with **SDK: Docker** and point it at this project's `Dockerfile`
and `artifacts/`. The free tier gives 16 GB RAM, which is enough for this image.
