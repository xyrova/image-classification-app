FROM python:3.14-slim

WORKDIR /app

# Install Runtime deps only(fastapi, uvicorn, etc..), pinned to the versions in the lock file
COPY requirements-lock.txt .

RUN pip install --no-cache-dir -c requirements-lock.txt \
    fastapi uvicorn python-multipart numpy scikit-learn pillow pyyaml joblib

# Models must be present locally first: dvc pull models/model.joblib models/feature_selector.joblib
COPY params.yaml .
COPY src/mlp.py src/
COPY models/ models/
COPY app/ app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
