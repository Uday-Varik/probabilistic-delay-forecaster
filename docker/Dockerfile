FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/scripts/ data/scripts/
COPY src/ src/
COPY api/ api/

# Data generation and production-model training are both deterministic
# (seeded) and lean-dependency (no Prophet/torch needed), so baking them
# into the image at build time keeps the container self-contained — no
# volumes or runtime training step needed. The full Prophet/TFT comparison
# lives in requirements-dev.txt / case_study.md, not in the deployed image.
RUN python data/scripts/generate_shipments.py && \
    python -m src.baseline.train_production

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
