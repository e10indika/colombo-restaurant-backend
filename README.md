# Colombo Restaurant Recommendation Backend

A **production-grade FastAPI + PySpark** backend that serves personalised Colombo restaurant recommendations powered by an ALS collaborative-filtering model.

---

## Architecture

```
colombo-restaurant-pipeline/          ← data collection + ALS training pipeline
        ↓  (shares colombo_restaurants.csv)
colombo-restaurant-backend/
├── app/                              ← FastAPI application
│   ├── main.py                       ← routes
│   ├── config.py                     ← all env-var config in one place
│   ├── model/
│   │   └── als_model.py              ← Spark + ALSModel singletons
│   ├── repositories/
│   │   └── restaurant_repo.py        ← Spark SQL queries (data access layer)
│   ├── services/
│   │   └── recommendation_service.py ← business logic
│   ├── schemas/
│   │   └── response.py               ← Pydantic response models
│   └── utils/
│       └── metadata.py               ← cuisine / location / price helpers
├── training/                         ← one-time ALS training pipeline
│   ├── train.py                      ← orchestrator entry point
│   ├── spark_session.py              ← SparkSession factory for training
│   ├── data_loader.py                ← load restaurant CSV + assign integer IDs
│   ├── rating_generator.py           ← synthetic user-rating matrix
│   ├── model_trainer.py              ← ALS fit + RMSE evaluation
│   └── artefact_builder.py           ← build top_restaurants.json
├── model/
│   ├── saved_als_model/              ← trained PySpark ALS model
│   └── top_restaurants.json          ← pre-computed top-50 list
├── data/
│   └── colombo_restaurants.csv       ← restaurant source data (symlinked/copied)
├── run.sh                            ← unified run script (setup / train / start)
└── logs/
    ├── backend.log                   ← live API request log
    ├── train.log                     ← model training output
    └── setup.log                     ← pip install output
```

---

## Quick Start

### Option A — Use `run.sh` (recommended)

`run.sh` handles venv creation, dependency install, model training, and server startup in one command.

```bash
cd colombo-restaurant-backend

# First-time setup: create venv, install deps, train model, start server
./run.sh --setup --train --start

# After the first run (venv + model already exist):
./run.sh --start
```

#### All `run.sh` flags

| Flag | What it does |
|------|-------------|
| `--setup` | Create Python venv at `venv/` and install all dependencies |
| `--train` | Train the ALS model from the restaurant CSV and save to `model/` |
| `--start` | Start the FastAPI server on port 8000 (background process) |

Flags can be combined: `./run.sh --setup --train --start`

> **Prerequisites**: The restaurant CSV must exist before `--train`.  
> Generate it first with the pipeline:
> ```bash
> cd ../colombo-restaurant-pipeline
> ./run.sh --mock     # no API key needed
> # or
> ./run.sh --collect  # real Google Places data (needs GOOGLE_API_KEY)
> ```

#### Logs

| Log file | Contains |
|----------|---------|
| `logs/setup.log` | pip install output |
| `logs/train.log` | Spark training progress |
| `logs/backend.log` | Live API request log |
| `logs/backend.pid` | PID of the running server process |

```bash
# Watch live server logs
tail -f logs/backend.log

# Stop the server
kill $(cat logs/backend.pid)
```

---

### Option B — Manual steps

```bash
# 1. Create venv and install dependencies
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# 2. Train the ALS model
venv/bin/python -m training.train

# 3. Start the server
venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `RESTAURANTS_CSV` | `../colombo-restaurant-pipeline/colombo_restaurants.csv` | Path to restaurant data |
| `MODEL_PATH` | `model/saved_als_model` | Path to trained ALS model |
| `TOP_RESTAURANTS_PATH` | `model/top_restaurants.json` | Path to pre-computed top list |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/recommend/{user_id}?top_n=10` | Top N personalised recommendations |
| GET | `/top-restaurants?limit=20` | Most popular restaurants (pre-computed JSON) |
| GET | `/restaurants?cuisine=&district=&min_rating=&sort_by=rating&limit=100` | Browse / filter all restaurants |
| GET | `/analytics/trending` | Top cuisines + busiest areas (live Spark SQL) |

### `/recommend/{user_id}` — example response

```json
{
  "userId": 5,
  "is_cold_start": false,
  "recommendations": [
    {
      "restaurantId": 42,
      "name": "The Lagoon — Cinnamon Grand",
      "cuisine": "Seafood",
      "location": "Colombo 03",
      "price_range": "Expensive",
      "predicted_rating": 4.72,
      "busyness_score": 83.4
    }
  ]
}
```

> **Cold start**: if `user_id` was not seen during training, `is_cold_start` is `true` and the response falls back to top-rated restaurants. HTTP status remains **200**.

### `/analytics/trending` — example response

```json
{
  "top_cuisines": [
    { "cuisine": "Seafood", "restaurant_count": 34, "avg_rating": 4.3 }
  ],
  "busiest_areas": [
    { "location": "Colombo 03", "restaurant_count": 58, "avg_busyness_score": 76.5, "avg_rating": 4.1 }
  ]
}
```

---

## Interactive Docs

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)  
ReDoc:       [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Deployment Notes

- Use **a single uvicorn worker** (`--workers 1`) — Spark is not process-safe across multiple workers.
- Minimum **4 GB RAM** recommended (Spark driver: 2 GB + overhead).
- **Java 11 or 17** must be on `PATH`.
- The `venv/` created here is **shared** with `colombo-restaurant-pipeline` — the pipeline's `run.sh` will reuse it automatically.

