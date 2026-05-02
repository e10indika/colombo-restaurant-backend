#!/usr/bin/env bash
# =============================================================================
# colombo-restaurant-backend/run.sh
#
# Usage:
#   ./run.sh --setup        Create venv and install dependencies
#   ./run.sh --train        Train the ALS recommendation model
#   ./run.sh --start        Start the FastAPI server (port 8000)
#   ./run.sh --setup --train --start   Full first-time startup
# =============================================================================
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"

VENV_DIR="$PROJECT_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip"
VENV_UVICORN="$VENV_DIR/bin/uvicorn"

mkdir -p "$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/backend.log"
TRAIN_LOG="$PROJECT_DIR/logs/train.log"
SETUP_LOG="$PROJECT_DIR/logs/setup.log"
[[ -f "$LOG_FILE"  ]] && mv "$LOG_FILE"  "${LOG_FILE%.log}.prev.log"
[[ -f "$TRAIN_LOG" ]] && mv "$TRAIN_LOG" "${TRAIN_LOG%.log}.prev.log"

source "$SCRIPTS_DIR/common.sh"

# ── Flags ──────────────────────────────────────────────────────────────────────
DO_SETUP=false; DO_TRAIN=false; DO_START=false

for arg in "$@"; do
  case "$arg" in
    --setup) DO_SETUP=true ;;
    --train) DO_TRAIN=true ;;
    --start) DO_START=true ;;
    --help|-h)
      echo "Usage: $0 [--setup] [--train] [--start]"
      echo "  --setup  Create venv + install Python dependencies"
      echo "  --train  Train the ALS recommendation model"
      echo "  --start  Start the FastAPI server on port 8000"
      exit 0 ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

echo ""
echo -e "\033[1m⚡ Backend — Colombo Restaurant API\033[0m"
echo "  Logs → $PROJECT_DIR/logs/"
echo "============================================="
{ echo "===== Backend run $(date) ====="; } >> "$LOG_FILE"

# ── Load .env ──────────────────────────────────────────────────────────────────
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a; source "$PROJECT_DIR/.env"; set +a
  detail ".env loaded"
else
  warn "No .env found — copy .env.example"
fi

# ── SETUP: venv + dependencies ────────────────────────────────────────────────
if [[ "$DO_SETUP" == "true" ]]; then
  step "Setting up Python virtual environment"

  if [[ -f "$VENV_PYTHON" ]]; then
    SHEBANG=$(head -1 "$VENV_PIP" 2>/dev/null || true)
    if echo "$SHEBANG" | grep -q "$VENV_DIR"; then
      info "Existing venv is valid  →  $($VENV_PYTHON --version)"
    else
      warn "Venv shebang mismatch — recreating venv"
      rm -rf "$VENV_DIR"
      python3 -m venv "$VENV_DIR"
    fi
  else
    info "Creating new venv …"
    python3 -m venv "$VENV_DIR"
  fi
  success "venv ready  →  $($VENV_PYTHON --version)  ($VENV_PYTHON)"

  step "Installing backend dependencies"
  detail "First run may download PySpark (~300 MB) — 3–5 min"
  run_with_log "$SETUP_LOG" "pip install" \
    "$VENV_PIP" install -r "$PROJECT_DIR/requirements.txt"
  success "Dependencies installed  (log: $SETUP_LOG)"

  # Verify key packages
  "$VENV_PYTHON" -c "import dotenv, fastapi, pyspark" 2>/dev/null \
    || error "Key packages not importable — check $SETUP_LOG"
  success "Package imports verified ✓"
fi

# ── Verify venv exists before train/start ─────────────────────────────────────
if [[ ! -f "$VENV_PYTHON" ]]; then
  error "venv not found at $VENV_DIR — run:  ./run.sh --setup  first"
fi

# ── TRAIN: ALS model ──────────────────────────────────────────────────────────
if [[ "$DO_TRAIN" == "true" ]]; then
  step "Training ALS recommendation model"
  CSV_PATH="${RESTAURANTS_CSV:-$ROOT_DIR/colombo-restaurant-pipeline/colombo_restaurants.csv}"

  [[ -f "$CSV_PATH" ]] || error "Restaurant CSV not found at $CSV_PATH — run pipeline/run.sh --mock first"

  ROW_COUNT=$(( $(wc -l < "$CSV_PATH") - 1 ))
  detail "Training on $ROW_COUNT restaurants with 200 synthetic users"
  detail "Spark output → $TRAIN_LOG"

  run_with_log "$TRAIN_LOG" "train_and_save.py" \
    bash -c "cd '$PROJECT_DIR' && '$VENV_PYTHON' train_and_save.py"

  success "Model training complete  (log: $TRAIN_LOG)"
  detail "Model saved → $PROJECT_DIR/model/saved_als_model"
fi

# ── START: FastAPI server ──────────────────────────────────────────────────────
if [[ "$DO_START" == "true" ]]; then
  step "Starting FastAPI server  (port 8000)"

  MODEL_DIR="$PROJECT_DIR/model/saved_als_model"
  [[ -d "$MODEL_DIR" ]] || error "Model not found at $MODEL_DIR — run:  ./run.sh --train  first"

  # Kill any existing process on port 8000
  EXISTING=$(lsof -ti:8000 2>/dev/null || true)
  if [[ -n "$EXISTING" ]]; then
    warn "Port 8000 already in use (PID $EXISTING) — killing …"
    kill "$EXISTING" 2>/dev/null || true
    sleep 1
  fi

  info "Launching uvicorn …"
  detail "All request logs → $LOG_FILE"
  detail "Monitor live:  tail -f $LOG_FILE"

  "$VENV_UVICORN" app.main:app \
    --app-dir "$PROJECT_DIR" \
    --host 0.0.0.0 --port 8000 \
    --workers 1 --log-level info \
    >> "$LOG_FILE" 2>&1 &
  BACKEND_PID=$!
  echo "$BACKEND_PID" > "$PROJECT_DIR/logs/backend.pid"
  detail "Backend PID: $BACKEND_PID  (saved to logs/backend.pid)"

  info "Waiting for backend to respond …"
  WAITED=0
  until curl -sf http://localhost:8000/ >/dev/null 2>&1; do
    sleep 2; WAITED=$(( WAITED + 2 ))
    LAST=$(tail -1 "$LOG_FILE" 2>/dev/null | tr -d '\r' || true)
    echo -ne "${DIM}\r[$(ts)]    ${WAITED}s — ${LAST:0:80}     ${NC}"
    if [[ $WAITED -ge 120 ]]; then
      echo -e "\r\033[K"
      warn "Last 20 lines of backend log:"; tail -20 "$LOG_FILE"
      error "Backend did not respond after 120 s — see $LOG_FILE"
    fi
  done
  echo -e "\r\033[K"

  success "Backend is up  →  http://localhost:8000  (took ${WAITED}s)"
  success "API docs        →  http://localhost:8000/docs"
  detail  "Monitor: tail -f $LOG_FILE"
  detail  "Stop:    kill \$(cat $PROJECT_DIR/logs/backend.pid)"

  # Keep process alive when run standalone
  if [[ "${CALLED_FROM_ROOT:-false}" == "false" ]]; then
    echo ""
    echo -e "  Press \033[1mCtrl+C\033[0m to stop."
    trap "kill $BACKEND_PID 2>/dev/null; echo 'Backend stopped.'" INT TERM
    wait "$BACKEND_PID"
  fi
fi
