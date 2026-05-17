#!/usr/bin/env bash
# =============================================================================
# revision/run_baselines_sweep.sh  —  Phase 10 classical-baseline sweep driver
# =============================================================================
#
# Loops over the full 5 models x 2 pipelines x 5 seeds = 50 training matrix for
# the classical-baseline study (Plan 10-03 / BASE-01 / BASE-02). Each triple is
# dispatched to the per-run CLI driver at `revision/run_baselines.py`. The sweep
# is resumable: triples whose run directory already contains the full five-file
# artifact bundle are skipped; failed or in-progress triples are retried on the
# next invocation. Status is recorded to `revision/results/baselines/
# sweep_status.json` atomically (tmp-file + os.rename, advisory flock guard).
#
# Resume semantics
# ----------------
#   For each (model, pipeline, seed) the script considers the triple "complete"
#   iff ALL five artifacts exist AND are non-empty:
#       config.yaml, checkpoint.pt|.npz, samples.npy, metrics.json,
#       inverse_kwargs.npz
#   The checkpoint name is `.npz` for the AR model (D-10-14) and `.pt` for every
#   other model kind. Re-invoking the script after a crash, reboot, or partial
#   run skips already-complete triples and resumes from the first incomplete
#   one. The CLI driver itself is idempotent (overwrites a run_dir cleanly), so
#   partially-written directories are safe to retry.
#
# Parallelism guardrail (Assumption A3 — Mac thermal limit)
# ---------------------------------------------------------
#   --parallel 1  : sequential (default)
#   --parallel 2  : two simultaneous python processes via `xargs -P 2 -L 1`
#                   (D-10-10 recommendation, ~110 min estimate)
#   --parallel >=3: REJECTED with non-zero exit. Mac M-series throttles under
#                   sustained 3+ heavy CPU jobs; observed in v1.1 training.
#
# RESEARCH Pitfall 5 — never multiprocessing.Pool (D-10-24, LOCKED)
# -----------------------------------------------------------------
#   This script intentionally uses `xargs -P N`, NOT Python's
#   `multiprocessing.Pool`. `xargs` spawns fully independent OS processes; each
#   inherits a fresh interpreter + a fresh numpy RNG global state. With
#   `multiprocessing.Pool(fork)`, child workers share the parent's already-
#   warm numpy global RNG, which has corrupted at least one prior reproduction
#   attempt (Pitfall 5, inherited from 09.1 Pitfall 4). Do NOT replace `xargs`
#   with a `Pool`-based driver. There is zero Python multiprocessing anywhere
#   in this sweep or the driver.
#
# Status file schema (revision/results/baselines/sweep_status.json)
# -----------------------------------------------------------------
#   {
#     "started_at": "2026-05-17T12:00:00Z",
#     "epochs": 1000,
#     "parallel": 2,
#     "runs": [
#       {"model":"wgan_mlp","pipeline":"A","seed":42,"status":"complete",
#        "started_at":"...","ended_at":"...","wall_seconds":10800,
#        "return_code":0,"skipped_already_done":false}
#       ...
#     ],
#     "all_complete": false,
#     "completed_count": 0,
#     "total_count": 50
#   }
#
# Canonical invocation
# --------------------
#   # In tmux (preferred — survives ssh/terminal close):
#   tmux new -s baselines_sweep \
#     './revision/run_baselines_sweep.sh --parallel 2 2>&1 | \
#      tee revision/results/baselines/sweep.log'
#
#   # Or nohup background:
#   nohup ./revision/run_baselines_sweep.sh --parallel 2 \
#     > revision/results/baselines/sweep.log 2>&1 &
#
# CLI flags
# ---------
#   --parallel N   : 1 or 2 only (default 1)
#   --epochs M     : override per-run epoch count (default 1000)
#   --dry-run      : print all 50 (model, pipeline, seed) triples with their
#                    current would-run / already-complete status, then exit 0.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants — D-10-05 / D-10-08 / D-10-14 / D-10-23
# -----------------------------------------------------------------------------
MODELS="wgan_mlp wgan_cnn wgan_lstm vae ar"
PIPELINES="A B"
SEEDS="42 43 44 45 46"
EPOCHS=1000
OUT_ROOT="revision/results/baselines"
STATUS_FILE="${OUT_ROOT}/sweep_status.json"
LOCK_FILE="${OUT_ROOT}/.status.lock"

# Python interpreter — prefer project venv (qgan_env) over system python.
# Venv's activate script has a hardcoded incorrect path from a different
# machine, so we invoke the venv binary directly rather than sourcing.
if [ -x "./qgan_env/bin/python" ]; then
  PYTHON="./qgan_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
else
  echo "ERROR: no python interpreter found (looked for ./qgan_env/bin/python, python3, python)" >&2
  exit 2
fi
export PYTHON

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
PARALLEL=1
DRY_RUN=0

usage() {
  sed -n '2,55p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      PARALLEL="${2:-}"
      shift 2
      ;;
    --epochs)
      EPOCHS="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      echo "Usage: $0 [--parallel 1|2] [--epochs N] [--dry-run]" >&2
      exit 2
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Guardrail: --parallel must be 1 or 2 (Assumption A3 — Mac thermal)
# -----------------------------------------------------------------------------
if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [[ "$PARALLEL" -lt 1 ]] || [[ "$PARALLEL" -gt 2 ]]; then
  echo "ERROR: --parallel must be 1 or 2 (got: '${PARALLEL}')." >&2
  echo "       Mac M-series thermal throttling under sustained 3+ heavy jobs" >&2
  echo "       (Assumption A3). D-10-10 recommends --parallel 2 for the" >&2
  echo "       ~110 min wall-time target. If you genuinely need more" >&2
  echo "       parallelism, run on a non-thermal-constrained host and edit" >&2
  echo "       this guardrail intentionally — do not silently lift it." >&2
  exit 3
fi

if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [[ "$EPOCHS" -lt 1 ]]; then
  echo "ERROR: --epochs must be a positive integer (got: '${EPOCHS}')." >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/runs"

# -----------------------------------------------------------------------------
# is_complete <model> <pipeline> <seed>
#   -> exit 0 iff all 5 artifacts exist & non-empty
#
# .npz-aware (D-10-14): the AR model checkpoints to checkpoint.npz; every other
# model kind checkpoints to checkpoint.pt. Keyed on the (model,pipeline,seed)
# 3-tuple (09.1 keyed only on (pipeline,seed) with a fixed checkpoint.pt).
# -----------------------------------------------------------------------------
is_complete() {
  local m="$1" p="$2" s="$3"
  local d="${OUT_ROOT}/runs/${m}/${p}/${s}"
  local ckpt="checkpoint.pt"
  [[ "$m" == "ar" ]] && ckpt="checkpoint.npz"
  [[ -s "${d}/config.yaml" \
     && -s "${d}/${ckpt}" \
     && -s "${d}/samples.npy" \
     && -s "${d}/metrics.json" \
     && -s "${d}/inverse_kwargs.npz" ]]
}

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

# -----------------------------------------------------------------------------
# update_status <model> <pipeline> <seed> <status> <return_code> <wall_seconds>
#               <started_at> <ended_at> <skipped_already_done>
#
# Atomically merges a single per-run record into sweep_status.json.
#
# Protected by an advisory flock on ${LOCK_FILE} so two parallel xargs workers
# can safely write through this helper (T-10-07 mitigation).
# -----------------------------------------------------------------------------
update_status() {
  local m="$1" p="$2" s="$3" st="$4" rc="$5" wall="$6" sa="$7" ea="$8" skipped="$9"

  # Acquire advisory lock. flock blocks until granted, then runs the helper.
  # Subshell + redirection makes flock automatically release on exit.
  (
    flock -x 9
    python3 - "$m" "$p" "$s" "$st" "$rc" "$wall" "$sa" "$ea" "$skipped" \
            "$PARALLEL" "$EPOCHS" "$STATUS_FILE" <<'PY'
import json, os, sys, tempfile

(m, p, s, st, rc, wall, sa, ea, skipped,
 parallel, epochs, status_file) = sys.argv[1:]
s = int(s)
rc = int(rc) if rc != "" else None
wall = int(wall) if wall != "" else None
parallel = int(parallel)
epochs = int(epochs)
skipped_bool = (skipped == "true")

if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
    with open(status_file) as fh:
        doc = json.load(fh)
else:
    doc = {
        "started_at": sa,
        "epochs": epochs,
        "parallel": parallel,
        "runs": [],
        "all_complete": False,
        "completed_count": 0,
        "total_count": 50,
    }

# Always overwrite top-level config fields with the current invocation's values
# (a resumed sweep may use a different --parallel or --epochs).
doc["epochs"] = epochs
doc["parallel"] = parallel
doc.setdefault("total_count", 50)
doc.setdefault("runs", [])

# Find any existing record for this (m, p, s) and replace it; otherwise append.
runs = [
    r for r in doc["runs"]
    if not (r.get("model") == m and r["pipeline"] == p and r["seed"] == s)
]
record = {
    "model": m,
    "pipeline": p,
    "seed": s,
    "status": st,
    "started_at": sa,
    "ended_at": ea,
    "wall_seconds": wall,
    "return_code": rc,
    "skipped_already_done": skipped_bool,
}
runs.append(record)
# Keep runs sorted by (model, pipeline, seed) for human readability.
runs.sort(key=lambda r: (r.get("model", ""), r["pipeline"], r["seed"]))
doc["runs"] = runs

completed = sum(1 for r in runs if r["status"] == "complete")
doc["completed_count"] = completed
doc["all_complete"] = (completed == doc["total_count"])

# Atomic write: tmp file + os.rename (POSIX-atomic when on same filesystem).
dirpath = os.path.dirname(status_file) or "."
fd, tmp = tempfile.mkstemp(prefix=".sweep_status.", suffix=".json", dir=dirpath)
try:
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.rename(tmp, status_file)
except Exception:
    try:
        os.unlink(tmp)
    except OSError:
        pass
    raise
PY
  ) 9>"${LOCK_FILE}"
}

# -----------------------------------------------------------------------------
# run_one <model> <pipeline> <seed>
#
# Invokes the per-run CLI, records status, never aborts the sweep on failure.
# Designed to be safe to call from both sequential and parallel (xargs) modes.
# -----------------------------------------------------------------------------
run_one() {
  local m="$1" p="$2" s="$3"
  local run_dir="${OUT_ROOT}/runs/${m}/${p}/${s}"
  mkdir -p "${run_dir}"

  if is_complete "$m" "$p" "$s"; then
    local sa
    sa="$(iso_now)"
    update_status "$m" "$p" "$s" "complete" "0" "0" "$sa" "$sa" "true"
    echo "[$(iso_now)] SKIP  model=${m} pipeline=${p} seed=${s} (already complete)"
    return 0
  fi

  local sa
  sa="$(iso_now)"
  update_status "$m" "$p" "$s" "running" "" "" "$sa" "" "false"
  echo "[$(iso_now)] START model=${m} pipeline=${p} seed=${s} epochs=${EPOCHS}"

  local start_epoch end_epoch wall rc
  start_epoch=$(date +%s)
  # Disable -e for the python invocation only so a single-run failure does NOT
  # abort the whole sweep. Capture rc explicitly.
  set +e
  "$PYTHON" -m revision.run_baselines \
    --model "$m" \
    --pipeline "$p" \
    --seed "$s" \
    --epochs "$EPOCHS" \
    --out-root "$OUT_ROOT" \
    > "${run_dir}/_stdout.log" 2> "${run_dir}/_stderr.log"
  rc=$?
  set -e
  end_epoch=$(date +%s)
  wall=$((end_epoch - start_epoch))
  local ea
  ea="$(iso_now)"

  if [[ $rc -eq 0 ]] && is_complete "$m" "$p" "$s"; then
    update_status "$m" "$p" "$s" "complete" "$rc" "$wall" "$sa" "$ea" "false"
    echo "[$(iso_now)] OK    model=${m} pipeline=${p} seed=${s} wall=${wall}s"
  else
    update_status "$m" "$p" "$s" "failed" "$rc" "$wall" "$sa" "$ea" "false"
    echo "[$(iso_now)] FAIL  model=${m} pipeline=${p} seed=${s} rc=${rc} wall=${wall}s -- see ${run_dir}/_stderr.log" >&2
  fi
  return 0
}
export -f is_complete iso_now update_status run_one
export OUT_ROOT STATUS_FILE LOCK_FILE EPOCHS PARALLEL

# -----------------------------------------------------------------------------
# Dry-run: print all 50 triples with their current would-run / complete status
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN — listing all 50 (model, pipeline, seed) triples (--parallel ${PARALLEL}, --epochs ${EPOCHS}):"
  for m in $MODELS; do
    for p in $PIPELINES; do
      for s in $SEEDS; do
        if is_complete "$m" "$p" "$s"; then
          echo "  model=${m} pipeline=${p} seed=${s} status=skip-already-complete"
        else
          echo "  model=${m} pipeline=${p} seed=${s} status=would-run"
        fi
      done
    done
  done
  exit 0
fi

# -----------------------------------------------------------------------------
# Initialize sweep_status.json header if it does not exist yet
# -----------------------------------------------------------------------------
if [[ ! -s "${STATUS_FILE}" ]]; then
  (
    flock -x 9
    python3 - "$(iso_now)" "$EPOCHS" "$PARALLEL" "$STATUS_FILE" <<'PY'
import json, os, sys, tempfile
sa, epochs, parallel, status_file = sys.argv[1:]
doc = {
    "started_at": sa,
    "epochs": int(epochs),
    "parallel": int(parallel),
    "runs": [],
    "all_complete": False,
    "completed_count": 0,
    "total_count": 50,
}
dirpath = os.path.dirname(status_file) or "."
fd, tmp = tempfile.mkstemp(prefix=".sweep_status.", suffix=".json", dir=dirpath)
with os.fdopen(fd, "w") as fh:
    json.dump(doc, fh, indent=2)
    fh.flush()
    os.fsync(fh.fileno())
os.rename(tmp, status_file)
PY
  ) 9>"${LOCK_FILE}"
fi

SWEEP_START=$(date +%s)

# -----------------------------------------------------------------------------
# Main dispatch — sequential (PARALLEL=1) or xargs -P 2 (PARALLEL=2)
# -----------------------------------------------------------------------------
if [[ "${PARALLEL}" -eq 1 ]]; then
  for m in $MODELS; do
    for p in $PIPELINES; do
      for s in $SEEDS; do
        run_one "$m" "$p" "$s"
      done
    done
  done
else
  # Build a (model, pipeline, seed) worklist for xargs. We dispatch ALL 50
  # triples; run_one() short-circuits on already-complete triples internally so
  # the work distribution stays simple.
  WORKLIST=$(mktemp)
  trap 'rm -f "$WORKLIST"' EXIT
  for m in $MODELS; do
    for p in $PIPELINES; do
      for s in $SEEDS; do
        printf "%s %s %s\n" "$m" "$p" "$s" >> "$WORKLIST"
      done
    done
  done
  # xargs -P 2 -L 1: at most 2 concurrent OS processes, one (m, p, s) per
  # invocation. NEVER replace this with multiprocessing.Pool (Pitfall 5).
  # bash -c receives the (m, p, s) triple as positional args $0 $1 $2.
  < "$WORKLIST" xargs -P 2 -L 1 bash -c 'run_one "$0" "$1" "$2"'
  rm -f "$WORKLIST"
  trap - EXIT
fi

SWEEP_END=$(date +%s)
SWEEP_WALL=$((SWEEP_END - SWEEP_START))

# -----------------------------------------------------------------------------
# Final summary: recompute counts and print a human-readable table
# -----------------------------------------------------------------------------
python3 - "$STATUS_FILE" "$SWEEP_WALL" <<'PY'
import json, os, sys
status_file, sweep_wall = sys.argv[1], int(sys.argv[2])
with open(status_file) as fh:
    doc = json.load(fh)
runs = doc["runs"]
complete = [r for r in runs if r["status"] == "complete"]
failed = [r for r in runs if r["status"] == "failed"]
skipped = [r for r in complete if r.get("skipped_already_done")]
print("=== Sweep complete ===")
print(f"Total: {doc['total_count']} | "
      f"Complete: {len(complete)} | "
      f"Failed: {len(failed)} | "
      f"Skipped (already done): {len(skipped)}")
def fmt_hms(s):
    s = int(s or 0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s"
print(f"Sweep wall time: {fmt_hms(sweep_wall)}")
# Per-model mean wall time (only for non-skipped completed runs).
by_model = {}
for r in complete:
    if r.get("skipped_already_done"):
        continue
    by_model.setdefault(r.get("model", "?"), []).append(r.get("wall_seconds") or 0)
if by_model:
    parts = []
    for mk, vals in sorted(by_model.items()):
        mean = sum(vals) / len(vals)
        parts.append(f"{mk}={mean/3600:.2f}h (n={len(vals)})")
    print("Per-model mean wall time (excl. skipped): " + ", ".join(parts))
if failed:
    print("FAILED runs (rerun the sweep to retry):")
    for r in failed:
        print(f"  model={r.get('model')} pipeline={r['pipeline']} "
              f"seed={r['seed']} rc={r.get('return_code')} "
              f"wall={r.get('wall_seconds')}s")
print(f"all_complete: {doc['all_complete']}")
PY

# Exit non-zero if not all runs are complete (so callers can detect the need
# for a retry invocation without parsing JSON).
if ! python3 -c "
import json, sys
with open('${STATUS_FILE}') as fh:
    doc = json.load(fh)
sys.exit(0 if doc.get('all_complete') else 1)
"; then
  echo "Sweep finished but not all runs are complete — re-run this script to retry failed runs." >&2
  exit 4
fi

exit 0
