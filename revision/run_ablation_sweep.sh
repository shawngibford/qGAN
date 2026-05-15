#!/usr/bin/env bash
# =============================================================================
# revision/run_ablation_sweep.sh  —  Phase 09.1-r1 multi-seed ablation driver
# =============================================================================
#
# Loops over the full 3 pipelines x 5 seeds = 15 training matrix for the
# preprocessing-ablation study (Plan 09.1-03 / ABL-02). Each pair is dispatched
# to the per-pair CLI driver at `revision/run_ablation.py`. The sweep is
# resumable: pairs whose run directory already contains the full five-file
# artifact bundle are skipped; failed or in-progress pairs are retried on the
# next invocation. Status is recorded to `revision/results/transform_ablation/
# sweep_status.json` atomically (tmp-file + os.rename, advisory flock guard).
#
# Resume semantics
# ----------------
#   For each (pipeline, seed) the script considers the pair "complete" iff
#   ALL five artifacts exist AND are non-empty:
#       config.yaml, checkpoint.pt, samples.npy, metrics.json, inverse_kwargs.npz
#   Re-invoking the script after a crash, reboot, or partial run skips already-
#   complete pairs and resumes from the first incomplete one. The CLI driver
#   itself is idempotent (overwrites a run_dir cleanly), so partially-written
#   directories are safe to retry.
#
# Parallelism guardrail (Assumption A3 — Mac thermal limit)
# ---------------------------------------------------------
#   --parallel 1  : sequential (default, ~45h)
#   --parallel 2  : two simultaneous python processes via `xargs -P 2 -L 1`
#                   (~24h, RESEARCH Q6 recommendation)
#   --parallel >=3: REJECTED with non-zero exit. Mac M-series throttles under
#                   sustained 3+ heavy CPU jobs; observed in v1.1 training.
#
# RESEARCH Pitfall 4 — never multiprocessing.Pool
# -----------------------------------------------
#   This script intentionally uses `xargs -P N`, NOT Python's
#   `multiprocessing.Pool`. `xargs` spawns fully independent OS processes; each
#   inherits a fresh interpreter + a fresh numpy RNG global state. With
#   `multiprocessing.Pool(fork)`, child workers share the parent's already-
#   warm numpy global RNG, which has corrupted at least one prior reproduction
#   attempt (Pitfall 4). Do NOT replace `xargs` with a `Pool`-based driver.
#
# Status file schema (revision/results/transform_ablation/sweep_status.json)
# --------------------------------------------------------------------------
#   {
#     "started_at": "2026-05-15T12:00:00Z",
#     "epochs": 1000,
#     "parallel": 2,
#     "runs": [
#       {"pipeline":"A","seed":42,"status":"complete",
#        "started_at":"...","ended_at":"...","wall_seconds":10800,
#        "return_code":0,"skipped_already_done":false}
#       ...
#     ],
#     "all_complete": false,
#     "completed_count": 0,
#     "total_count": 15
#   }
#
# Canonical invocation
# --------------------
#   # In tmux (preferred — survives ssh/terminal close):
#   tmux new -s ablation_sweep \
#     './revision/run_ablation_sweep.sh --parallel 2 2>&1 | \
#      tee revision/results/transform_ablation/sweep.log'
#
#   # Or nohup background:
#   nohup ./revision/run_ablation_sweep.sh --parallel 2 \
#     > revision/results/transform_ablation/sweep.log 2>&1 &
#
# CLI flags
# ---------
#   --parallel N   : 1 or 2 only (default 1)
#   --epochs M     : override per-pair epoch count (default 1000)
#   --dry-run      : print all 15 (pipeline, seed) pairs with their current
#                    would-run / already-complete status, then exit 0.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants — D-09.1-04 / D-09.1-06 / RESEARCH Q2
# -----------------------------------------------------------------------------
PIPELINES="A B C"
SEEDS="42 43 44 45 46"
EPOCHS=1000
OUT_ROOT="revision/results/transform_ablation"
STATUS_FILE="${OUT_ROOT}/sweep_status.json"
LOCK_FILE="${OUT_ROOT}/.status.lock"

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
  echo "       (Assumption A3). RESEARCH Q6 recommends --parallel 2 for the" >&2
  echo "       ~24h wall-time target. If you genuinely need more parallelism," >&2
  echo "       run on a non-thermal-constrained host and edit this guardrail" >&2
  echo "       intentionally — do not silently lift it." >&2
  exit 3
fi

if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [[ "$EPOCHS" -lt 1 ]]; then
  echo "ERROR: --epochs must be a positive integer (got: '${EPOCHS}')." >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/runs"

# -----------------------------------------------------------------------------
# is_complete <pipeline> <seed> -> exit 0 iff all 5 artifacts exist & non-empty
# -----------------------------------------------------------------------------
is_complete() {
  local p="$1" s="$2"
  local d="${OUT_ROOT}/runs/${p}/${s}"
  [[ -s "${d}/config.yaml" \
     && -s "${d}/checkpoint.pt" \
     && -s "${d}/samples.npy" \
     && -s "${d}/metrics.json" \
     && -s "${d}/inverse_kwargs.npz" ]]
}

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

# -----------------------------------------------------------------------------
# update_status <pipeline> <seed> <status> <return_code> <wall_seconds>
#               <started_at> <ended_at> <skipped_already_done>
#
# Atomically merges a single per-pair record into sweep_status.json.
#
# Protected by an advisory flock on ${LOCK_FILE} so two parallel xargs workers
# can safely write through this helper (T-09.1.03-01 mitigation).
# -----------------------------------------------------------------------------
update_status() {
  local p="$1" s="$2" st="$3" rc="$4" wall="$5" sa="$6" ea="$7" skipped="$8"

  # Acquire advisory lock. flock blocks until granted, then runs the helper.
  # Subshell + redirection makes flock automatically release on exit.
  (
    flock -x 9
    python3 - "$p" "$s" "$st" "$rc" "$wall" "$sa" "$ea" "$skipped" \
            "$PARALLEL" "$EPOCHS" "$STATUS_FILE" <<'PY'
import json, os, sys, tempfile

(p, s, st, rc, wall, sa, ea, skipped,
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
        "total_count": 15,
    }

# Always overwrite top-level config fields with the current invocation's values
# (a resumed sweep may use a different --parallel or --epochs).
doc["epochs"] = epochs
doc["parallel"] = parallel
doc.setdefault("total_count", 15)
doc.setdefault("runs", [])

# Find any existing record for this (p, s) and replace it; otherwise append.
runs = [r for r in doc["runs"] if not (r["pipeline"] == p and r["seed"] == s)]
record = {
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
# Keep runs sorted by (pipeline, seed) for human readability.
runs.sort(key=lambda r: (r["pipeline"], r["seed"]))
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
# run_one <pipeline> <seed>
#
# Invokes the per-pair CLI, records status, never aborts the sweep on failure.
# Designed to be safe to call from both sequential and parallel (xargs) modes.
# -----------------------------------------------------------------------------
run_one() {
  local p="$1" s="$2"
  local run_dir="${OUT_ROOT}/runs/${p}/${s}"
  mkdir -p "${run_dir}"

  if is_complete "$p" "$s"; then
    local sa
    sa="$(iso_now)"
    update_status "$p" "$s" "complete" "0" "0" "$sa" "$sa" "true"
    echo "[$(iso_now)] SKIP  pipeline=${p} seed=${s} (already complete)"
    return 0
  fi

  local sa
  sa="$(iso_now)"
  update_status "$p" "$s" "running" "" "" "$sa" "" "false"
  echo "[$(iso_now)] START pipeline=${p} seed=${s} epochs=${EPOCHS}"

  local start_epoch end_epoch wall rc
  start_epoch=$(date +%s)
  # Disable -e for the python invocation only so a single-pair failure does NOT
  # abort the whole sweep. Capture rc explicitly.
  set +e
  python -m revision.run_ablation \
    --pipeline "$p" \
    --seed "$s" \
    --epochs "$EPOCHS" \
    > "${run_dir}/_stdout.log" 2> "${run_dir}/_stderr.log"
  rc=$?
  set -e
  end_epoch=$(date +%s)
  wall=$((end_epoch - start_epoch))
  local ea
  ea="$(iso_now)"

  if [[ $rc -eq 0 ]] && is_complete "$p" "$s"; then
    update_status "$p" "$s" "complete" "$rc" "$wall" "$sa" "$ea" "false"
    echo "[$(iso_now)] OK    pipeline=${p} seed=${s} wall=${wall}s"
  else
    update_status "$p" "$s" "failed" "$rc" "$wall" "$sa" "$ea" "false"
    echo "[$(iso_now)] FAIL  pipeline=${p} seed=${s} rc=${rc} wall=${wall}s -- see ${run_dir}/_stderr.log" >&2
  fi
  return 0
}
export -f is_complete iso_now update_status run_one
export OUT_ROOT STATUS_FILE LOCK_FILE EPOCHS PARALLEL

# -----------------------------------------------------------------------------
# Dry-run: print all 15 pairs with their current would-run / complete status
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN — listing all 15 (pipeline, seed) pairs (--parallel ${PARALLEL}, --epochs ${EPOCHS}):"
  for p in $PIPELINES; do
    for s in $SEEDS; do
      if is_complete "$p" "$s"; then
        echo "  pipeline=${p} seed=${s} status=skip-already-complete"
      else
        echo "  pipeline=${p} seed=${s} status=would-run"
      fi
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
    "total_count": 15,
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
  for p in $PIPELINES; do
    for s in $SEEDS; do
      run_one "$p" "$s"
    done
  done
else
  # Build a (pipeline, seed) worklist for xargs. We dispatch ALL 15 pairs;
  # run_one() short-circuits on already-complete pairs internally so the work
  # distribution stays simple.
  WORKLIST=$(mktemp)
  trap 'rm -f "$WORKLIST"' EXIT
  for p in $PIPELINES; do
    for s in $SEEDS; do
      printf "%s %s\n" "$p" "$s" >> "$WORKLIST"
    done
  done
  # xargs -P 2 -L 1: at most 2 concurrent OS processes, one (p, s) per invocation.
  # NEVER replace this with multiprocessing.Pool (RESEARCH Pitfall 4).
  # bash -c receives the (p, s) pair as positional args $0 $1.
  < "$WORKLIST" xargs -P 2 -L 1 bash -c 'run_one "$0" "$1"'
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
# Per-pipeline mean wall time (only for non-skipped completed pairs).
by_pipeline = {}
for r in complete:
    if r.get("skipped_already_done"):
        continue
    by_pipeline.setdefault(r["pipeline"], []).append(r.get("wall_seconds") or 0)
if by_pipeline:
    parts = []
    for p, vals in sorted(by_pipeline.items()):
        mean = sum(vals) / len(vals)
        parts.append(f"{p}={mean/3600:.2f}h (n={len(vals)})")
    print("Per-pipeline mean wall time (excl. skipped): " + ", ".join(parts))
if failed:
    print("FAILED pairs (rerun the sweep to retry):")
    for r in failed:
        print(f"  pipeline={r['pipeline']} seed={r['seed']} "
              f"rc={r.get('return_code')} wall={r.get('wall_seconds')}s")
print(f"all_complete: {doc['all_complete']}")
PY

# Exit non-zero if not all pairs are complete (so callers can detect the need
# for a retry invocation without parsing JSON).
if ! python3 -c "
import json, sys
with open('${STATUS_FILE}') as fh:
    doc = json.load(fh)
sys.exit(0 if doc.get('all_complete') else 1)
"; then
  echo "Sweep finished but not all pairs are complete — re-run this script to retry failed pairs." >&2
  exit 4
fi

exit 0
