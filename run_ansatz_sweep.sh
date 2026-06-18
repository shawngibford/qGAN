#!/usr/bin/env bash
# =============================================================================
# revision/run_ansatz_sweep.sh  —  Phase 13 ARCH-01 quantum-ansatz sweep driver
# =============================================================================
#
# Loops over the 2 variants x 5 seeds = 10 training matrix for the ARCH-01/02
# quantum-vs-quantum ansatz study (Plan 13-02). Each (variant, seed) is
# dispatched to the per-run CLI driver at `revision/run_ansatz.py`. V1
# (depth-4, range, 75 params) is NOT in this matrix — its 5-seed final metrics
# are REUSED from revision/results/transform_ablation/runs/B/{42..46} with NO
# recompute (D-13-01). The sweep is resumable: (variant, seed) pairs whose run
# directory already contains the full five-file artifact bundle are skipped;
# failed or in-progress pairs are retried on the next invocation. Status is
# recorded to `revision/results/ansatz/sweep_status.json` atomically (tmp-file
# + os.rename, advisory flock guard).
#
# Resume semantics
# ----------------
#   For each (variant, seed) the script considers the pair "complete" iff ALL
#   five artifacts exist AND are non-empty:
#       config.yaml, checkpoint.pt, samples.npy, metrics.json,
#       inverse_kwargs.npz
#   Re-invoking the script after a crash, reboot, or partial run skips
#   already-complete pairs and resumes from the first incomplete one. The CLI
#   driver itself is idempotent (shutil.rmtree overwrites a run_dir cleanly),
#   so partially-written directories are safe to retry (T-13-07).
#
# Parallelism guardrail (Mac thermal limit)
# -----------------------------------------
#   --parallel 1  : sequential (default)
#   --parallel 2  : two simultaneous python processes via `xargs -P 2 -L 1`
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
# Status file schema (revision/results/ansatz/sweep_status.json)
# --------------------------------------------------------------
#   {
#     "started_at": "2026-05-19T12:00:00Z",
#     "epochs": 1000,
#     "parallel": 2,
#     "runs": [
#       {"variant":"V2","seed":42,"status":"complete",
#        "started_at":"...","ended_at":"...","wall_seconds":3600,
#        "return_code":0,"skipped_already_done":false}
#       ...
#     ],
#     "all_complete": false,
#     "completed_count": 0,
#     "total_count": 10
#   }
#
# Canonical invocation
# --------------------
#   # In tmux (preferred — survives ssh/terminal close):
#   tmux new -s ansatz_sweep \
#     './revision/run_ansatz_sweep.sh --parallel 2 2>&1 | \
#      tee revision/results/ansatz/sweep.log'
#
#   # Or nohup background:
#   nohup ./revision/run_ansatz_sweep.sh --parallel 2 \
#     > revision/results/ansatz/sweep.log 2>&1 &
#
# CLI flags
# ---------
#   --parallel N   : 1 or 2 only (default 1)
#   --epochs M     : override per-run epoch count (default 1000)
#   --dry-run      : print all 10 (variant, seed) pairs with their current
#                    would-run / already-complete status, then exit 0.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants — ARCH-01 matrix (D-13-01: V1 reused, NOT swept)
# -----------------------------------------------------------------------------
VARIANTS="V2 V3"
SEEDS="42 43 44 45 46"
EPOCHS=1000
OUT_ROOT="revision/results/ansatz"
STATUS_FILE="${OUT_ROOT}/sweep_status.json"
LOCK_FILE="${OUT_ROOT}/.status.lock"
TOTAL_COUNT=10

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
# Guardrail: --parallel must be 1 or 2 (Mac thermal)
# -----------------------------------------------------------------------------
if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [[ "$PARALLEL" -lt 1 ]] || [[ "$PARALLEL" -gt 2 ]]; then
  echo "ERROR: --parallel must be 1 or 2 (got: '${PARALLEL}')." >&2
  echo "       Mac M-series thermal throttling under sustained 3+ heavy jobs." >&2
  echo "       --parallel 2 is recommended for the wall-time target. If you" >&2
  echo "       genuinely need more parallelism, run on a non-thermal-" >&2
  echo "       constrained host and edit this guardrail intentionally — do" >&2
  echo "       not silently lift it." >&2
  exit 3
fi

if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [[ "$EPOCHS" -lt 1 ]]; then
  echo "ERROR: --epochs must be a positive integer (got: '${EPOCHS}')." >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/runs"

# -----------------------------------------------------------------------------
# is_complete <variant> <seed>
#   -> exit 0 iff all 5 artifacts exist & non-empty
#
# Keyed on the (variant, seed) 2-tuple (run_baselines keyed on the
# (model, pipeline, seed) 3-tuple). All quantum variants checkpoint to
# checkpoint.pt (no AR .npz special-case here).
# -----------------------------------------------------------------------------
is_complete() {
  local v="$1" s="$2"
  local d="${OUT_ROOT}/runs/${v}/${s}"
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
# update_status <variant> <seed> <status> <return_code> <wall_seconds>
#               <started_at> <ended_at> <skipped_already_done>
#
# Atomically merges a single per-run record into sweep_status.json.
#
# Protected by an advisory flock on ${LOCK_FILE} so two parallel xargs workers
# can safely write through this helper (T-13-06 mitigation).
# -----------------------------------------------------------------------------
update_status() {
  local v="$1" s="$2" st="$3" rc="$4" wall="$5" sa="$6" ea="$7" skipped="$8"

  # Acquire advisory lock. flock blocks until granted, then runs the helper.
  # Subshell + redirection makes flock automatically release on exit.
  (
    flock -x 9
    python3 - "$v" "$s" "$st" "$rc" "$wall" "$sa" "$ea" "$skipped" \
            "$PARALLEL" "$EPOCHS" "$STATUS_FILE" "$TOTAL_COUNT" <<'PY'
import json, os, sys, tempfile

(v, s, st, rc, wall, sa, ea, skipped,
 parallel, epochs, status_file, total_count) = sys.argv[1:]
s = int(s)
rc = int(rc) if rc != "" else None
wall = int(wall) if wall != "" else None
parallel = int(parallel)
epochs = int(epochs)
total_count = int(total_count)
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
        "total_count": total_count,
    }

# Always overwrite top-level config fields with the current invocation's values
# (a resumed sweep may use a different --parallel or --epochs).
doc["epochs"] = epochs
doc["parallel"] = parallel
doc["total_count"] = total_count
doc.setdefault("runs", [])

# Find any existing record for this (v, s) and replace it; otherwise append.
runs = [
    r for r in doc["runs"]
    if not (r.get("variant") == v and r["seed"] == s)
]
record = {
    "variant": v,
    "seed": s,
    "status": st,
    "started_at": sa,
    "ended_at": ea,
    "wall_seconds": wall,
    "return_code": rc,
    "skipped_already_done": skipped_bool,
}
runs.append(record)
# Keep runs sorted by (variant, seed) for human readability.
runs.sort(key=lambda r: (r.get("variant", ""), r["seed"]))
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
# run_one <variant> <seed>
#
# Invokes the per-run CLI, records status, never aborts the sweep on failure.
# Designed to be safe to call from both sequential and parallel (xargs) modes.
# -----------------------------------------------------------------------------
run_one() {
  local v="$1" s="$2"
  local run_dir="${OUT_ROOT}/runs/${v}/${s}"
  mkdir -p "${run_dir}"

  if is_complete "$v" "$s"; then
    local sa
    sa="$(iso_now)"
    update_status "$v" "$s" "complete" "0" "0" "$sa" "$sa" "true"
    echo "[$(iso_now)] SKIP  variant=${v} seed=${s} (already complete)"
    return 0
  fi

  local sa
  sa="$(iso_now)"
  update_status "$v" "$s" "running" "" "" "$sa" "" "false"
  echo "[$(iso_now)] START variant=${v} seed=${s} epochs=${EPOCHS}"

  local start_epoch end_epoch wall rc
  start_epoch=$(date +%s)
  # Disable -e for the python invocation only so a single-run failure does NOT
  # abort the whole sweep. Capture rc explicitly.
  set +e
  "$PYTHON" -m revision.run_ansatz \
    --variant "$v" \
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

  if [[ $rc -eq 0 ]] && is_complete "$v" "$s"; then
    update_status "$v" "$s" "complete" "$rc" "$wall" "$sa" "$ea" "false"
    echo "[$(iso_now)] OK    variant=${v} seed=${s} wall=${wall}s"
  else
    update_status "$v" "$s" "failed" "$rc" "$wall" "$sa" "$ea" "false"
    echo "[$(iso_now)] FAIL  variant=${v} seed=${s} rc=${rc} wall=${wall}s -- see ${run_dir}/_stderr.log" >&2
  fi
  return 0
}
export -f is_complete iso_now update_status run_one
export OUT_ROOT STATUS_FILE LOCK_FILE EPOCHS PARALLEL TOTAL_COUNT

# -----------------------------------------------------------------------------
# Dry-run: print all 10 pairs with their current would-run / complete status
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN — listing all ${TOTAL_COUNT} (variant, seed) pairs (--parallel ${PARALLEL}, --epochs ${EPOCHS}):"
  for v in $VARIANTS; do
    for s in $SEEDS; do
      if is_complete "$v" "$s"; then
        echo "  variant=${v} seed=${s} status=skip-already-complete"
      else
        echo "  variant=${v} seed=${s} status=would-run"
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
    python3 - "$(iso_now)" "$EPOCHS" "$PARALLEL" "$STATUS_FILE" "$TOTAL_COUNT" <<'PY'
import json, os, sys, tempfile
sa, epochs, parallel, status_file, total_count = sys.argv[1:]
doc = {
    "started_at": sa,
    "epochs": int(epochs),
    "parallel": int(parallel),
    "runs": [],
    "all_complete": False,
    "completed_count": 0,
    "total_count": int(total_count),
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
  for v in $VARIANTS; do
    for s in $SEEDS; do
      run_one "$v" "$s"
    done
  done
else
  # Build a (variant, seed) worklist for xargs. We dispatch ALL 10 pairs;
  # run_one() short-circuits on already-complete pairs internally so the work
  # distribution stays simple.
  WORKLIST=$(mktemp)
  trap 'rm -f "$WORKLIST"' EXIT
  for v in $VARIANTS; do
    for s in $SEEDS; do
      printf "%s %s\n" "$v" "$s" >> "$WORKLIST"
    done
  done
  # xargs -P 2 -L 1: at most 2 concurrent OS processes, one (v, s) per
  # invocation. NEVER replace this with multiprocessing.Pool (Pitfall 5).
  # bash -c receives the (v, s) pair as positional args $0 $1.
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
import json, sys
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
# Per-variant mean wall time (only for non-skipped completed runs).
by_variant = {}
for r in complete:
    if r.get("skipped_already_done"):
        continue
    by_variant.setdefault(r.get("variant", "?"), []).append(r.get("wall_seconds") or 0)
if by_variant:
    parts = []
    for vk, vals in sorted(by_variant.items()):
        mean = sum(vals) / len(vals)
        parts.append(f"{vk}={mean/3600:.2f}h (n={len(vals)})")
    print("Per-variant mean wall time (excl. skipped): " + ", ".join(parts))
if failed:
    print("FAILED runs (rerun the sweep to retry):")
    for r in failed:
        print(f"  variant={r.get('variant')} seed={r['seed']} "
              f"rc={r.get('return_code')} wall={r.get('wall_seconds')}s")
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
