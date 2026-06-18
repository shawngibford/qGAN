#!/usr/bin/env bash
# =============================================================================
# revision/run_matched2000_sweep.sh  —  Phase 14 matched-2000ep sweep driver
# =============================================================================
#
# Copied END-TO-END from revision/run_ansatz_sweep.sh (the proven Phase-13
# skeleton); ONLY the model matrix, the 5-file artifact-bundle definition, the
# OUT_ROOT, the per-run CLI module, and the post-run strict-accept pass were
# changed. The thermal guardrail, xargs -P 2 -L 1 dispatch, resumable
# is_complete() skip-already-done semantics, atomic flock'd sweep_status.json
# writer, and ./qgan_env/bin/python direct invocation are VERBATIM.
#
# Loops over the 9 models x 5 seeds = 45 training matrix at a MATCHED 2000-epoch
# budget (D-14-08) so the paper's numbers are coherent and fair:
#     models = iqp_sel_55_repro V1 V2 V3 wgan_mlp wgan_cnn wgan_lstm vae ar
#     seeds  = 42 43 44 45 46          EPOCHS = 2000 (matched for ALL)
# Each (model, seed) is dispatched to revision.run_matched2000. The
# iqp_sel_55_repro run is the NON-load-bearing reproduction instance, tagged
# distinctly from the frozen-checkpoint headline (D-14-10).
#
# Resume semantics
# ----------------
#   A (model, seed) pair is "complete" iff ALL five artifacts exist AND are
#   non-empty AND the D-14-13 strict accept gate PASSES for that pair:
#       config.yaml, checkpoint.{pt|npz}, samples.npy, metrics.json,
#       inverse_kwargs.npz   (+ strict-accept gate green)
#   Re-invoking after a crash/reboot/partial run skips already-accepted pairs
#   and resumes from the first incomplete one. The CLI driver is idempotent
#   (shutil.rmtree overwrites a run_dir cleanly), so partially-written or
#   gate-rejected directories are safe to retry. is_complete() requiring the
#   strict gate (not just file presence) prevents silent mixed-budget /
#   wrong-hash contamination from looking "done" (D-14-13).
#
# Tiering (D-14-14): T1 (Plan 01 recovery, already done) is NOT in this
#   matrix. T2 (reproduction + baseline-bearing: iqp_sel_55_repro, wgan_*,
#   vae, ar) and T3 (ansatz V1/V2/V3) are each independently acceptable so the
#   sweep is run-to-completion with no hard time-box (correctness over speed).
#
# Parallelism guardrail (Mac thermal limit, D-14-12)
# --------------------------------------------------
#   --parallel 1  : sequential (default)
#   --parallel 2  : two simultaneous python processes via `xargs -P 2 -L 1`
#   --parallel >=3: REJECTED with non-zero exit. Mac M-series throttles under
#                   sustained 3+ heavy CPU jobs; observed in v1.1 training.
#
# RESEARCH Pitfall 5 — never an in-process Python worker pool (D-10-24, LOCKED)
# ----------------------------------------------------------------------------
#   This script intentionally uses `xargs -P N`, NOT Python's in-process
#   parallel worker-pool API. `xargs` spawns fully independent OS processes;
#   each inherits a fresh interpreter + a fresh numpy RNG global state. A
#   forked in-process worker pool instead shares the parent's already-warm
#   numpy global RNG, which has corrupted at least one prior reproduction
#   attempt. Do NOT replace `xargs` with a pool-based driver. There is zero
#   in-process Python parallelism anywhere in this sweep or the driver.
#
# Backend lock (D-14-11): default.qubit + diff_method="backprop". NO
#   lightning.qubit. PennyLane sims are CPU-only — the quantum rows are
#   CORRECTLY CPU; classical rows must not silently fall back. Every per-run
#   emits a device/dtype manifest and hard-asserts the actual backend.
#
# Status file schema (revision/results/matched2000/sweep_status.json)
# -------------------------------------------------------------------
#   {
#     "started_at": "...", "epochs": 2000, "parallel": 2,
#     "runs": [ {"model":"vae","seed":42,"status":"complete",
#                "started_at":"...","ended_at":"...","wall_seconds":3600,
#                "return_code":0,"skipped_already_done":false} ... ],
#     "all_complete": false, "completed_count": 0, "total_count": 45
#   }
#
# Canonical invocation
# --------------------
#   # In tmux (preferred — survives ssh/terminal close):
#   tmux new -s m2k_sweep \
#     './revision/run_matched2000_sweep.sh --parallel 2 2>&1 | \
#      tee revision/results/matched2000/sweep.log'
#
#   # Or nohup background:
#   nohup ./revision/run_matched2000_sweep.sh --parallel 2 \
#     > revision/results/matched2000/sweep.log 2>&1 &
#
# CLI flags
# ---------
#   --parallel N   : 1 or 2 only (default 1)
#   --epochs M     : override per-run epoch count (default 2000)
#   --models "..." : space-separated subset (default: all 9) — enables
#                    tier-at-a-time runs (e.g. --models "iqp_sel_55_repro
#                    wgan_mlp wgan_cnn wgan_lstm vae ar" for T2 first).
#   --dry-run      : print all pairs with their current would-run /
#                    already-complete status, then exit 0.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants — Phase-14 matched-2000ep matrix (D-14-08/10/14)
# -----------------------------------------------------------------------------
MODELS="iqp_sel_55_repro V1 V2 V3 wgan_mlp wgan_cnn wgan_lstm vae ar"
SEEDS="42 43 44 45 46"
EPOCHS=2000
OUT_ROOT="revision/results/matched2000"
STATUS_FILE="${OUT_ROOT}/sweep_status.json"
LOCK_FILE="${OUT_ROOT}/.status.lock"
RUN_MODULE="revision.run_matched2000"

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
  sed -n '2,90p' "$0"
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
    --models)
      MODELS="${2:-}"
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
      echo "Usage: $0 [--parallel 1|2] [--epochs N] [--models \"...\"] [--dry-run]" >&2
      exit 2
      ;;
  esac
done

# Compute TOTAL_COUNT from the (possibly subset) model list x seeds.
_n_models=$(printf "%s\n" $MODELS | grep -c .)
_n_seeds=$(printf "%s\n" $SEEDS | grep -c .)
TOTAL_COUNT=$(( _n_models * _n_seeds ))

# -----------------------------------------------------------------------------
# Guardrail: --parallel must be 1 or 2 (Mac thermal, D-14-12)
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
# is_complete <model> <seed>
#   -> exit 0 iff all 5 artifacts exist & non-empty AND the D-14-13 strict
#      accept gate PASSES (file-presence alone is NOT sufficient — a
#      wrong-hash / mixed-budget / device-failed bundle must NOT look "done",
#      D-14-13). AR checkpoints to checkpoint.npz; the rest to checkpoint.pt.
# -----------------------------------------------------------------------------
is_complete() {
  local m="$1" s="$2"
  local d="${OUT_ROOT}/runs/${m}/${s}"
  local ckpt="${d}/checkpoint.pt"
  if [[ "$m" == "ar" ]]; then
    ckpt="${d}/checkpoint.npz"
  fi
  [[ -s "${d}/config.yaml" \
     && -s "${ckpt}" \
     && -s "${d}/samples.npy" \
     && -s "${d}/metrics.json" \
     && -s "${d}/inverse_kwargs.npz" ]] || return 1
  # Strict accept gate (D-14-13) — explicit-raise; non-zero exit on rejection.
  "$PYTHON" -m "${RUN_MODULE}" --accept --model "$m" --seed "$s" \
    --out-root "$OUT_ROOT" >/dev/null 2>&1
}

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

# -----------------------------------------------------------------------------
# update_status <model> <seed> <status> <return_code> <wall_seconds>
#               <started_at> <ended_at> <skipped_already_done>
#
# Atomically merges a single per-run record into sweep_status.json.
# Protected by an advisory flock on ${LOCK_FILE} so two parallel xargs workers
# can safely write through this helper.
# -----------------------------------------------------------------------------
update_status() {
  local m="$1" s="$2" st="$3" rc="$4" wall="$5" sa="$6" ea="$7" skipped="$8"

  (
    flock -x 9
    python3 - "$m" "$s" "$st" "$rc" "$wall" "$sa" "$ea" "$skipped" \
            "$PARALLEL" "$EPOCHS" "$STATUS_FILE" "$TOTAL_COUNT" <<'PY'
import json, os, sys, tempfile

(m, s, st, rc, wall, sa, ea, skipped,
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

doc["epochs"] = epochs
doc["parallel"] = parallel
doc["total_count"] = total_count
doc.setdefault("runs", [])

runs = [
    r for r in doc["runs"]
    if not (r.get("model") == m and r["seed"] == s)
]
record = {
    "model": m,
    "seed": s,
    "status": st,
    "started_at": sa,
    "ended_at": ea,
    "wall_seconds": wall,
    "return_code": rc,
    "skipped_already_done": skipped_bool,
}
runs.append(record)
runs.sort(key=lambda r: (r.get("model", ""), r["seed"]))
doc["runs"] = runs

completed = sum(1 for r in runs if r["status"] == "complete")
doc["completed_count"] = completed
doc["all_complete"] = (completed == doc["total_count"])

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
# run_one <model> <seed>
#
# Invokes the per-run CLI, records status, never aborts the sweep on failure.
# Safe to call from both sequential and parallel (xargs) modes.
# -----------------------------------------------------------------------------
run_one() {
  local m="$1" s="$2"
  local run_dir="${OUT_ROOT}/runs/${m}/${s}"
  mkdir -p "${run_dir}"

  if is_complete "$m" "$s"; then
    local sa
    sa="$(iso_now)"
    update_status "$m" "$s" "complete" "0" "0" "$sa" "$sa" "true"
    echo "[$(iso_now)] SKIP  model=${m} seed=${s} (already complete + accepted)"
    return 0
  fi

  local sa
  sa="$(iso_now)"
  update_status "$m" "$s" "running" "" "" "$sa" "" "false"
  echo "[$(iso_now)] START model=${m} seed=${s} epochs=${EPOCHS}"

  local start_epoch end_epoch wall rc
  start_epoch=$(date +%s)
  # Disable -e for the python invocation only so a single-run failure does NOT
  # abort the whole sweep. Capture rc explicitly.
  set +e
  "$PYTHON" -m "${RUN_MODULE}" \
    --model "$m" \
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

  # A run is "complete" only if it trained AND passes the strict accept gate.
  if [[ $rc -eq 0 ]] && is_complete "$m" "$s"; then
    update_status "$m" "$s" "complete" "$rc" "$wall" "$sa" "$ea" "false"
    echo "[$(iso_now)] OK    model=${m} seed=${s} wall=${wall}s (accepted)"
  else
    update_status "$m" "$s" "failed" "$rc" "$wall" "$sa" "$ea" "false"
    echo "[$(iso_now)] FAIL  model=${m} seed=${s} rc=${rc} wall=${wall}s -- see ${run_dir}/_stderr.log (train rc or strict-accept gate)" >&2
  fi
  return 0
}
export -f is_complete iso_now update_status run_one
export OUT_ROOT STATUS_FILE LOCK_FILE EPOCHS PARALLEL TOTAL_COUNT RUN_MODULE

# -----------------------------------------------------------------------------
# Dry-run: print all pairs with their current would-run / complete status
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN — listing all ${TOTAL_COUNT} (model, seed) pairs (--parallel ${PARALLEL}, --epochs ${EPOCHS}):"
  for m in $MODELS; do
    for s in $SEEDS; do
      if is_complete "$m" "$s"; then
        echo "  model=${m} seed=${s} status=skip-already-complete"
      else
        echo "  model=${m} seed=${s} status=would-run"
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
  for m in $MODELS; do
    for s in $SEEDS; do
      run_one "$m" "$s"
    done
  done
else
  # Build a (model, seed) worklist for xargs. We dispatch ALL pairs;
  # run_one() short-circuits on already-complete pairs internally so the work
  # distribution stays simple.
  WORKLIST=$(mktemp)
  trap 'rm -f "$WORKLIST"' EXIT
  for m in $MODELS; do
    for s in $SEEDS; do
      printf "%s %s\n" "$m" "$s" >> "$WORKLIST"
    done
  done
  # xargs -P 2 -L 1: at most 2 concurrent OS processes, one (m, s) per
  # invocation. NEVER replace this with an in-process Python worker pool
  # (Pitfall 5 — shared warm numpy RNG corrupts reproductions).
  # bash -c receives the (m, s) pair as positional args $0 $1.
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
print("=== Matched-2000ep sweep complete ===")
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
by_model = {}
for r in complete:
    if r.get("skipped_already_done"):
        continue
    by_model.setdefault(r.get("model", "?"), []).append(r.get("wall_seconds") or 0)
if by_model:
    parts = []
    for vk, vals in sorted(by_model.items()):
        mean = sum(vals) / len(vals)
        parts.append(f"{vk}={mean/3600:.2f}h (n={len(vals)})")
    print("Per-model mean wall time (excl. skipped): " + ", ".join(parts))
if failed:
    print("FAILED runs (rerun the sweep to retry):")
    for r in failed:
        print(f"  model={r.get('model')} seed={r['seed']} "
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
