#!/usr/bin/env bash
set -euo pipefail

# Defaults
ENV_NAME="micro_bf39"
PY_BIN=""                         # if empty, we'll activate the conda env; otherwise use this python
NICE_LEVEL=10
IONICE_CLASS=2                    # 2=best-effort (normal); use --idle-io to switch to class 3
IONICE_N=5
TMPDIR_DEFAULT="${HOME}/tmp_nd2"
LOG=""
FOVS=""
SCRIPT="/home/kklabs/microscopy_processing/max_project_nd2_stream.py"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] <input.nd2> <output.ome.tif>

Options:
  --env NAME           Conda env to activate (default: ${ENV_NAME})
  --python PATH        Use this Python instead of activating an env
  --tmpdir DIR         Directory for memmap/state (default: ${TMPDIR_DEFAULT})
  --fovs N             Process at most N FOVs (optional)
  --log FILE           Log file path (default: ~/logs/<output_basename>.log)
  --nice N             nice level (default: ${NICE_LEVEL})
  --idle-io            Use ionice class 3 (idle) instead of class 2 (best-effort)
  -h, --help           Show this help

Examples:
  $(basename "$0") /mnt/staging/round3.nd2 ~/max_round3.ome.tif
  $(basename "$0") --tmpdir ~/tmp_nd2 --fovs 500 /mnt/staging/round3.nd2 ~/max_round3.ome.tif
  $(basename "$0") --python /home/kklabs/miniconda3/envs/micro_bf39/bin/python \\
                   /mnt/staging/round3.nd2 ~/max_round3.ome.tif
EOF
}

# Parse args
TMPDIR="${TMPDIR_DEFAULT}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)       ENV_NAME="$2"; shift 2 ;;
    --python)    PY_BIN="$2"; shift 2 ;;
    --tmpdir)    TMPDIR="$2"; shift 2 ;;
    --fovs)      FOVS="$2"; shift 2 ;;
    --log)       LOG="$2"; shift 2 ;;
    --nice)      NICE_LEVEL="$2"; shift 2 ;;
    --idle-io)   IONICE_CLASS=3; IONICE_N=7; shift ;;
    -h|--help)   usage; exit 0 ;;
    --)          shift; break ;;
    -*)
      echo "Unknown option: $1" >&2
      usage; exit 2 ;;
    *) break ;;
  esac
done

# Positional args
if [[ $# -lt 2 ]]; then
  usage; exit 2
fi
IN="$1"; OUT="$2"

# Sanity checks
[[ -f "$IN" ]] || { echo "Input not found: $IN" >&2; exit 1; }
mkdir -p "$TMPDIR" "$HOME/logs"

# Default log name if not given
if [[ -z "${LOG}" ]]; then
  base="$(basename "$OUT")"
  LOG="${HOME}/logs/${base%.ome.tif}_nd2_run.log"
fi

# Choose Python
if [[ -z "${PY_BIN}" ]]; then
  # activate conda env
  if command -v conda >/dev/null 2>&1; then
    # shell hook ensures 'conda activate' works in non-interactive script
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    PY_BIN="$(command -v python)"
  else
    echo "conda not found; please install it or pass --python /path/to/python" >&2
    exit 1
  fi
fi

# Build command
CMD=( "$PY_BIN" -u "$SCRIPT" "$IN" "$OUT" )
# The Python script you’re using defaults tmpdir to ~/tmp_nd2,
# but we pass it explicitly so it’s obvious:
CMD+=( "--tmpdir" "$TMPDIR" )
if [[ -n "${FOVS}" ]]; then
  CMD+=( "--fovs" "$FOVS" )
fi

echo "Launching:"
echo "  tmpdir : $TMPDIR"
echo "  log    : $LOG"
echo "  python : $PY_BIN"
echo "  cmd    : ${CMD[*]}"
echo

# Run with nohup / nice / ionice
# Use 'disown' so it survives shell exit (bash builtin).
nohup nice -n "${NICE_LEVEL}" ionice -c "${IONICE_CLASS}" -n "${IONICE_N}" \
  "${CMD[@]}" > "$LOG" 2>&1 & disown

PID=$!
echo "Started PID $PID"
echo "Tail the log: tail -f \"$LOG\""
