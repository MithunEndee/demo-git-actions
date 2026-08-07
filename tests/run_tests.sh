#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--token <api_token>] [--base-url <url>] [--unit] [--integration] [--clean] [-- <pytest args>]

Run the endee-llamaindex test suite. It creates a venv, installs dependencies,
sets environment variables, and invokes pytest.

Two kinds of tests, selected automatically or explicitly:
  unit         Fast, no network. The endee client is mocked. No token needed.
  integration  Requires a live or local Endee server and ENDEE_API_TOKEN.
               Skips automatically if no token is supplied.

Options:
  --token <api_token>    Sets ENDEE_API_TOKEN and enables integration tests.
  --base-url <url>       Sets ENDEE_BASE_URL (e.g. for a local server).
  --unit                 Run only unit tests (-m unit).
  --integration          Run only integration tests (-m integration). Requires --token.
  --clean                Recreate the venv before the run, then remove it after.
  --                     Pass remaining arguments to pytest (paths, -k, -x, -s, etc.).
  -h, --help             Show this help message and exit.

Environment overrides:
  PYTHON_BIN    Python interpreter to build the venv with (defaults to "python3"
                on PATH). Example: PYTHON_BIN=python3.12 $0 --unit

Examples:
  # Unit tests only (fast, no server needed): the common local/CI case
  $0 --unit

  # Full suite (unit + integration) against a live server
  $0 --token <api_token>

  # Full suite against a local server
  $0 --token <api_token> --base-url http://localhost:8080/api/v2

  # Integration tests only
  $0 --token <api_token> --integration

  # Fresh environment: recreate the venv before the run and clean up after
  $0 --unit --clean

  # Run a single test file
  $0 --unit -- tests/test_unit.py

  # Run tests matching a keyword pattern
  $0 --token <api_token> -- -k test_retrieval

  # Stop on first failure and show captured output
  $0 --unit -- -x -s
EOF
  exit "${1:-1}"
}

# Strip --clean from any position before standard flag parsing begins
CLEAN=0
FILTERED_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--clean" ]]; then
    CLEAN=1
  else
    FILTERED_ARGS+=("$arg")
  fi
done
set -- "${FILTERED_ARGS[@]+"${FILTERED_ARGS[@]}"}"

# Parse remaining flags
TOKEN=""
BASE_URL=""
MARKER=""
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token)        TOKEN="$2";           shift 2 ;;
    --base-url)     BASE_URL="$2";        shift 2 ;;
    --unit)         MARKER="unit";        shift ;;
    --integration)  MARKER="integration"; shift ;;
    --)             shift; PYTEST_ARGS+=("$@"); break ;;
    -h|--help)      usage 0 ;;
    *)              echo "Unknown argument: $1"; usage ;;
  esac
done

if [[ "$MARKER" == "integration" && -z "$TOKEN" ]]; then
  echo "Error: --integration requires --token"
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
VENV_DIR="$REPO_ROOT/venv"

cd "$REPO_ROOT"

if [[ "$CLEAN" -eq 1 && -d "$VENV_DIR" ]]; then
  echo "Removing existing virtual environment ..."
  rm -rf "$VENV_DIR"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment venv with $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1)) ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies ..."
pip install --quiet --upgrade pip
pip install --quiet -e .
pip install --quiet pytest pytest-mock pytest-timeout numpy

if [[ -n "$TOKEN" ]]; then
  export ENDEE_API_TOKEN="$TOKEN"
fi
if [[ -n "$BASE_URL" ]]; then
  export ENDEE_BASE_URL="$BASE_URL"
fi

MARKER_ARGS=()
if [[ -n "$MARKER" ]]; then
  MARKER_ARGS=(-m "$MARKER")
fi

echo "Running tests ..."
EXIT_CODE=0
pytest \
  -v \
  --timeout=120 \
  --tb=short \
  -p no:warnings \
  "${MARKER_ARGS[@]}" \
  "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}" || EXIT_CODE=$?

if [[ "$CLEAN" -eq 1 ]]; then
  echo "Removing virtual environment ..."
  deactivate 2>/dev/null || true
  rm -rf "$VENV_DIR"
fi

exit $EXIT_CODE
