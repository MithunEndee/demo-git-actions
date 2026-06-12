#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --token <api_token> [--base-url <url>] [--clean] [-- <pytest args>]

Run the Endee functional test suite against Endee Serverless.

Options:
  --token <api_token>   Endee Serverless API token (required)
  --base-url <url>      Override the API base URL (derived from token if omitted)
  --clean               Wipe and recreate .venv before running, delete it after
  --                    Pass all following arguments directly to pytest
  -h, --help            Show this help message

Examples:
  $0 --token <your_token>
  $0 --token <your_token> --base-url http://localhost:8080/api/v1
  $0 --token <your_token> --clean
  $0 --token <your_token> -- tests/test_querying.py
  $0 --token <your_token> -- -k test_filter_eq
  $0 --token <your_token> --clean -- tests/test_querying.py
EOF
  exit 1
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
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token)    TOKEN="$2";    shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --)         shift; PYTEST_ARGS+=("$@"); break ;;
    -h|--help)  usage ;;
    *)          echo "Unknown argument: $1"; usage ;;
  esac
done

if [[ -z "$TOKEN" ]]; then
  echo "Error: --token is required"
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
VENV_DIR="$REPO_ROOT/.venv"

cd "$REPO_ROOT"

if [[ "$CLEAN" -eq 1 && -d "$VENV_DIR" ]]; then
  echo "Removing existing virtual environment ..."
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at .venv ..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies ..."
pip install --quiet --upgrade pip
pip install --quiet -e .
pip install --quiet pytest pytest-timeout numpy

export ENDEE_TOKEN="$TOKEN"
if [[ -n "$BASE_URL" ]]; then
  export ENDEE_BASE_URL="$BASE_URL"
fi

echo "Running tests ..."
pytest \
  -v \
  --timeout=120 \
  --tb=short \
  -p no:warnings \
  "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
EXIT_CODE=$?

if [[ "$CLEAN" -eq 1 ]]; then
  echo "Removing virtual environment ..."
  deactivate 2>/dev/null || true
  rm -rf "$VENV_DIR"
fi

exit $EXIT_CODE
