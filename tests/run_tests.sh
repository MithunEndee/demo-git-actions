#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --token <api_token> [--root-token <root_token>] [--base-url <url>] [--clean] [-- <pytest args>]

Run the Endee functional test suite against a live server. Creates a venv,
installs dependencies, sets environment variables, and invokes pytest.

Options:
  --token <api_token>          Database-level API token (required).
                               Sets ENDEE_TOKEN for the test run.
  --root-token <root_token>    Root/admin token (optional).
                               Sets NDD_ROOT_TOKEN; enables test_admin.py tests.
                               Admin tests are automatically skipped if omitted.
  --base-url <url>             Override the server base URL (default: cloud endpoint).
                               Sets ENDEE_BASE_URL. Use for local or staging servers.
  --clean                      Delete and recreate venv before the run, then
                               remove it afterwards. Ensures a fresh environment.
  --                           Pass all remaining arguments directly to pytest.
                               Accepts file paths, -k keyword filters, -x, -s, etc.
  -h, --help                   Show this help message and exit.

Examples:
  # Run the full test suite (admin tests skipped, cloud endpoint)
  $0 --token <api_token>

  # Run against a local dev server
  $0 --token <api_token> --base-url http://localhost:8080/api/v2

  # Enable admin tests by supplying a root token
  $0 --token <api_token> --root-token <root_token>

  # Full local run with admin tests enabled
  $0 --token <api_token> --root-token <root_token> --base-url http://localhost:8080/api/v2

  # Fresh environment - recreate venv before and clean up after
  $0 --token <api_token> --clean

  # Run a single test file
  $0 --token <api_token> -- tests/test_searching.py

  # Run admin tests only (root token required)
  $0 --token <api_token> --root-token <root_token> -- tests/test_admin.py

  # Run tests matching a keyword pattern
  $0 --token <api_token> -- -k test_filter_eq

  # Stop on first failure and show captured output
  $0 --token <api_token> -- -x -s

  # Local dev, admin enabled, target one file, stop on first failure
  $0 --token <api_token> --root-token <root_token> \\
     --base-url http://localhost:8080/api/v2 -- tests/test_filtering.py -x
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
ROOT_TOKEN=""
BASE_URL=""
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token)       TOKEN="$2";       shift 2 ;;
    --root-token)  ROOT_TOKEN="$2";  shift 2 ;;
    --base-url)    BASE_URL="$2";    shift 2 ;;
    --)            shift; PYTEST_ARGS+=("$@"); break ;;
    -h|--help)     usage 0 ;;
    *)             echo "Unknown argument: $1"; usage ;;
  esac
done

if [[ -z "$TOKEN" ]]; then
  echo "Error: --token is required"
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

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment venv ..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies ..."
pip install --quiet --upgrade pip
pip install --quiet -e .
pip install --quiet pytest pytest-timeout numpy

export ENDEE_TOKEN="$TOKEN"
if [[ -n "$ROOT_TOKEN" ]]; then
  export NDD_ROOT_TOKEN="$ROOT_TOKEN"
fi
if [[ -n "$BASE_URL" ]]; then
  export ENDEE_BASE_URL="$BASE_URL"
fi

echo "Running tests ..."
EXIT_CODE=0
pytest \
  -v \
  --timeout=120 \
  --tb=short \
  -p no:warnings \
  "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}" || EXIT_CODE=$?

if [[ "$CLEAN" -eq 1 ]]; then
  echo "Removing virtual environment ..."
  deactivate 2>/dev/null || true
  rm -rf "$VENV_DIR"
fi

exit $EXIT_CODE
