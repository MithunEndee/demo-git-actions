#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --token <api_token> [--base-url <url>] [--clean] [-- <pytest args>]"
  echo ""
  echo "  --token      Endee Serverless API token (required)"
  echo "  --base-url   Override API base URL (optional, derived from token if omitted)"
  echo "  --clean      Delete and recreate the .venv before running"
  echo "  --           Pass any additional pytest arguments after this"
  echo ""
  echo "Examples:"
  echo "  $0 --token user:mytoken:us-east"
  echo "  $0 --token user:mytoken:us-east --base-url http://0.0.0.0:8080/api/v1"
  echo "  $0 --token user:mytoken:us-east --clean"
  echo "  $0 --token user:mytoken:us-east -- -k test_query_basic -v"
  exit 1
}

TOKEN=""
BASE_URL=""
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token)    TOKEN="$2";    shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --clean)    CLEAN=1;       shift ;;
    --)         shift; break ;;
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

# Remove existing venv if --clean was requested
if [[ "$CLEAN" -eq 1 && -d "$VENV_DIR" ]]; then
  echo "Removing existing virtual environment ..."
  rm -rf "$VENV_DIR"
fi

# Create virtual environment if it does not exist
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at .venv ..."
  python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Install / sync dependencies
echo "Installing dependencies ..."
pip install --quiet --upgrade pip
pip install --quiet -e .
pip install --quiet pytest pytest-html pytest-timeout numpy pytest-github-actions-annotate-failures

export ENDEE_TOKEN="$TOKEN"
if [[ -n "$BASE_URL" ]]; then
  export ENDEE_BASE_URL="$BASE_URL"
fi

echo "Running tests ..."
pytest tests/ \
  -v \
  --timeout=120 \
  --tb=short \
  -p no:warnings \
  "$@"
