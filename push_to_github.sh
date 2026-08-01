#!/usr/bin/env bash
# ============================================================================
# ScriptVault OCR — Secure one-command GitHub release push
#
# Initializes the repository, wires the remote, creates a conventional
# commit, and pushes to main. No credentials are embedded: authentication
# is delegated to your local Git credential helper or `gh auth login`.
#
# Usage:   ./push_to_github.sh
# Windows: push_to_github.bat
# ============================================================================
set -euo pipefail

REPO="https://github.com/Omar-khecharem/scriptvault_ocr.git"
BRANCH="main"
COMMIT_MSG="feat: initial enterprise-grade release of ScriptVault OCR"

echo "[1/4] Git repository"
if [ ! -d ".git" ]; then
  git init -b "$BRANCH"
  echo "      Initialized repository on branch '$BRANCH'."
else
  echo "      Repository already initialized."
fi
git branch -M "$BRANCH"

echo "[2/4] Remote 'origin'"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO"
else
  git remote add origin "$REPO"
fi
echo "      origin -> $REPO"

echo "[3/4] Stage & commit"
git add -A
if git diff --cached --quiet; then
  echo "      Nothing to commit; working tree already clean."
else
  git commit -m "$COMMIT_MSG"
  echo "      Committed: $COMMIT_MSG"
fi

echo "[4/4] Push"
echo "      Authenticate via your Git credential helper or 'gh auth login' if prompted."
git push -u origin "$BRANCH"
echo "      Pushed to $REPO ($BRANCH)."
