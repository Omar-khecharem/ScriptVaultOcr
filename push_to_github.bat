@echo off
setlocal
REM ============================================================================
REM ScriptVault OCR - Secure one-command GitHub release push (Windows)
REM
REM Initializes the repository, wires the remote, creates a conventional
REM commit, and pushes to main. No credentials are embedded: authentication
REM is delegated to Git Credential Manager or `gh auth login`.
REM
REM Usage:   push_to_github.bat
REM ============================================================================

set "REPO=https://github.com/Omar-khecharem/scriptvault_ocr.git"
set "BRANCH=main"
set "COMMIT_MSG=feat: initial enterprise-grade release of ScriptVault OCR"

echo [1/4] Git repository
if not exist ".git" (
  git init -b %BRANCH%
  echo       Initialized repository on branch '%BRANCH%'.
) else (
  echo       Repository already initialized.
)
git branch -M %BRANCH%

echo [2/4] Remote 'origin'
git remote get-url origin >nul 2>&1
if errorlevel 1 (
  git remote add origin %REPO%
) else (
  git remote set-url origin %REPO%
)
echo       origin -^> %REPO%

echo [3/4] Stage ^& commit
git add -A
git diff --cached --quiet
if errorlevel 1 (
  git commit -m "%COMMIT_MSG%"
  echo       Committed: %COMMIT_MSG%
) else (
  echo       Nothing to commit; working tree already clean.
)

echo [4/4] Push
echo       Authenticate via Git Credential Manager or 'gh auth login' if prompted.
git push -u origin %BRANCH%
echo       Pushed to %REPO% (%BRANCH%).
endlocal
