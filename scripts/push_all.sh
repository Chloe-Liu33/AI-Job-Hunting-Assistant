#!/usr/bin/env bash
#
# push_all.sh — one-click sync of this repo to BOTH remotes:
#   • GitHub            (remote: origin)
#   • Hugging Face Space (remote: space)
#
# Usage:
#   ./scripts/push_all.sh                  # push already-committed changes to both
#   ./scripts/push_all.sh "commit message" # stage everything, commit, then push
#
# Environment overrides:
#   ORIGIN_REMOTE  (default: origin)        GitHub remote name
#   SPACE_REMOTE   (default: space)         Hugging Face remote name
#   BRANCH         (default: current branch)
#
# What it does, in order:
#   1. Move to the repo root and detect the branch.
#   2. Refuse to push if any secret file (.env / users.json / data/users/) is tracked.
#   3. If the tree is dirty: commit (only when a message is given), else stop.
#   4. Push to GitHub.
#   5. Push to Hugging Face (auto force-push fallback — the Space mirrors local).
#   6. Print local vs origin vs space commit SHAs so you can confirm they match.

set -euo pipefail

ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
SPACE_REMOTE="${SPACE_REMOTE:-space}"

# ---- 1. repo root + branch ----
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "✗ Not inside a git repository."; exit 1;
}
cd "$ROOT"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

echo "Repo:   $ROOT"
echo "Branch: $BRANCH"
echo

# ---- 2. secret-leak guard (never push real secrets; .env.example is fine) ----
LEAKS="$(git ls-files | grep -E '(^|/)\.env$|(^|/)users\.json$|(^|/)data/users/' || true)"
if [ -n "$LEAKS" ]; then
  echo "✗ Refusing to push — these secret files are tracked by git:"
  echo "$LEAKS" | sed 's/^/      /'
  echo "  Untrack them, then retry:"
  echo "      git rm --cached <file>   &&   echo '<file>' >> .gitignore"
  exit 1
fi
echo "✓ No secret files tracked (.env / users.json safe to push around)."

# ---- 3. optional commit ----
if [ -n "$(git status --porcelain)" ]; then
  if [ "${1:-}" != "" ]; then
    echo "• Working tree dirty — staging and committing all changes…"
    git add -A
    git commit -m "$1"
  else
    echo "✗ You have uncommitted changes. Either commit them yourself, or pass a message:"
    echo "      ./scripts/push_all.sh \"your commit message\""
    echo "  Changes:"
    git status --short | sed 's/^/      /'
    exit 1
  fi
else
  echo "✓ Working tree clean."
  [ "${1:-}" != "" ] && echo "  (nothing to commit — ignoring the provided message)"
fi

push_to() {
  local remote="$1"
  local label="$2"
  echo
  echo "── Pushing to $label ($remote) ──"
  if ! git remote get-url "$remote" >/dev/null 2>&1; then
    echo "! Remote '$remote' is not configured — skipping."
    echo "    add it with:  git remote add $remote <url>"
    return 0
  fi
  if git push "$remote" "$BRANCH"; then
    return 0
  fi
  # The Hugging Face Space is a deploy mirror of local; if it diverged
  # (e.g. an edit made in the HF web UI), force it back in line with local.
  echo "! Normal push to '$remote' was rejected (remote diverged)."
  echo "  Force-pushing to make '$remote' match local…"
  git push "$remote" "$BRANCH" --force
}

# ---- 4 & 5. push to both ----
push_to "$ORIGIN_REMOTE" "GitHub"
push_to "$SPACE_REMOTE"  "Hugging Face"

# ---- 6. verify ----
echo
echo "── Verify (short SHAs should match) ──"
git fetch -q "$ORIGIN_REMOTE" "$SPACE_REMOTE" 2>/dev/null || true
printf "  %-22s %s\n" "local"                 "$(git rev-parse --short HEAD)"
git rev-parse --short "$ORIGIN_REMOTE/$BRANCH" >/dev/null 2>&1 && \
  printf "  %-22s %s\n" "$ORIGIN_REMOTE/$BRANCH" "$(git rev-parse --short "$ORIGIN_REMOTE/$BRANCH")"
git rev-parse --short "$SPACE_REMOTE/$BRANCH" >/dev/null 2>&1 && \
  printf "  %-22s %s\n" "$SPACE_REMOTE/$BRANCH"  "$(git rev-parse --short "$SPACE_REMOTE/$BRANCH")"

echo
echo "✓ Done. (Hugging Face will rebuild the Space automatically after a push.)"
