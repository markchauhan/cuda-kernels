#!/usr/bin/env bash
# Compile-check the project inside the same nvidia/cuda image CI uses.
# Needs Docker only: no GPU and no local CUDA toolkit. Nothing is run, so this
# catches compile and link errors but says nothing about whether tests pass.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Read the image from the CI workflow so the two cannot drift apart.
IMAGE="$(sed -n 's/^[[:space:]]*image:[[:space:]]*//p' "$REPO_ROOT/.github/workflows/ci.yml" | head -n 1)"
if [[ -z "$IMAGE" ]]; then
    echo "compile_check: could not find the container image in .github/workflows/ci.yml" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "compile_check: docker not found on PATH" >&2
    exit 1
fi

echo "compile_check: building in $IMAGE"

# The repo is mounted read-only and the build tree lives in /tmp/build inside
# the container, so no build files land in the repo.
docker run --rm --platform linux/amd64 \
    -e DEBIAN_FRONTEND=noninteractive \
    -v "$REPO_ROOT:/src:ro" \
    "$IMAGE" \
    bash -euo pipefail -c '
        if ! command -v cmake >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
            apt-get update
            apt-get install -y --no-install-recommends cmake git ca-certificates
        fi
        cmake -B /tmp/build -S /src
        cmake --build /tmp/build -j
    '

echo "compile_check: build succeeded"
