#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting address matcher (Gradio)"

# Ensure application directories are writable (important when volumes are bind-mounted).
for dir in \
    "${GRADIO_OUTPUT_FOLDER:-/home/user/app/output}" \
    "${GRADIO_INPUT_FOLDER:-/home/user/app/input}" \
    "${GRADIO_TEMP_DIR:-/tmp/gradio_tmp}" \
    "${CONFIG_FOLDER:-/home/user/app/config}" \
    "/home/user/app/feedback" \
    "/home/user/app/logs" \
    "/home/user/app/usage"; do
    mkdir -p "$dir" 2>/dev/null || true
    if [ ! -w "$dir" ]; then
        echo "WARNING: Directory $dir is not writable by current user (uid=$(id -u)). File I/O may fail." >&2
    fi
done

exec python app.py
