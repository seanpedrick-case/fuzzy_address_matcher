# =============================================================================
# Stage 1: install Python dependencies (CPU PyTorch + app requirements)
# =============================================================================
FROM public.ecr.aws/docker/library/python:3.12.13-slim-trixie AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ \
        make \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

ARG INSTALL_NNET=False
ENV INSTALL_NNET=${INSTALL_NNET}

COPY requirements.txt .
COPY requirements_nnet.txt .

# Production image: omit pytest (keep it in requirements.txt for local dev / CI).
RUN grep -v '^pytest' requirements.txt > /tmp/requirements.docker.txt \
    && pip install --verbose --no-cache-dir --target=/install \
        -r /tmp/requirements.docker.txt \
    && if [ "$INSTALL_NNET" = "True" ]; then \
        pip install --verbose --no-cache-dir --target=/install \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            -r requirements_nnet.txt; \
    fi \
    && rm -f requirements.txt requirements_nnet.txt /tmp/requirements.docker.txt

# =============================================================================
# Stage 2: Gradio runtime (non-root). No Lambda stage — use a separate image later if needed.
# =============================================================================
FROM public.ecr.aws/docker/library/python:3.12.13-slim-trixie AS gradio

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV APP_HOME=/home/user

# Align with fuzzy_address_matcher/constants.py (GRADIO_OUTPUT_FOLDER) and fuzzy_address_matcher/config.py (CONFIG_FOLDER, APP_CONFIG_PATH).
# Paths are under the app working directory so relative feedback/logs/usage trees work (see app.py).
ENV GRADIO_TEMP_DIR=/tmp/gradio_tmp/ \
    GRADIO_OUTPUT_FOLDER=$APP_HOME/app/output/ \
    GRADIO_INPUT_FOLDER=$APP_HOME/app/input/ \
    CONFIG_FOLDER=$APP_HOME/app/config/ \
    MPLCONFIGDIR=/tmp/matplotlib_cache/ \
    XDG_CACHE_HOME=/tmp/xdg_cache/user_1000 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PATH=$APP_HOME/.local/bin:$PATH \
    PYTHONPATH=$APP_HOME/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_ANALYTICS_ENABLED=False

COPY --from=builder /install /usr/local/lib/python3.12/site-packages/
COPY --from=builder /install/bin /usr/local/bin/

COPY . ${APP_HOME}/app
COPY entrypoint.sh ${APP_HOME}/app/entrypoint.sh

RUN sed -i 's/\r$//' ${APP_HOME}/app/entrypoint.sh \
    && chmod +x ${APP_HOME}/app/entrypoint.sh

WORKDIR ${APP_HOME}/app

RUN useradd -m -u 1000 user \
    && mkdir -p ${APP_HOME}/app \
    && chown user:user ${APP_HOME}/app

RUN mkdir -p \
    ${APP_HOME}/app/output \
    ${APP_HOME}/app/input \
    ${APP_HOME}/app/logs \
    ${APP_HOME}/app/usage \
    ${APP_HOME}/app/feedback \
    ${APP_HOME}/app/config \
    && chown user:user \
    ${APP_HOME}/app/output \
    ${APP_HOME}/app/input \
    ${APP_HOME}/app/logs \
    ${APP_HOME}/app/usage \
    ${APP_HOME}/app/feedback \
    ${APP_HOME}/app/config \
    && chmod 755 \
    ${APP_HOME}/app/output \
    ${APP_HOME}/app/input \
    ${APP_HOME}/app/logs \
    ${APP_HOME}/app/usage \
    ${APP_HOME}/app/feedback \
    ${APP_HOME}/app/config

RUN mkdir -p /tmp/gradio_tmp /tmp/matplotlib_cache /tmp/xdg_cache/user_1000 \
    && chown user:user /tmp/gradio_tmp /tmp/matplotlib_cache /tmp/xdg_cache/user_1000 \
    && chmod 1777 /tmp/gradio_tmp \
    && chmod 700 /tmp/xdg_cache/user_1000

RUN chown -R user:user /home/user \
    && chmod 755 /usr/local/bin/python

VOLUME ["/tmp/gradio_tmp"]
VOLUME ["/tmp/matplotlib_cache"]
VOLUME ["/home/user/app/output"]
VOLUME ["/home/user/app/input"]
VOLUME ["/home/user/app/logs"]
VOLUME ["/home/user/app/usage"]
VOLUME ["/home/user/app/feedback"]
VOLUME ["/home/user/app/config"]

USER user

EXPOSE 7860

ENTRYPOINT ["/home/user/app/entrypoint.sh"]
