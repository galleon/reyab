#!/bin/sh
# Renders the templates in /etc/envoy-src into /etc/envoy, then starts Envoy.
set -eu
SRC=/etc/envoy-src
DST=/etc/envoy
mkdir -p "$DST"

: "${OSS_A_NAME:?OSS_A_NAME must be set}"
: "${OSS_B_NAME:?OSS_B_NAME must be set}"
: "${SHADOW_MODEL:=claude-opus-5-5}"
: "${ENABLE_AUTO:=0}"
: "${AUTO_ROUTER:=vsr}"

for f in envoy.yaml common.lua public.lua eval.lua models.json; do
  sed -e "s|__OSS_A_NAME__|${OSS_A_NAME}|g" \
      -e "s|__OSS_B_NAME__|${OSS_B_NAME}|g" \
      -e "s|__SHADOW_MODEL__|${SHADOW_MODEL}|g" \
      -e "s|__ENABLE_AUTO__|${ENABLE_AUTO}|g" \
      -e "s|__AUTO_ROUTER__|${AUTO_ROUTER}|g" \
      "$SRC/$f" > "$DST/$f"
done

# Team keys are optional: without teams.lua the gateway runs open (team "anon").
if [ -f "$SRC/teams.lua" ]; then cp "$SRC/teams.lua" "$DST/teams.lua"; fi

exec envoy -c "$DST/envoy.yaml" --log-level info
