#!/usr/bin/env bash
# mujoco-characters (noarch): build the mjchar wheel with the prefix's CPython 3.12, install it
# into the staging prefix, and add a relocatable bin/mjchar launcher, the docs and the example.
set -euo pipefail

: "${CVC_SOURCE_DIR:?CVC_SOURCE_DIR must be set}"     # this repository's root (vendored)
: "${CVC_INSTALL_DIR:?CVC_INSTALL_DIR must be set}"
: "${CVC_DEPS_PREFIX:?CVC_DEPS_PREFIX must be set}"

PY="${CVC_DEPS_PREFIX}/bin/python3.12"
[ -x "${PY}" ] || { echo "mujoco-characters: ${PY} not found (build-depends on python312)" >&2; exit 1; }

# setuptools is a build-only dependency, staged in the build prefix.
if [ -n "${CVC_BUILD_PREFIX:-}" ]; then
  export PYTHONPATH="${CVC_BUILD_PREFIX}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
fi

WORK="${CVC_BUILD_DIR:-$(mktemp -d)}"
# Build from a copy: setuptools writes build/ and *.egg-info next to the sources.
SRC="${WORK}/mjchar-src"
rm -rf "${SRC}" "${WORK}/wheelhouse"
mkdir -p "${SRC}" "${WORK}/wheelhouse"
cp -R "${CVC_SOURCE_DIR}/pyproject.toml" "${CVC_SOURCE_DIR}/README.md" "${CVC_SOURCE_DIR}/LICENSE" \
      "${CVC_SOURCE_DIR}/src" "${SRC}/"
find "${SRC}" -name '__pycache__' -type d -prune -exec rm -rf {} +

"${PY}" -m pip wheel --no-build-isolation --no-deps --no-index --no-cache-dir \
    --wheel-dir "${WORK}/wheelhouse" "${SRC}"
WHEEL=""
for WHEEL in "${WORK}"/wheelhouse/mjchar-*.whl; do [ -e "${WHEEL}" ] && break; WHEEL=""; done
[ -n "${WHEEL}" ] || { echo "mujoco-characters: no wheel under ${WORK}/wheelhouse" >&2; exit 1; }
echo "mujoco-characters: built $(basename "${WHEEL}")"

"${PY}" -m pip install --no-index --no-deps --no-compile --ignore-installed \
    --prefix "${CVC_INSTALL_DIR}" "${WHEEL}"

# pip records the wheel's path on this builder in direct_url.json; it means nothing elsewhere.
DIST_INFO="$(ls -d "${CVC_INSTALL_DIR}"/lib/python3.12/site-packages/mjchar-*.dist-info)"
rm -f "${DIST_INFO}/direct_url.json"
grep -v '^mjchar-[^/]*.dist-info/direct_url.json,' "${DIST_INFO}/RECORD" > "${DIST_INFO}/RECORD.new"
mv "${DIST_INFO}/RECORD.new" "${DIST_INFO}/RECORD"

# pip's console script names this build's interpreter by absolute path, so it would break once
# the bundle is installed elsewhere. Replace it with a launcher that finds the prefix's Python.
mkdir -p "${CVC_INSTALL_DIR}/bin"
rm -f "${CVC_INSTALL_DIR}/bin/mjchar"
{
  echo '#!/bin/sh'
  echo '# mjchar: runs the Python 3.12 installed beside this script in the cvcpkg prefix,'
  echo '# else python3.12 on PATH.'
  echo 'here=$(cd "$(dirname "$0")" && pwd)'
  echo 'if [ -x "${here}/python3.12" ]; then'
  echo '  exec "${here}/python3.12" -m mjchar "$@"'
  echo 'fi'
  echo 'exec python3.12 -m mjchar "$@"'
} > "${CVC_INSTALL_DIR}/bin/mjchar"
chmod 0755 "${CVC_INSTALL_DIR}/bin/mjchar"

SHARE="${CVC_INSTALL_DIR}/share/mujoco-characters"
DOC="${CVC_INSTALL_DIR}/share/doc/mujoco-characters"
mkdir -p "${SHARE}/examples" "${DOC}"
cp "${CVC_SOURCE_DIR}/examples/demo_g1.py" "${SHARE}/examples/"
cp "${CVC_SOURCE_DIR}/README.md" "${CVC_SOURCE_DIR}/LICENSE" "${DOC}/"

# The package and its bone tables import without the runtime dependencies.
PYTHONPATH="${CVC_INSTALL_DIR}/lib/python3.12/site-packages" "${PY}" -c \
  "import mjchar, mjchar.bones; assert mjchar.bones.canonical_name('mixamorig:LeftUpLeg') == 'LeftUpLeg'"
echo "mujoco-characters: installed into ${CVC_INSTALL_DIR}"
