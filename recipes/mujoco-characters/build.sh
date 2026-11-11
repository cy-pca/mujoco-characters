#!/usr/bin/env bash
# mujoco-characters (noarch): build the mjchar wheel with the prefix's CPython 3.12, install it
# into lib/python3.12/site-packages of the staging prefix, and add a relocatable bin/mjchar
# launcher, the docs and the example.
#
# The noarch layout does not depend on the build host, so any CPython 3.12 prefix builds it:
# bin/python3.12 on Linux, macOS and the BSDs, or python.exe at the root of a Windows prefix
# (this script then runs under Git Bash).
set -euo pipefail

: "${CVC_SOURCE_DIR:?CVC_SOURCE_DIR must be set}"     # this repository's root (vendored)
: "${CVC_INSTALL_DIR:?CVC_INSTALL_DIR must be set}"
: "${CVC_DEPS_PREFIX:?CVC_DEPS_PREFIX must be set}"

# Paths handed to a native Windows python.exe must be Windows paths, joined with ';'.
WINHOST=0
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) WINHOST=1 ;; esac
native() { if [ "${WINHOST}" = 1 ]; then cygpath -w "$1"; else printf '%s\n' "$1"; fi; }
PATHSEP=":"; [ "${WINHOST}" = 1 ] && PATHSEP=";"

PY=""
for c in "${CVC_DEPS_PREFIX}/bin/python3.12" "${CVC_DEPS_PREFIX}/python.exe"; do
  if [ -x "${c}" ]; then PY="${c}"; break; fi
done
[ -n "${PY}" ] || { echo "mujoco-characters: no CPython in ${CVC_DEPS_PREFIX} (build-depends on python312)" >&2; exit 1; }
"${PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# setuptools is a build-only dependency, staged in the build prefix.
if [ -n "${CVC_BUILD_PREFIX:-}" ]; then
  for sp in "${CVC_BUILD_PREFIX}/lib/python3.12/site-packages" "${CVC_BUILD_PREFIX}/Lib/site-packages"; do
    if [ -d "${sp}" ]; then
      PYTHONPATH="$(native "${sp}")${PYTHONPATH:+${PATHSEP}${PYTHONPATH}}"
      export PYTHONPATH
      break
    fi
  done
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
    --wheel-dir "$(native "${WORK}/wheelhouse")" "$(native "${SRC}")"
WHEEL=""
for WHEEL in "${WORK}"/wheelhouse/mjchar-*.whl; do [ -e "${WHEEL}" ] && break; WHEEL=""; done
[ -n "${WHEEL}" ] || { echo "mujoco-characters: no wheel under ${WORK}/wheelhouse" >&2; exit 1; }
echo "mujoco-characters: built $(basename "${WHEEL}")"

# --target, not --prefix: the site-packages path is then the same whatever the build host.
SITE="${CVC_INSTALL_DIR}/lib/python3.12/site-packages"
mkdir -p "${SITE}"
"${PY}" -m pip install --no-index --no-deps --no-compile --target "$(native "${SITE}")" "$(native "${WHEEL}")"

# pip's console script names this build's interpreter by absolute path (and --target puts it in
# site-packages/bin), and direct_url.json records the wheel's path on this builder. Drop both;
# the launcher below replaces the script.
rm -rf "${SITE}/bin"
DIST_INFO="$(ls -d "${SITE}"/mjchar-*.dist-info)"
rm -f "${DIST_INFO}/direct_url.json"
grep -v -E -e '^mjchar-[^/]*\.dist-info/direct_url\.json,' -e '(^|/)bin/mjchar[^/]*,' "${DIST_INFO}/RECORD" > "${DIST_INFO}/RECORD.new"
mv "${DIST_INFO}/RECORD.new" "${DIST_INFO}/RECORD"

mkdir -p "${CVC_INSTALL_DIR}/bin"
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
PYTHONPATH="$(native "${SITE}")" "${PY}" -c \
  "import mjchar, mjchar.bones; assert mjchar.bones.canonical_name('mixamorig:LeftUpLeg') == 'LeftUpLeg'"
echo "mujoco-characters: installed into ${CVC_INSTALL_DIR}"
