# mujoco-characters (Windows): build the mjchar wheel with the prefix's CPython 3.12, install it
# into the staging prefix, and add a relocatable bin\mjchar.cmd launcher, the docs and the example.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

foreach ($v in 'CVC_SOURCE_DIR', 'CVC_INSTALL_DIR', 'CVC_DEPS_PREFIX') {
    if (-not (Get-Item "env:$v" -ErrorAction SilentlyContinue)) { throw "$v must be set" }
}

# Windows CPython sits at the prefix root as python.exe.
$py = Join-Path $env:CVC_DEPS_PREFIX 'python.exe'
if (-not (Test-Path $py)) { throw "mujoco-characters: $py not found (build-depends on python312)" }
$ver = (& $py -c 'import sys; print("%d.%d" % sys.version_info[:2])').Trim()
if ($ver -ne '3.12') { throw "mujoco-characters: $py is Python $ver, need 3.12" }

# setuptools is a build-only dependency, staged in the build prefix.
if ($env:CVC_BUILD_PREFIX) {
    $sp = Join-Path $env:CVC_BUILD_PREFIX 'Lib\site-packages'
    $env:PYTHONPATH = if ($env:PYTHONPATH) { "$sp;$env:PYTHONPATH" } else { $sp }
}

$work = if ($env:CVC_BUILD_DIR) { $env:CVC_BUILD_DIR } else { Join-Path ([IO.Path]::GetTempPath()) ("mjchar-" + [guid]::NewGuid()) }
# Build from a copy: setuptools writes build\ and *.egg-info next to the sources.
$src = Join-Path $work 'mjchar-src'
$wheelhouse = Join-Path $work 'wheelhouse'
foreach ($d in $src, $wheelhouse) {
    if (Test-Path $d) { Remove-Item -Recurse -Force $d }
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}
foreach ($f in 'pyproject.toml', 'README.md', 'LICENSE', 'src') {
    Copy-Item -Recurse -Path (Join-Path $env:CVC_SOURCE_DIR $f) -Destination $src
}
Get-ChildItem -Path $src -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force

& $py -m pip wheel --no-build-isolation --no-deps --no-index --no-cache-dir --wheel-dir $wheelhouse $src
if ($LASTEXITCODE -ne 0) { throw "mujoco-characters: pip wheel failed ($LASTEXITCODE)" }
$wheel = Get-ChildItem -Path $wheelhouse -Filter 'mjchar-*.whl' -File | Select-Object -First 1
if (-not $wheel) { throw "mujoco-characters: no wheel under $wheelhouse" }
Write-Output "mujoco-characters: built $($wheel.Name)"

& $py -m pip install --no-index --no-deps --no-compile --ignore-installed --prefix $env:CVC_INSTALL_DIR $wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "mujoco-characters: pip install failed ($LASTEXITCODE)" }

# pip records the wheel's path on this builder in direct_url.json, and the console script below
# is replaced; drop both from the dist-info.
$distInfo = (Get-ChildItem -Directory -Path (Join-Path $env:CVC_INSTALL_DIR 'Lib\site-packages') -Filter 'mjchar-*.dist-info' |
             Select-Object -First 1).FullName
$directUrl = Join-Path $distInfo 'direct_url.json'
if (Test-Path $directUrl) { Remove-Item -Force $directUrl }
$record = Join-Path $distInfo 'RECORD'
$keep = Get-Content $record | Where-Object { $_ -notmatch '^mjchar-[^/]*\.dist-info/direct_url\.json,' -and $_ -notmatch 'Scripts/mjchar(\.exe|-script\.py),' }
[IO.File]::WriteAllLines($record, [string[]]$keep)

# pip's Scripts\mjchar.exe names this build's interpreter by absolute path, so it would break once
# the bundle is installed elsewhere. Replace it with a launcher that finds the prefix's Python.
$scripts = Join-Path $env:CVC_INSTALL_DIR 'Scripts'
foreach ($f in 'mjchar.exe', 'mjchar-script.py') {
    $p = Join-Path $scripts $f
    if (Test-Path $p) { Remove-Item -Force $p }
}
if ((Test-Path $scripts) -and -not (Get-ChildItem $scripts)) { Remove-Item $scripts }
$bin = Join-Path $env:CVC_INSTALL_DIR 'bin'
New-Item -ItemType Directory -Force -Path $bin | Out-Null
$launcher = @(
    '@echo off',
    'rem mjchar: runs the Python 3.12 at the root of this cvcpkg prefix, else python on PATH.',
    'setlocal',
    'set "MJCHAR_PY=%~dp0..\python.exe"',
    'if not exist "%MJCHAR_PY%" set "MJCHAR_PY=python"',
    '"%MJCHAR_PY%" -m mjchar %*',
    'exit /b %ERRORLEVEL%'
)
Set-Content -Path (Join-Path $bin 'mjchar.cmd') -Value $launcher -Encoding ascii

$share = Join-Path $env:CVC_INSTALL_DIR 'share\mujoco-characters'
$doc = Join-Path $env:CVC_INSTALL_DIR 'share\doc\mujoco-characters'
New-Item -ItemType Directory -Force -Path (Join-Path $share 'examples'), $doc | Out-Null
Copy-Item (Join-Path $env:CVC_SOURCE_DIR 'examples\demo_g1.py') (Join-Path $share 'examples')
Copy-Item (Join-Path $env:CVC_SOURCE_DIR 'README.md'), (Join-Path $env:CVC_SOURCE_DIR 'LICENSE') $doc

# The package and its bone tables import without the runtime dependencies.
$env:PYTHONPATH = Join-Path $env:CVC_INSTALL_DIR 'Lib\site-packages'
& $py -c "import mjchar, mjchar.bones; assert mjchar.bones.canonical_name('mixamorig:LeftUpLeg') == 'LeftUpLeg'"
if ($LASTEXITCODE -ne 0) { throw "mujoco-characters: import check failed ($LASTEXITCODE)" }
Write-Output "mujoco-characters: installed into $env:CVC_INSTALL_DIR"
