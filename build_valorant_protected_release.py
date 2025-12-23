# -*- coding: utf-8 -*-
"""
Windows 一体化流水线（参考你的 build_gui_protected_release.py 的组织方式）：
1) Nuitka 把 valorant.py 编译成扩展模块 .pyd（只保护入口逻辑，不编译 torch 等依赖）
2) robocopy 项目 -> _release_src（排除开发/构建目录）
3) 将 _compiled_app 中产物替换进 _release_src（删除 valorant.py，仅保留 .pyd/.pyi 等）
4) 生成 PyInstaller spec + runtime hook，并在 _release_src 内构建
5) 将 dist/<APP_NAME> 拷回 dist_protected/<APP_NAME>

用法：
  (venv) > python build_valorant_protected_release.py

输出：
  dist_protected/<APP_NAME>/

注意：
- Nuitka 需要本机 MSVC/Build Tools（否则会报找不到编译器）。
- 如果你的 valorant.py 只有 `if __name__ == "__main__":` 入口，建议把主逻辑提成 `def main(): ...`
  这样 bootstrap.py 可以稳定调用（见下方 BOOTSTRAP_CODE）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ========= 可按需修改 =========
APP_NAME = "ValorantApp"          # PyInstaller 输出目录/EXE 名称
ENTRY_PY = "bootstrap.py"         # PyInstaller 入口脚本名（由本脚本生成）
SRC_ENTRY = "valorant.py"         # 你要保护的源码入口
YOLo_DIRNAME = "yolov5-master"    # 本地 yolov5 目录名（保留原名）

# 是否把 runs 整个目录一起带上（如果你的 best.pt 等权重在 runs 里，这个要 True）
INCLUDE_RUNS_DIR = True

# PyInstaller：是否显示控制台（GUI 程序一般 False；命令行/调试一般 True）
PYI_CONSOLE = True
# =============================

ROOT = Path(__file__).resolve().parent

COMPILED_ROOT = ROOT / "_compiled_app"
RELEASE_SRC = ROOT / "_release_src"
DIST_PROTECTED = ROOT / "dist_protected" / APP_NAME

# 你项目里不想带进 release 的目录（按需增删）
EXCLUDE_DIRS = [
    ".venv",
    ".git",
    "__pycache__",
    "dist",
    "dist_nuitka",
    "_compiled_app",
    "_release_src",
    "dist_protected",
    "._pyi_gen",
]

EXCLUDE_FILES = [
    "*.build",
    "*.dist",
    "*.onefile-build",
    "*.spec",
]

# ---------- PyInstaller runtime hook ----------
RTHOOK_CODE = r"""# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

def _add_dll_dir(p: Path) -> None:
    if not p.exists():
        return
    os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(str(p))
    except Exception:
        pass

# 稳定性：避免 TorchScript 某些情况下在 frozen 环境里自检源码失败
os.environ.setdefault("PYTORCH_JIT", "0")

if getattr(sys, "frozen", False):
    base = Path(sys.executable).resolve().parent
    internal = base / "_internal"

    # 1) DLL 搜索路径（torch/cuda/opencv/qt 等）
    _add_dll_dir(internal)
    _add_dll_dir(internal / "torch" / "lib")

    # 2) 让“本地目录模块”可 import（yolov5-master 不是标准包，经常靠 sys.path）
    for p in (base, base / "driver", base / "yolov5-master"):
        if p.exists():
            sys.path.insert(0, str(p))
"""

# ---------- bootstrap：稳定调用编译后的 valorant 模块 ----------
BOOTSTRAP_CODE = r"""# -*- coding: utf-8 -*-
import sys

def _call_entry(m):
    # 约定优先级：main() > run() > import side-effect
    for name in ("main", "run"):
        fn = getattr(m, name, None)
        if callable(fn):
            return fn()
    return 0

def main():
    import valorant  # compiled .pyd
    return _call_entry(valorant)

if __name__ == "__main__":
    raise SystemExit(main())
"""

# ---------- Spec 模板（尽量“稳打稳扎”，不追求极致小） ----------
SPEC_TEMPLATE = r"""# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os
import site

from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

block_cipher = None

spec_dir = Path(CONF["specpath"]).resolve()
project_dir = spec_dir.parent

def _norm_2tuple_list(items):
    out = []
    for x in items or []:
        if isinstance(x, tuple) and len(x) >= 2:
            out.append((x[0], x[1]))
    return out

def _dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

datas = []
binaries = []
hiddenimports = []

def _collect_pkg(pkg_name: str):
    global datas, binaries, hiddenimports
    try:
        d, b, h = collect_all(pkg_name)
        datas += _norm_2tuple_list(d)
        binaries += _norm_2tuple_list(b)
        hiddenimports += list(h or [])
        return True
    except Exception:
        return False

# ---- 重点：Torch/Ultralytics（你的 requirements 里有） ----
_collect_pkg("torch")
_collect_pkg("torchvision")
_collect_pkg("torchaudio")
_collect_pkg("ultralytics")

# 常见依赖（尽量不漏）
for pkg in ("numpy", "pandas", "cv2", "mss", "psutil", "PIL", "yaml"):
    _collect_pkg(pkg)

# OpenCV 有时需要显式
hiddenimports += ["cv2", "cv2.cv2"]

# torch 动态库
try:
    binaries += _norm_2tuple_list(collect_dynamic_libs("torch"))
except Exception:
    pass

# 把 site-packages/nvidia 下的 CUDA dll 一并收进 torch\\lib（提高在“干净机器”上的成功率）
def _collect_nvidia_cuda_dlls_to_torch_lib():
    out = []
    sp_list = []
    try:
        sp_list += site.getsitepackages()
    except Exception:
        pass
    try:
        sp_list += [site.getusersitepackages()]
    except Exception:
        pass

    seen_src = set()
    for sp in sp_list:
        if not sp:
            continue
        nvidia_dir = Path(sp) / "nvidia"
        if not nvidia_dir.exists():
            continue
        for pat in ("**/bin/*.dll", "**/lib/*.dll"):
            for p in nvidia_dir.glob(pat):
                src = str(p.resolve())
                k = src.lower()
                if k in seen_src:
                    continue
                seen_src.add(k)
                out.append((src, os.path.join("torch", "lib")))
    return out

try:
    binaries += _collect_nvidia_cuda_dlls_to_torch_lib()
except Exception:
    pass

# ---- 把你的本地目录/资源带进去（目录用 (src_dir, dest_dir) 2 元组） ----
cfg = project_dir / "config.json"
if cfg.exists():
    datas.append((str(cfg), "."))

driver_dir = project_dir / "driver"
if driver_dir.exists():
    datas.append((str(driver_dir), "driver"))

yolo_dir = project_dir / "{yolo_dirname}"
if yolo_dir.exists():
    datas.append((str(yolo_dir), "{yolo_dirname}"))

runs_dir = project_dir / "runs"
if {include_runs} and runs_dir.exists():
    datas.append((str(runs_dir), "runs"))

# 让分析阶段能找到 yolov5-master 下的“非包式模块”（models/common 等）
pathex = [str(project_dir)]
if yolo_dir.exists():
    pathex.append(str(yolo_dir))

binaries = _dedup(binaries)
datas = _dedup(datas)
hiddenimports = _dedup([h for h in hiddenimports if isinstance(h, str)])

entry_script = project_dir / "{entry_py}"
hook_script  = spec_dir / "{hook_name}"

a = Analysis(
    [str(entry_script)],
    pathex=pathex,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(hook_script)],
    excludes=[
        "torch.testing",
        "torch.testing._internal",
        "torch.testing._internal.opinfo",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="{app_name}",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console={console_flag},
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="{app_name}",
)
"""

# ------------------ 工具函数 ------------------
def _die(msg: str) -> None:
    print(f"[ERROR] {msg}")
    raise SystemExit(1)

def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN] " + " ".join(map(str, cmd)))
    subprocess.run([str(x) for x in cmd], check=True, cwd=str(cwd) if cwd else None)

def _clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

def _robocopy_dir(src: Path, dst: Path, exclude_dirs: list[str], exclude_files: list[str]) -> None:
    if os.name != "nt":
        _die("此脚本仅面向 Windows（依赖 robocopy）。")
    cmd = ["robocopy", str(src), str(dst), "/E"]
    if exclude_dirs:
        cmd += ["/XD"] + exclude_dirs
    if exclude_files:
        cmd += ["/XF"] + exclude_files
    print("[RUN] " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(ROOT))
    # robocopy: 0-7 success, >=8 failure
    if p.returncode >= 8:
        _die(f"robocopy failed, code={p.returncode}")
    print(f"[OK] robocopy done (code={p.returncode})")


# ------------------ 1) Nuitka 编译入口 ------------------
def compile_entry_to_pyd() -> None:
    src = ROOT / SRC_ENTRY
    if not src.exists():
        _die(f"找不到入口文件：{src}")

    # check nuitka
    _run([sys.executable, "-m", "nuitka", "--version"])

    _clean_dir(COMPILED_ROOT)

    # 只编译入口本体，避免把 torch/ultralytics 全部拉进 C 编译（会很慢且容易踩坑）
    cmd = [
        sys.executable, "-m", "nuitka",
        "--module",
        "--nofollow-imports",
        "--assume-yes-for-downloads",
        f"--output-dir={COMPILED_ROOT}",
        str(src),
    ]
    _run(cmd)

    pyds = sorted(COMPILED_ROOT.glob("valorant*.pyd"))
    if not pyds:
        # 有时会输出到 valorant.build/ 下面，再找一次
        pyds = sorted(COMPILED_ROOT.rglob("valorant*.pyd"))
    if not pyds:
        _die("Nuitka 没有产出 .pyd（请检查是否装了 MSVC Build Tools，或查看输出日志）")

    print("[OK] Nuitka compiled:")
    for p in pyds[:5]:
        print("   -", p.relative_to(ROOT))
    if len(pyds) > 5:
        print(f"   ... ({len(pyds)} files)")

# ------------------ 2) 生成 _release_src ------------------
def prepare_release_src() -> None:
    _clean_dir(RELEASE_SRC)

    exclude_dirs = EXCLUDE_DIRS.copy()
    if not INCLUDE_RUNS_DIR and "runs" not in exclude_dirs:
        exclude_dirs.append("runs")

    _robocopy_dir(ROOT, RELEASE_SRC, exclude_dirs=exclude_dirs, exclude_files=EXCLUDE_FILES)

    print(f"[OK] release source prepared: {RELEASE_SRC}")

# ------------------ 3) 替换入口：删除 valorant.py，拷贝 .pyd 等 ------------------
def replace_entry_in_release() -> None:
    # 删除源码入口（保护）
    py = RELEASE_SRC / SRC_ENTRY
    if py.exists():
        py.unlink()

    # 拷贝 Nuitka 产物（.pyd/.pyi/.dll/.pdb/.exp/.lib 可能存在）
    exts = {".pyd", ".pyi", ".dll", ".pdb", ".exp", ".lib"}
    for p in COMPILED_ROOT.rglob("valorant*"):
        if p.is_file() and p.suffix.lower() in exts:
            dst = RELEASE_SRC / p.name
            shutil.copy2(p, dst)

    if not list(RELEASE_SRC.glob("valorant*.pyd")):
        _die("替换后 _release_src 里仍找不到 valorant*.pyd，说明拷贝失败")

    # 写 bootstrap.py
    (RELEASE_SRC / ENTRY_PY).write_text(BOOTSTRAP_CODE, encoding="utf-8")

    print("[OK] entry replaced: removed valorant.py, added valorant*.pyd and bootstrap.py")

# ------------------ 4) PyInstaller 构建 ------------------
def pyinstaller_build_in_release() -> None:
    _run([sys.executable, "-m", "PyInstaller", "--version"])

    gen_dir = RELEASE_SRC / "._pyi_gen"
    gen_dir.mkdir(exist_ok=True)

    hook_name = "rthook_runtime.py"
    hook_path = gen_dir / hook_name
    spec_path = gen_dir / f"{APP_NAME}.spec"

    hook_path.write_text(RTHOOK_CODE, encoding="utf-8")

    spec_text = SPEC_TEMPLATE.format(
        entry_py=ENTRY_PY,
        hook_name=hook_name,
        app_name=APP_NAME,
        yolo_dirname=YOLo_DIRNAME,
        include_runs=("True" if INCLUDE_RUNS_DIR else "False"),
        console_flag=("True" if PYI_CONSOLE else "False"),
    )
    spec_path.write_text(spec_text, encoding="utf-8")

    cmd = [sys.executable, "-m", "PyInstaller", str(spec_path), "--clean", "-y"]
    print("[BUILD] " + " ".join(map(str, cmd)))
    subprocess.run([str(x) for x in cmd], check=True, cwd=str(RELEASE_SRC))

    print(f"[OK] PyInstaller build done: _release_src\\dist\\{APP_NAME}\\")

# ------------------ 5) 拷回 dist_protected ------------------
def collect_dist_back() -> None:
    src_dist = RELEASE_SRC / "dist" / APP_NAME
    if not src_dist.exists():
        _die(f"未找到 PyInstaller 输出：{src_dist}")

    if DIST_PROTECTED.exists():
        shutil.rmtree(DIST_PROTECTED, ignore_errors=True)
    DIST_PROTECTED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dist, DIST_PROTECTED, dirs_exist_ok=True)

    print(f"[OK] protected dist copied to: {DIST_PROTECTED}")

def main() -> None:
    # safety
    if "_release_src" in str(ROOT).lower():
        _die("请在项目根目录运行本脚本，不要在 _release_src 里运行。")

    # 0) sanity
    if not (ROOT / SRC_ENTRY).exists():
        _die(f"项目根目录缺少 {SRC_ENTRY}")

    # 1) nuitka
    compile_entry_to_pyd()

    # 2) release src
    prepare_release_src()

    # 3) replace entry
    replace_entry_in_release()

    # 4) pyinstaller
    pyinstaller_build_in_release()

    # 5) collect
    collect_dist_back()

    print("\n[DONE]")
    print(f"  {DIST_PROTECTED}")

if __name__ == "__main__":
    main()
