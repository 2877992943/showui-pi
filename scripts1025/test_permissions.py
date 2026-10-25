#!/usr/bin/env python
"""macOS 权限自检脚本 — ShowUI-pi 实时屏幕控制所需的前置权限测试。

测试三项权限（macOS 需授权给“宿主终端 App”，如 Terminal / iTerm2 / VS Code）：
  1. 屏幕录制 (Screen Recording) — 截取屏幕内容
  2. 辅助功能 (Accessibility)   — 控制鼠标（移动/按下/拖拽）
  3. 实际功能验证               — 真实截屏 + 读鼠标位置

用法:
    python scripts/test_permissions.py           # 检查 + 弹出授权弹窗
    python scripts/test_permissions.py --no-prompt  # 只检查，不弹窗
    python scripts/test_permissions.py --wait 120    # 弹窗后轮询等待授权，最长 120 秒

授权方法: 系统设置 → 隐私与安全性 → 屏幕录制 / 辅助功能 → 勾选你的终端 App。
注意: 修改“屏幕录制”授权后，macOS 要求重启终端 App 才生效（先勾选、退出并重开终端、再跑本脚本）。

退出码: 0 = 全部通过; 1 = 有缺失。
"""

import argparse
import ctypes
import sys
import time
from pathlib import Path

# ---------- macOS 系统 API（ctypes 直调，不依赖 pyobjc） ----------

_cg = ctypes.cdll.LoadLibrary(
    "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
)
_appsvc = ctypes.cdll.LoadLibrary(
    "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
)

# macOS 10.15+ 的屏幕录制权限 API
_cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
_cg.CGPreflightScreenCaptureAccess.argtypes = []
_cg.CGRequestScreenCaptureAccess.restype = ctypes.c_bool
_cg.CGRequestScreenCaptureAccess.argtypes = []

# AXIsProcessTrustedWithOptions(options: CFDictionary?) -> Bool
# 传 kAXTrustedCheckOptionPrompt=True 会弹出“辅助功能”授权引导弹窗。
_appsvc.AXIsProcessTrusted.restype = ctypes.c_bool
_appsvc.AXIsProcessTrusted.argtypes = []
_appsvc.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
_appsvc.AXIsProcessTrustedWithOptions.argtypes = [ctypes.c_void_p]


def check_screen_capture() -> bool:
    return bool(_cg.CGPreflightScreenCaptureAccess())


def request_screen_capture() -> bool:
    """主动弹出系统授权弹窗；已被拒过时返回 False（需去系统设置手动开）。"""
    return bool(_cg.CGRequestScreenCaptureAccess())


def _cfstr(s: str) -> ctypes.c_void_p:
    cf = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    return cf.CFStringCreateWithCString(None, s.encode("utf-8"), 0x08000100)  # kCFStringEncodingUTF8


def _cfbool(b: bool) -> ctypes.c_void_p:
    cf = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    cf.CFBooleanCreate.restype = ctypes.c_void_p
    cf.CFBooleanCreate.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
    return cf.CFBooleanCreate(None, 1 if b else 0)


def check_accessibility(prompt: bool = False) -> bool:
    if not prompt:
        return bool(_appsvc.AXIsProcessTrusted())
    cf = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    cf.CFDictionaryCreate.restype = ctypes.c_void_p
    cf.CFDictionaryCreate.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
    ]
    key = _cfstr("AXTrustedCheckOptionPrompt")
    val = _cfbool(True)
    keys = (ctypes.c_void_p * 1)(key)
    vals = (ctypes.c_void_p * 1)(val)
    opts = cf.CFDictionaryCreate(None, keys, vals, 1, None, None)
    return bool(_appsvc.AXIsProcessTrustedWithOptions(opts))


# ---------- 功能性验证 ----------


def test_screenshot(out_dir: Path) -> bool:
    import pyautogui

    try:
        img = pyautogui.screenshot()
        if img is None or img.size[0] < 100:
            return False
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "perm_test_screenshot.png"
        img.save(out)
        print(f"      截屏成功: {img.size[0]}x{img.size[1]}，已保存 {out}（请打开确认能看到各窗口内容，"
              f"若只有壁纸/黑屏说明屏幕录制权限未真正生效）")
        return True
    except Exception as e:
        print(f"      截屏失败: {e}")
        return False


def test_mouse() -> bool:
    import pyautogui

    try:
        x, y = pyautogui.position()  # 读当前位置，不产生副作用
        # 无感微移验证: 向右 1px 再移回
        pyautogui.moveTo(x + 1, y, duration=0.05)
        pyautogui.moveTo(x, y, duration=0.05)
        print(f"      鼠标控制成功（当前位置 {x},{y}，已微移验证并复原）")
        return True
    except Exception as e:
        print(f"      鼠标控制失败: {e}")
        return False


# ---------- 主流程 ----------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-prompt", action="store_true", help="只检查，不弹授权弹窗")
    ap.add_argument("--wait", type=int, default=0, metavar="SEC",
                    help="弹窗后轮询等待授权完成，最长 SEC 秒")
    args = ap.parse_args()

    print("=" * 60)
    print("macOS 权限自检（宿主 App 需为你的终端）")
    print("=" * 60)

    ok_ax = check_accessibility(prompt=False)
    ok_sc = check_screen_capture()
    print(f"[1] 辅助功能 (Accessibility) : {'✅ 已授权' if ok_ax else '❌ 未授权'}")
    print(f"[2] 屏幕录制 (Screen Recording): {'✅ 已授权' if ok_sc else '❌ 未授权'}")

    if not (ok_ax and ok_sc) and not args.no_prompt:
        print("\n  → 正在请求系统弹窗，请在弹窗中打开 系统设置 并授权本终端 App …")
        if not ok_ax:
            check_accessibility(prompt=True)
        if not ok_sc:
            request_screen_capture()
        if args.wait > 0:
            deadline = time.time() + args.wait
            while time.time() < deadline:
                time.sleep(2)
                ok_ax, ok_sc = check_accessibility(False), check_screen_capture()
                if ok_ax and ok_sc:
                    break
            print(f"[重检] 辅助功能: {'✅' if ok_ax else '❌'}  屏幕录制: {'✅' if ok_sc else '❌'}")
        else:
            # 短暂等待，用户可能在弹窗里点了
            time.sleep(3)
            ok_ax, ok_sc = check_accessibility(False), check_screen_capture()

    if not (ok_ax and ok_sc):
        print("\n" + "=" * 60)
        print("❌ 权限未齐，手动授权步骤:")
        print("  1. 系统设置 → 隐私与安全性 → 辅助功能 → 勾选你的终端 App")
        print("  2. 系统设置 → 隐私与安全性 → 屏幕录制 → 勾选你的终端 App")
        print("  3. 屏幕录制授权后必须【完全退出并重开终端】再重新运行本脚本")
        print("=" * 60)
        return 1

    # 权限均到位，做功能级验证
    print("\n[3] 功能性验证:")
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    sc_ok = test_screenshot(out_dir)
    mouse_ok = test_mouse()

    print("=" * 60)
    if sc_ok and mouse_ok:
        print("✅ 全部权限测试通过，可以运行 demo_live_drag.py")
        return 0
    print("❌ 权限已授权但功能验证失败，请重启终端后重试")
    return 1


if __name__ == "__main__":
    sys.exit(main())
