"""
action_runner.py
Executes per-letter system actions (like open Calculator when 'C' is predicted).
Customize mappings in actions_config.json (windows/darwin/linux).
"""
import os, json, subprocess, platform, shlex

CONFIG_PATHS = ["actions_config.json", os.path.join("config","actions_config.json")]

def _load_config():
    for p in CONFIG_PATHS:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {"windows": {}, "darwin": {}, "linux": {}}

def _oskey():
    sys = platform.system().lower()
    if "windows" in sys:
        return "windows"
    if "darwin" in sys or "mac" in sys:
        return "darwin"
    return "linux"

def run_action(letter: str) -> dict:
    if not letter:
        return {"ok": False, "error": "Empty letter"}
    L = letter.upper()[0]
    cfg = _load_config()
    key = _oskey()
    mapping = cfg.get(key, {})
    cmd = (mapping or {}).get(L, "").strip()
    if not cmd:
        return {"ok": False, "error": f"No action configured for '{L}' on {key}"}

    try:
        # Support fallbacks joined by '||'
        parts = [c.strip() for c in cmd.split("||")] if "||" in cmd else [cmd]
        last_err = None
        for part in parts:
            try:
                if key == "windows":
                    subprocess.Popen(part, shell=True)
                else:
                    subprocess.Popen(shlex.split(part))
                return {"ok": True, "run": part}
            except Exception as e:
                last_err = str(e)
                continue
        return {"ok": False, "error": last_err or "Failed to run any fallback"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
