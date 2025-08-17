"""Utilities for interacting with an Ollama model.

The original project accessed Ollama exclusively through the local
command-line interface.  This module extends that behaviour by adding an
optional network wrapper.  The network interface is now preferred and
defaults to a server running on ``localhost:11434``.  Passing ``None`` for
``host`` will fall back to the local CLI.
"""

from typing import Optional
import json
import subprocess
from urllib import request, error

def query_ollama(
    prompt: str,
    model: str = "llama3:instruct",
    host: Optional[str] = "127.0.0.1:11434",
    timeout: Optional[int] = None,
    debug: bool = True,   # new flag for logging
) -> str:
    """Query an Ollama model via HTTP or local CLI (fallback)."""

    if host:
        url = f"http://{host}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        data = json.dumps(payload).encode()

        # Disable proxies so localhost requests go directly
        opener = request.build_opener(request.ProxyHandler({}))
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        if debug:
            print("---- OLLAMA REQUEST ----")
            print("URL:", url)
            print("Payload:", json.dumps(payload, indent=2))
            print("------------------------")

        try:
            with opener.open(req, timeout=timeout) as resp:
                body = resp.read()
            if debug:
                print("---- OLLAMA RESPONSE ----")
                print(body.decode(errors="replace"))
                print("-------------------------")
            try:
                obj = json.loads(body)
                return obj.get("response") or obj.get("message", "") or body.decode()
            except json.JSONDecodeError:
                return body.decode()
        except error.HTTPError as e:
            err_body = e.read().decode(errors="replace")
            if debug:
                print("---- OLLAMA HTTP ERROR ----")
                print("Status:", e.code)
                print("Body:", err_body)
                print("---------------------------")
            msg = err_body or getattr(e, "reason", str(e)) or str(e)
            return f"[ollama network error] HTTP {e.code}: {msg}"
        except error.URLError as e:
            return f"[ollama network error] {getattr(e, 'reason', e)}"

    # Fallback to local CLI invocation
    try:
        proc = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        if proc.returncode != 0:
            return f"[ollama error] {stderr.strip()}"
        return stdout
    except FileNotFoundError:
        return "[ollama error] 'ollama' CLI not found."
    except subprocess.TimeoutExpired:
        proc.kill()
        return "[ollama error] request timed out."

