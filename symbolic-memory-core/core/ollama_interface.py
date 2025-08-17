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
    model: str = "llama2",
    host: Optional[str] = "localhost:11434",
    timeout: Optional[int] = None,
) -> str:
    """Query an Ollama model.

    If ``host`` is provided, the function sends an HTTP request to the
    specified ``host`` (which should include the port, e.g.
    ``"localhost:11434"``).  If ``host`` is ``None`` the local ``ollama``
    CLI is invoked.  The default host is ``"localhost:11434"``, meaning the
    network interface is used unless explicitly disabled.

    Args:
        prompt: Text prompt to send to the model.
        model: Name of the Ollama model to use.
        host: Optional ``host:port`` of a remote Ollama server.
        timeout: Optional timeout in seconds for the request/process.

    Returns:
        Model response text.  Errors are returned as strings prefixed with
        ``[ollama error]`` or ``[ollama network error]``.
    """

    if host:
        data = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = request.Request(
            f"http://{host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            try:
                return json.loads(payload).get("response", "")
            except json.JSONDecodeError:
                return payload.decode()
        except error.URLError as exc:  # includes HTTPError
            return f"[ollama network error] {exc.reason}" if hasattr(exc, "reason") else f"[ollama network error] {exc}"

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

