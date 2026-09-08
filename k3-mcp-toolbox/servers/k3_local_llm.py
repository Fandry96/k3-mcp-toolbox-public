#!/usr/bin/env python3
"""
k3-local-llm — Local Model Bridge MCP Server
Orchestrates bundled llama-server.exe and GGUF models for offline embeddings and inference.
Provides auto-starting, model discovery, and health telemetry.
"""

import os
import atexit
import subprocess
import time
import socket
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("k3-local-llm")
mcp._mcp_server.version = "1.0.0"

TOOLBOX_DIR = Path(os.environ.get("K3_TOOLBOX_DIR", Path(__file__).resolve().parent.parent)).resolve()
LLAMA_BIN_DIR = TOOLBOX_DIR / "src" / "llama_cpp_server"
LLAMA_EXE = LLAMA_BIN_DIR / "llama-server.exe"
MODELS_DIR = TOOLBOX_DIR / "src" / "models"
DEFAULT_MODEL = MODELS_DIR / "embeddinggemma-300m" / "embeddinggemma-300M-Q8_0.gguf"
DEFAULT_PORT = 8081

# Subprocess tracker
_SERVER_PROC: Optional[subprocess.Popen] = None
_ACTIVE_MODEL: Optional[str] = None
_ACTIVE_PORT: int = DEFAULT_PORT


def _cleanup_server_proc() -> None:
    """Ensures child llama-server process terminates when MCP server exits."""
    global _SERVER_PROC
    if _SERVER_PROC and _SERVER_PROC.poll() is None:
        try:
            _SERVER_PROC.terminate()
            _SERVER_PROC.wait(timeout=3)
        except Exception:
            _SERVER_PROC.kill()


atexit.register(_cleanup_server_proc)


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Tests if a TCP port is accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _check_health(port: int = DEFAULT_PORT) -> bool:
    """Pings the llama-server health endpoint and returns True if healthy."""
    if not _is_port_open(port):
        return False
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


def _find_gguf_models() -> List[Dict[str, Any]]:
    """Discovers all .gguf model files in the models directory."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for p in MODELS_DIR.rglob("*.gguf"):
        sz_mb = p.stat().st_size / (1024 * 1024)
        name_lower = p.name.lower()
        is_embedding = "embed" in name_lower or "bge" in name_lower or "nomic" in name_lower
        models.append({
            "name": p.name,
            "path": str(p),
            "size_mb": round(sz_mb, 1),
            "type": "embedding" if is_embedding else "generative",
        })
    return models


@mcp.tool()
def local_models_list() -> str:
    """
    Discovers all local GGUF models stored in the K3 models directory.
    Categorizes each model as either 'embedding' or 'generative'.
    """
    models = _find_gguf_models()
    if not models:
        return f"No .gguf models found in {MODELS_DIR}."

    lines = [
        f"=== Local GGUF Model Registry ===",
        f"Directory: {MODELS_DIR}",
        f"Total Models: {len(models)}\n",
    ]
    for m in sorted(models, key=lambda x: x["name"]):
        lines.append(
            f"• {m['name']:<35} | {m['size_mb']:>7.1f} MB | Type: {m['type'].upper()}"
        )
    return "\n".join(lines)


CURATED_MODELS: Dict[str, Dict[str, Any]] = {
    "bge-small-en": {
        "name": "bge-small-en-v1.5-q8_0.gguf",
        "url": "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
        "type": "embedding",
        "size_mb": 35.5,
        "description": "Ultra-compact 384-dim embedding model (only 35MB). Recommended for fast offline RAG.",
    },
    "qwen2.5-0.5b": {
        "name": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "type": "generative",
        "size_mb": 398.0,
        "description": "Ultra-lightweight 0.5B instruction-tuned model for local completions.",
    },
    "smollm2-360m": {
        "name": "smollm2-360m-instruct-q8_0.gguf",
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf",
        "type": "generative",
        "size_mb": 387.0,
        "description": "HuggingFace SmolLM2 compact instruction model.",
    },
}


@mcp.tool()
def local_model_download(model_key_or_url: str = "list") -> str:
    """
    Downloads a GGUF model into the K3 models directory for offline inference.
    Pass 'list' to see available curated models, or specify a model key
    ('bge-small-en', 'qwen2.5-0.5b', 'smollm2-360m') or direct HTTPS .gguf URL.
    """
    cleaned = model_key_or_url.strip()
    if cleaned.lower() == "list" or (cleaned not in CURATED_MODELS and not cleaned.startswith("http")):
        lines = [
            "=== Curated Downloadable Models ===",
            "Specify one of the keys below to download, or pass a direct HTTPS URL to a .gguf file:\n",
        ]
        for k, info in CURATED_MODELS.items():
            lines.append(f"• Key: {k:<15} | Size: {info['size_mb']:>5.1f} MB | Type: {info['type'].upper()}")
            lines.append(f"  Description: {info['description']}")
            lines.append(f"  Filename:    {info['name']}\n")
        return "\n".join(lines)

    if cleaned in CURATED_MODELS:
        info = CURATED_MODELS[cleaned]
        target_name = info["name"]
        download_url = info["url"]
        target_dir = MODELS_DIR / cleaned
    else:
        # Direct URL
        download_url = cleaned
        target_name = download_url.split("/")[-1].split("?")[0]
        if not target_name.endswith(".gguf"):
            return "Error: URL must point to a file ending in .gguf"
        target_dir = MODELS_DIR / "custom"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / target_name
    tmp_path = target_dir / f"{target_name}.tmp"

    if target_path.exists():
        sz = target_path.stat().st_size / (1024 * 1024)
        return f"Model already installed at {target_path} ({sz:.1f} MB)."

    try:
        sys.stderr.write(f"Downloading {target_name} from {download_url}...\n")
        with requests.get(download_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_bytes = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
        if tmp_path.exists():
            tmp_path.rename(target_path)
        sz_mb = target_path.stat().st_size / (1024 * 1024)
        return f"Successfully downloaded {target_name} to {target_path} ({sz_mb:.1f} MB)."
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        return f"Error downloading model: {e}"


@mcp.tool()
def local_server_status(port: int = DEFAULT_PORT) -> str:
    """
    Checks if llama-server.exe is currently active and healthy on the target port.
    Returns PID, target port, active model, and HTTP response.
    """
    global _SERVER_PROC, _ACTIVE_MODEL, _ACTIVE_PORT
    is_live = _check_health(port)

    lines = ["=== Local LLaMA Server Status ==="]
    if is_live:
        lines.append(f"Status     : ONLINE (Healthy)")
        lines.append(f"Port       : {port}")
        lines.append(f"Model      : {_ACTIVE_MODEL or 'Detected external/running instance'}")
        if _SERVER_PROC and _SERVER_PROC.poll() is None:
            lines.append(f"PID        : {_SERVER_PROC.pid}")
        try:
            r = requests.get(f"http://127.0.0.1:{port}/props", timeout=2)
            if r.status_code == 200:
                props = r.json()
                lines.append(f"Server Ver : {props.get('version', 'llama.cpp')}")
        except Exception:
            pass
    else:
        lines.append(f"Status     : OFFLINE (Port {port} closed)")
        lines.append("Use local_server_start() to boot llama-server.exe.")

    return "\n".join(lines)


@mcp.tool()
def local_server_start(
    model_name: Optional[str] = None,
    port: int = DEFAULT_PORT,
    embedding_mode: bool = True
) -> str:
    """
    Boots llama-server.exe in the background for local inference/embeddings.
    Args:
        model_name: Filename of GGUF model (defaults to embeddinggemma-300M-Q8_0.gguf).
        port: TCP port to host server on (default: 8081).
        embedding_mode: Enable --embedding flag for vector operations.
    """
    global _SERVER_PROC, _ACTIVE_MODEL, _ACTIVE_PORT

    if _check_health(port):
        return f"Server already online and healthy at http://127.0.0.1:{port} (Model: {_ACTIVE_MODEL or 'active'})."

    if port < 1 or port > 65535:
        return "Error: Port must be an integer between 1 and 65535."

    if not LLAMA_EXE.exists():
        return f"Error: llama-server.exe binary not found at {LLAMA_EXE}"

    # Resolve target model
    target_model_path = DEFAULT_MODEL
    if model_name:
        all_models = _find_gguf_models()
        match = next((m for m in all_models if model_name.lower() in m["name"].lower()), None)
        if match:
            target_model_path = Path(match["path"])
        else:
            return f"Model '{model_name}' not found. Use local_models_list() to see available files."

    if not target_model_path.exists():
        return f"Error: Model file does not exist at {target_model_path}"

    cmd = [
        str(LLAMA_EXE),
        "-m", str(target_model_path),
        "--port", str(port),
        "-c", "2048",
        "-t", "4",
    ]
    if embedding_mode:
        cmd.append("--embedding")

    try:
        _SERVER_PROC = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(LLAMA_BIN_DIR),
        )
        _ACTIVE_MODEL = target_model_path.name
        _ACTIVE_PORT = port

        # Poll health for up to 8 seconds
        for _ in range(16):
            time.sleep(0.5)
            if _check_health(port):
                return (
                    f"SUCCESS: llama-server.exe launched at http://127.0.0.1:{port}\n"
                    f"  - PID       : {_SERVER_PROC.pid}\n"
                    f"  - Model     : {target_model_path.name}\n"
                    f"  - Mode      : {'Embedding' if embedding_mode else 'Generative'}"
                )

        if _SERVER_PROC.poll() is not None:
            _, err = _SERVER_PROC.communicate(timeout=2)
            return f"Server process crashed immediately with exit code {_SERVER_PROC.returncode}:\n{err.decode('utf-8', errors='replace')}"

        return f"Started PID {_SERVER_PROC.pid}, but port {port} not responsive yet. Check local_server_status()."
    except Exception as e:
        return f"Failed to start llama-server: {e}"


@mcp.tool()
def local_server_stop() -> str:
    """
    Stops the active llama-server.exe subprocess cleanly.
    """
    global _SERVER_PROC, _ACTIVE_MODEL
    if _SERVER_PROC and _SERVER_PROC.poll() is None:
        pid = _SERVER_PROC.pid
        _SERVER_PROC.terminate()
        try:
            _SERVER_PROC.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _SERVER_PROC.kill()
        _SERVER_PROC = None
        _ACTIVE_MODEL = None
        return f"SUCCESS: Terminated llama-server.exe (PID {pid})."
    return "No managed llama-server process is currently running."


@mcp.tool()
def local_embed(text: str, port: int = DEFAULT_PORT) -> str:
    """
    Generates local embeddings using the active llama-server instance (e.g. EmbeddingGemma).
    Automatically starts the server if it is offline.
    Args:
        text: Input text string to generate embeddings for.
        port: Server port (default: 8081).
    """
    if not _check_health(port):
        start_res = local_server_start(port=port, embedding_mode=True)
        if not _check_health(port):
            return f"Cannot embed: failed to start local server.\n{start_res}"

    url = f"http://127.0.0.1:{port}/embedding"
    try:
        r = requests.post(url, json={"content": text}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            vec = data.get("embedding", [])
            dim = len(vec)
            norm = sum(x * x for x in vec) ** 0.5
            first_5 = [round(x, 4) for x in vec[:5]]
            return (
                f"=== Local Embedding Success ===\n"
                f"Vector Dimension : {dim}\n"
                f"L2 Norm          : {norm:.4f}\n"
                f"First 5 Values   : {first_5}...\n"
                f"Model            : {_ACTIVE_MODEL or 'Local GGUF'}"
            )
        else:
            return f"Embedding request failed (HTTP {r.status_code}): {r.text}"
    except Exception as e:
        return f"HTTP error during embedding: {e}"


@mcp.tool()
def local_complete(
    prompt: str,
    max_tokens: int = 256,
    port: int = DEFAULT_PORT
) -> str:
    """
    Executes local generative text completion.
    Note: Requires a generative model to be loaded (not embedding-only models).
    Args:
        prompt: Text prompt to generate completion for.
        max_tokens: Maximum tokens to generate (default: 256).
        port: Server port (default: 8081).
    """
    if not _check_health(port):
        return f"Local server is OFFLINE on port {port}. Start it with local_server_start() using a generative model."

    if _ACTIVE_MODEL and "embed" in _ACTIVE_MODEL.lower():
        return (
            f"Notice: The currently loaded model ({_ACTIVE_MODEL}) is an EMBEDDING-ONLY model.\n"
            f"For text completions, load a generative GGUF model (e.g. Qwen, LLaMA, Phi, or Gemma-IT) "
            f"via local_server_start(model_name='your-generative-model.gguf', embedding_mode=False)."
        )

    url = f"http://127.0.0.1:{port}/completion"
    try:
        r = requests.post(
            url,
            json={"prompt": prompt, "n_predict": max_tokens, "temperature": 0.7},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("content", "")
        else:
            return f"Completion failed (HTTP {r.status_code}): {r.text}"
    except Exception as e:
        return f"Completion error: {e}"


if __name__ == "__main__":
    mcp.run()
