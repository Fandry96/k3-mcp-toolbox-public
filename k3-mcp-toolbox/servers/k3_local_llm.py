#!/usr/bin/env python3
"""
k3-local-llm — Local Model Bridge MCP Server
Orchestrates bundled llama-server.exe and GGUF models for offline embeddings and inference.
Provides auto-starting, model discovery, and health telemetry.
"""

import os
import sys
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
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
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


def _detect_llama_backend() -> Dict[str, Any]:
    """Detects available execution backend libraries in the bundled llama-server directory."""
    has_cuda = (LLAMA_BIN_DIR / "ggml-cuda.dll").exists()
    has_vulkan = (LLAMA_BIN_DIR / "ggml-vulkan.dll").exists()
    cpu_dlls = sorted([p.name for p in LLAMA_BIN_DIR.glob("ggml-cpu-*.dll")]) if LLAMA_BIN_DIR.exists() else []

    if has_cuda:
        mode = "CUDA"
    elif has_vulkan:
        mode = "Vulkan"
    else:
        mode = "CPU-ONLY (Missing ggml-cuda.dll)"

    return {
        "has_cuda": has_cuda,
        "has_vulkan": has_vulkan,
        "cpu_dlls": cpu_dlls,
        "mode": mode,
    }


def _validate_rtx5080_vram_envelope(model_identifier: str, n_gpu_layers: int = 99) -> Optional[str]:
    """Validates whether a model and layer offload configuration fits safely within the 16 GB RTX 5080 VRAM envelope."""
    if not model_identifier:
        return None
    ident_lower = model_identifier.lower()
    if ".litertlm" in ident_lower:
        return (
            "[VRAM_GUARD_ERROR] LiteRT .litertlm checkpoints (16.8 GB) require >= 24 GB VRAM "
            "and cause severe PCIe paging on a 16 GB RTX 5080 (~14.4 GB usable). "
            "Use LocalOpenAIAgentConfig(...).lightweight() with GGUF instead."
        )
    if "31b" in ident_lower:
        return (
            "[VRAM_GUARD_ERROR] Gemma 4 31B exceeds the 16 GB RTX 5080 VRAM envelope "
            "even with Blackwell NVFP4 quantization. Use gemma-4-12b (Q4_0) or gemma-4-26b-a4b (MoE with DDR5 offload)."
        )
    if "26b" in ident_lower and n_gpu_layers >= 99:
        return (
            "[VRAM_GUARD_ERROR] Gemma 4 26B A4B (~16.0 GB) cannot fit 100% of layers (-ngl 99) "
            "inside ~14.4 GB usable RTX 5080 VRAM without PCIe paging. Set n_gpu_layers=28 for partial DDR5 RAM offload."
        )
    return None


def _find_gguf_models() -> List[Dict[str, Any]]:
    """Discovers all .gguf model files in both MODELS_DIR and HuggingFace cache snapshots with Windows pointer resolution."""
    models: List[Dict[str, Any]] = []
    seen_names = set()
    search_paths: List[Path] = []

    if MODELS_DIR.exists():
        search_paths.extend(MODELS_DIR.rglob("*.gguf"))
    if HF_CACHE_DIR.exists():
        search_paths.extend(HF_CACHE_DIR.glob("models--*--*/snapshots/*/*.gguf"))

    for p in search_paths:
        if p.name in seen_names:
            continue
        seen_names.add(p.name)

        name_lower = p.name.lower()
        if "mmproj" in name_lower:
            m_type = "mmproj"
        elif any(k in name_lower for k in ["embed", "bge", "nomic", "minilm", "e5"]):
            m_type = "embedding"
        else:
            m_type = "generative"

        sz = 0
        try:
            sz = p.stat().st_size
        except Exception:
            sz = 0

        resolved_path = str(p)
        if sz < 1024:
            try:
                r = p.resolve()
                if r.exists() and r.stat().st_size >= 1024:
                    sz = r.stat().st_size
                    resolved_path = str(r)
                else:
                    blobs_dir = p.parents[2] / "blobs"
                    if blobs_dir.exists():
                        blob_files = [b for b in blobs_dir.iterdir() if b.is_file()]
                        if blob_files:
                            if "mmproj" in name_lower:
                                target_blob = min(blob_files, key=lambda b: b.stat().st_size)
                            else:
                                target_blob = max(blob_files, key=lambda b: b.stat().st_size)
                            sz = target_blob.stat().st_size
                            resolved_path = str(target_blob)
            except Exception:
                pass
        else:
            try:
                resolved_path = str(p.resolve())
            except Exception:
                resolved_path = str(p)

        sz_mb = sz / (1024 * 1024)
        models.append({
            "name": p.name,
            "path": str(p),
            "resolved_path": resolved_path,
            "size_mb": round(sz_mb, 1),
            "type": m_type,
        })
    return models


@mcp.tool()
def local_models_list() -> str:
    """List all local GGUF model files in the K3 models directory with size and embedding/generative classification. Use INSTEAD OF Get-ChildItem or find/ls shell commands when checking installed offline GGUF models. Do NOT use for: querying remote cloud model catalogs or downloading models (use local_model_download)."""
    models = _find_gguf_models()
    backend = _detect_llama_backend()
    if not models:
        return f"No .gguf models found in {MODELS_DIR} (Backend: {backend['mode']})."

    lines = [
        f"=== Local GGUF Model Registry ===",
        f"Backend: {backend['mode']}",
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
        "filename": "bge-small-en-v1.5-q8_0.gguf",
        "url": "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
        "type": "embedding",
        "size_mb": 35.5,
        "description": "Ultra-compact 384-dim embedding model (only 35MB). Recommended for fast offline RAG.",
    },
    "qwen2.5-0.5b": {
        "name": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "type": "generative",
        "size_mb": 398.0,
        "description": "Ultra-lightweight 0.5B instruction-tuned model for local completions.",
    },
    "smollm2-360m": {
        "name": "smollm2-360m-instruct-q8_0.gguf",
        "filename": "smollm2-360m-instruct-q8_0.gguf",
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf",
        "type": "generative",
        "size_mb": 387.0,
        "description": "HuggingFace SmolLM2 compact instruction model.",
    },
    "gemma-4-12b": {
        "name": "gemma-4-12B-it-Q4_0.gguf",
        "filename": "gemma-4-12B-it-Q4_0.gguf",
        "fallback_filenames": ["gemma-4-12b-it-Q4_K_M.gguf", "gemma-4-12B-it-Q4_K_M.gguf"],
        "url": "https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q4_0.gguf",
        "size_mb": 6885.2,
        "type": "generative",
        "vram_gb": 6.8,
        "n_gpu_layers": 99,
        "ctx_size": 16384,
        "flash_attn": True,
        "description": "RTX 5080 Speed Sweet Spot (Google official QAT Q4_0, 67-90.4 tok/s, 6.8 GB VRAM, >7 GB GDDR7 KV headroom, 100% GPU resident)",
    },
    "gemma-4-26b-a4b": {
        "name": "gemma-4-26B-A4B-it-Q4_0.gguf",
        "filename": "gemma-4-26B-A4B-it-Q4_0.gguf",
        "fallback_filenames": ["gemma-4-26B-A4B-it-Q4_K_M.gguf", "gemma-4-26b-a4b-it-q4_k_m.gguf"],
        "url": "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q4_0.gguf",
        "size_mb": 13941.0,
        "type": "generative",
        "vram_gb": 13.5,
        "n_gpu_layers": 28,
        "ctx_size": 8192,
        "flash_attn": True,
        "description": "RTX 5080 Higher-Reasoning MoE (Google official QAT Q4_0, 4B active/token, partial CPU/DDR5 RAM offload -ngl 28 for 32GB+ hosts)",
    },
    "dflash-gemma-4-26b-a4b": {
        "name": "dflash-gemma-4-26B-A4B-it-Q8_0.gguf",
        "filename": "dflash-gemma-4-26B-A4B-it-Q8_0.gguf",
        "url": "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/dflash-gemma-4-26B-A4B-it-Q8_0.gguf",
        "type": "generative",
        "size_mb": 450.5,
        "description": "DFlash speculative drafter weights for Gemma 4 26B A4B MoE acceleration (Q8_0).",
    },
    "mtp-gemma-4-12b": {
        "name": "mtp-gemma-4-12B-it-Q4_0.gguf",
        "filename": "mtp-gemma-4-12B-it-Q4_0.gguf",
        "url": "https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/resolve/main/mtp-gemma-4-12B-it-Q4_0.gguf",
        "type": "generative",
        "size_mb": 242.0,
        "description": "MTP speculative drafter weights for Gemma 4 12B IT acceleration (Q4_0).",
    },
}


@mcp.tool()
def local_model_download(model_key_or_url: str = "list") -> str:
    """Download a curated or custom HuggingFace GGUF model into the K3 offline models directory. Use INSTEAD OF manual curl/wget or Invoke-WebRequest downloads when provisioning local GGUF weights. Do NOT use for: listing already-installed local models (use local_models_list) or non-GGUF weights.
    Args:
        model_key_or_url: 'list', curated key ('bge-small-en', 'qwen2.5-0.5b', 'smollm2-360m', 'gemma-4-12b', 'gemma-4-26b-a4b', 'dflash-gemma-4-26b-a4b', 'mtp-gemma-4-12b'), or HTTPS .gguf URL.
    """
    cleaned = model_key_or_url.strip()
    if cleaned.lower() == "list" or (cleaned not in CURATED_MODELS and not cleaned.startswith("http")):
        lines = [
            "=== Curated Downloadable Models ===",
            "Specify one of the keys below to download, or pass a direct HTTPS URL to a .gguf file:\n",
        ]
        for k, info in CURATED_MODELS.items():
            fname = info.get("filename") or info.get("name")
            lines.append(f"• Key: {k:<18} | Size: {info['size_mb']:>7.1f} MB | Type: {info['type'].upper()}")
            lines.append(f"  Description: {info['description']}")
            lines.append(f"  Filename:    {fname}\n")
        return "\n".join(lines)

    if cleaned in CURATED_MODELS:
        info = CURATED_MODELS[cleaned]
        target_name = info.get("filename") or info["name"]
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
    """Check whether the local llama-server.exe instance is online and healthy on the target TCP port. Use INSTEAD OF curl http://127.0.0.1:8081/health or Get-Process llama-server shell checks when verifying local LLM readiness. Do NOT use for: general multi-port conflict scanning (use ops_check_ports).
    Args:
        port: TCP port to check (default: 8081).
    """
    global _SERVER_PROC, _ACTIVE_MODEL, _ACTIVE_PORT
    is_live = _check_health(port)
    backend = _detect_llama_backend()

    lines = ["=== Local LLaMA Server Status ==="]
    if is_live:
        lines.append(f"Status          : ONLINE (Healthy)")
        lines.append(f"Port            : {port}")
        lines.append(f"Model           : {_ACTIVE_MODEL or 'Detected external/running instance'}")
        lines.append(f"Backend         : {backend['mode']}")
        lines.append(f"OpenAI Endpoint: http://127.0.0.1:{port}/v1 (LocalOpenAIAgentConfig.lightweight() ready)")
        if _SERVER_PROC and _SERVER_PROC.poll() is None:
            lines.append(f"PID             : {_SERVER_PROC.pid}")
        try:
            r = requests.get(f"http://127.0.0.1:{port}/props", timeout=2)
            if r.status_code == 200:
                props = r.json()
                lines.append(f"Server Ver      : {props.get('version', 'llama.cpp')}")
        except Exception:
            pass
    else:
        lines.append(f"Status          : OFFLINE (Port {port} closed)")
        lines.append(f"Backend         : {backend['mode']}")
        lines.append(f"OpenAI Endpoint: http://127.0.0.1:{port}/v1 (LocalOpenAIAgentConfig.lightweight() ready)")
        lines.append("Use local_server_start() to boot llama-server.exe.")

    return "\n".join(lines)


def _resolve_model_path(model_identifier: str) -> Optional[Path]:
    """Resolves a model name, curated key, or filename to a local GGUF file path (including fallback variants)."""
    if not model_identifier:
        return DEFAULT_MODEL if DEFAULT_MODEL.exists() else None

    search_candidates = [model_identifier]
    if model_identifier in CURATED_MODELS:
        curated = CURATED_MODELS[model_identifier]
        primary = curated.get("filename") or curated.get("name") or model_identifier
        search_candidates = [primary] + curated.get("fallback_filenames", [])
    elif "26b" in model_identifier.lower():
        search_candidates.extend(["gemma-4-26B-A4B-it-Q4_0.gguf", "gemma-4-26B-A4B-it-Q4_K_M.gguf"])

    for cand in search_candidates:
        cand_path = Path(cand)
        if cand_path.exists() and cand_path.is_file():
            return cand_path
        if (MODELS_DIR / cand).exists() and (MODELS_DIR / cand).is_file():
            return MODELS_DIR / cand

    all_models = _find_gguf_models()
    for cand in search_candidates:
        cand_lower = cand.lower()
        cand_stem = cand_lower.replace(".gguf", "")
        match = next(
            (
                m for m in all_models
                if cand_lower in m["name"].lower()
                or cand_lower in m.get("path", "").lower()
                or cand_stem in m["name"].lower()
            ),
            None
        )
        if match:
            resolved = match.get("resolved_path")
            if resolved and Path(resolved).exists():
                return Path(resolved)
            return Path(match["path"])
    return None


@mcp.tool()
def local_server_start(
    model_name: str = "",
    port: int = DEFAULT_PORT,
    embedding_mode: bool = True,
    n_gpu_layers: int = 99,
    ctx_size: int = 8192,
    flash_attn: bool = True,
) -> str:
    """Boot the bundled llama-server.exe background daemon with a local GGUF model and poll until healthy. Use INSTEAD OF manual llama-server.exe CLI invocations when starting local embedding or completion inference. Do NOT use for: running cloud Gemini models.
    Args:
        model_name: GGUF filename or curated key (defaults to embeddinggemma-300M-Q8_0.gguf).
        port: TCP port (default: 8081).
        embedding_mode: Enable --embedding flag.
        n_gpu_layers: Number of layers to offload to GPU (default: 99).
        ctx_size: Context window size (default: 8192).
        flash_attn: Enable Flash Attention (default: True).
    """
    global _SERVER_PROC, _ACTIVE_MODEL, _ACTIVE_PORT

    if _check_health(port):
        return (
            f"Server already online and healthy at http://127.0.0.1:{port} (Model: {_ACTIVE_MODEL or 'active'}).\n"
            f"OpenAI Endpoint: http://127.0.0.1:{port}/v1 (LocalOpenAIAgentConfig.lightweight() ready)"
        )

    if port < 1 or port > 65535:
        return "Error: Port must be an integer between 1 and 65535."

    if not LLAMA_EXE.exists():
        return f"Error: llama-server.exe binary not found at {LLAMA_EXE}"

    # Resolve target model and parameters
    target_model_path: Optional[Path] = None
    if model_name:
        search_target = model_name
        if model_name in CURATED_MODELS:
            curated = CURATED_MODELS[model_name]
            search_target = curated.get("filename") or curated.get("name") or model_name
            if n_gpu_layers == 99 and "n_gpu_layers" in curated:
                n_gpu_layers = curated["n_gpu_layers"]
            if ctx_size == 8192 and "ctx_size" in curated:
                ctx_size = curated["ctx_size"]
            if "flash_attn" in curated:
                flash_attn = curated["flash_attn"]

        vram_err = _validate_rtx5080_vram_envelope(model_name, n_gpu_layers) or _validate_rtx5080_vram_envelope(search_target, n_gpu_layers)
        if vram_err:
            return vram_err

        target_model_path = _resolve_model_path(model_name)
        if not target_model_path:
            return f"Model '{model_name}' not found. Use local_models_list() to see available files."
    else:
        target_model_path = DEFAULT_MODEL

    if not target_model_path or not target_model_path.exists():
        return f"Error: Model file does not exist at {target_model_path}"

    cmd = [
        str(LLAMA_EXE),
        "-m", str(target_model_path),
        "--port", str(port),
        "-c", str(ctx_size),
        "-ngl", str(n_gpu_layers),
        "-t", "4",
    ]
    if flash_attn:
        cmd.append("-fa")
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
                backend = _detect_llama_backend()
                return (
                    f"SUCCESS: llama-server.exe launched at http://127.0.0.1:{port}\n"
                    f"  - PID            : {_SERVER_PROC.pid}\n"
                    f"  - Model          : {target_model_path.name}\n"
                    f"  - Mode           : {'Embedding' if embedding_mode else 'Generative'}\n"
                    f"  - Backend        : {backend['mode']}\n"
                    f"  - GPU Layers     : {n_gpu_layers}\n"
                    f"  - Context Size   : {ctx_size}\n"
                    f"  - Flash Attention: {flash_attn}\n"
                    f"  - OpenAI Endpoint: http://127.0.0.1:{port}/v1 (LocalOpenAIAgentConfig.lightweight() ready)"
                )

        if _SERVER_PROC.poll() is not None:
            _, err = _SERVER_PROC.communicate(timeout=2)
            return f"Server process crashed immediately with exit code {_SERVER_PROC.returncode}:\n{err.decode('utf-8', errors='replace')}"

        return (
            f"Started PID {_SERVER_PROC.pid}, but port {port} not responsive yet. Check local_server_status().\n"
            f"OpenAI Endpoint: http://127.0.0.1:{port}/v1 (LocalOpenAIAgentConfig.lightweight() ready)"
        )
    except Exception as e:
        return f"Failed to start llama-server: {e}"


@mcp.tool()
def local_server_stop() -> str:
    """Stop the managed llama-server.exe background process cleanly and release its TCP port. Use INSTEAD OF raw taskkill /IM llama-server.exe or Stop-Process shell commands when shutting down the local model server. Do NOT use for: killing non-llama processes on other ports (use ops_free_port)."""
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
    """Generate offline vector embeddings via the local llama-server GGUF instance (auto-starting if offline). Use INSTEAD OF cloud Gemini embedding API calls or custom curl POSTs to /embedding when offline, zero-cost, or local CPU/GPU embedding is requested. Do NOT use for: querying the 768-dim MRL index (use mrl_search).
    Args:
        text: Input text to embed.
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
    """Generate offline text completions using a local generative GGUF model hosted on llama-server.exe. Use INSTEAD OF cloud LLM API calls or raw curl POSTs to /completion when offline, air-gapped, or local GPU/CPU inference is requested. Do NOT use for: vector embeddings (use local_embed) or frontier multi-file reasoning.
    Args:
        prompt: Text prompt.
        max_tokens: Max tokens (default: 256).
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
