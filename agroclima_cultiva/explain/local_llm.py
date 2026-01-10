from __future__ import annotations

import requests

# ---------------------------------------------------------------------
# Configuração do Ollama local
# ---------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:latest"


def llm_local(prompt: str) -> str:
    """
    Chamada ao LLM local via Ollama (HTTP API).

    Contrato:
    - entrada: prompt (str)
    - saída: texto gerado (str)
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
    except Exception as e:
        raise RuntimeError(f"Falha ao conectar ao Ollama em {OLLAMA_URL}: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(
            f"Erro do Ollama (status={resp.status_code}): {resp.text}"
        )

    data = resp.json()

    # A resposta vem neste campo
    text = data.get("response", "")
    return text.strip()

