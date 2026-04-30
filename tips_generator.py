import json
import random
import time
from dataclasses import dataclass

from google import genai
from google.genai import types

MAX_RETRIES = 5
BASE_DELAY = 2  # seconds


def _log(level: str, message: str) -> None:
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} | {level:<5} | {message}")


def _parse_tips_response(raw: str) -> list["Tip"]:
    """Parse raw Gemini response into a list of Tip objects."""
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(line for line in lines if not line.startswith("```")).strip()

    data = json.loads(raw)
    if not isinstance(data, list) or len(data) != 5:
        raise ValueError(f"Expected list of 5 tips, got: {raw}")

    return [Tip(title=item["title"], body=item["body"]) for item in data]


@dataclass
class Tip:
    title: str
    body: str


FALLBACK_TIPS = [
    Tip(
        title="🎯 *Prioriza lo Esencial*",
        body="Elige las tres tareas más importantes y termínalas antes de cualquier otra cosa.",
    ),
    Tip(
        title="⚡ *Elimina las Distracciones*",
        body="Silencia notificaciones innecesarias mientras trabajas en lo que realmente importa.",
    ),
    Tip(
        title="📅 *Planifica tu Mañana*",
        body="Dedica cinco minutos al final del día para organizar tus prioridades del siguiente día.",
    ),
    Tip(
        title="🔥 *Mantén el Impulso*",
        body="Trabaja en bloques concentrados y toma descansos cortos para renovar tu energía.",
    ),
    Tip(
        title="✅ *Celebra tus Logros*",
        body="Reconoce cada pequeño avance para mantener tu motivación y confianza en alto.",
    ),
]


def generate_tips(api_key: str) -> list[Tip]:
    """Use Gemini with Google Search to generate 5 current performance tips.

    Retries on transient errors (503) with exponential backoff and jitter.
    Falls back to curated evergreen tips if all retries are exhausted.
    """
    client = genai.Client(api_key=api_key)

    prompt = (
        "IMPORTANTE: Responde ÚNICAMENTE en español mexicano. "
        "Ninguna palabra en inglés. Todo el contenido debe estar en español mexicano.\n\n"
        "Busca los consejos más actuales y prácticos para mejorar el rendimiento "
        "personal y la ejecución en el trabajo en 2026. "
        "Devuelve exactamente 5 consejos como un arreglo JSON con esta estructura:\n"
        '[{"title": "EMOJI *Título en Español*", "body": "una oración accionable"}]\n'
        "Reglas para el título: empieza con un emoji relevante, luego escribe el título "
        "de 3-5 palabras entre asteriscos para negrita de WhatsApp: *Título Aquí*. "
        "El body debe ser una sola oración accionable de máximo 20 palabras en español mexicano. "
        "Enfócate en ejecución, productividad y momentum. "
        "Devuelve ÚNICAMENTE el arreglo JSON, sin bloques de código, sin texto extra."
    )

    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.7,
                ),
            )

            raw = response.text.strip()
            return _parse_tips_response(raw)

        except Exception as exc:
            last_exception = exc
            error_str = str(exc)
            is_unavailable = "503" in error_str or "UNAVAILABLE" in error_str

            if is_unavailable and attempt < MAX_RETRIES - 1:
                delay = (BASE_DELAY * (2**attempt)) + random.uniform(0, 1)
                _log(
                    "WARN",
                    f"Gemini unavailable (attempt {attempt + 1}/{MAX_RETRIES}), "
                    f"retrying in {delay:.1f}s...",
                )
                time.sleep(delay)
                continue
            raise

    _log("WARN", f"All {MAX_RETRIES} retries exhausted, using fallback tips")
    return list(FALLBACK_TIPS)


def tips_to_template_variables(name: str, tips: list[Tip]) -> dict[str, str]:
    """Map correctly for the 7 variable WhatsApp template."""
    if len(tips) != 5:
        raise ValueError("Exactly 5 tips required")

    def fmt(tip: Tip) -> str:
        return f"{tip.title} – {tip.body}"

    return {
        "1": name,
        "2": fmt(tips[0]),
        "3": fmt(tips[1]),
        "4": fmt(tips[2]),
        "5": fmt(tips[3]),
        "6": fmt(tips[4]),
    }
