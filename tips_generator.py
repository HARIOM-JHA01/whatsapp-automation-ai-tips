import json
import os
from dataclasses import dataclass

from google import genai
from google.genai import types


@dataclass
class Tip:
    title: str
    body: str


def generate_tips(api_key: str) -> list[Tip]:
    """Use Gemini with Google Search to generate 5 current performance tips."""
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

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.7,
        ),
    )

    raw = response.text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(line for line in lines if not line.startswith("```")).strip()

    data = json.loads(raw)
    if not isinstance(data, list) or len(data) != 5:
        raise ValueError(f"Expected list of 5 tips, got: {raw}")

    return [Tip(title=item["title"], body=item["body"]) for item in data]


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
