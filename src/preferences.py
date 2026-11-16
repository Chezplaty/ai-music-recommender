"""
Natural-language preference extraction.

Turns a free-text request ("upbeat music for the gym, no country") into a
UserProfile the existing scorer can use.

Flow:
  1. build_vocabulary()      - collect the genres/moods that actually exist in the catalog
  2. Gemini (LLM)            - reads the request and returns JSON matching ExtractedPreferences
  3. validate_preferences()  - code-side guardrail: drops anything not in the catalog vocabulary
                               (recorded in `unmatched`), clamps numbers, caps song count
  4. ParsedRequest           - UserProfile + extra constraints for the agent, plus a
                               `search_description` used as the RAG search query (src/rag/)

The LLM only *interprets* the request. It never picks songs; the scorer does.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

from recommender import Song, UserProfile

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"  # override with the GEMINI_MODEL env var
# Filling in a JSON form doesn't need deep reasoning; default thinking made a
# trivial request take ~25s. Override with GEMINI_THINKING (low/medium/high;
# gemini-3.8-flash rejects "minimal" with a 400 error).
DEFAULT_THINKING_LEVEL = "low"
REQUEST_TIMEOUT_MS = 60_000  # SDK default is no timeout, which looks like a hang
DEFAULT_NUM_SONGS = 10
MAX_NUM_SONGS = 20
MAX_SEARCH_DESCRIPTION_CHARS = 500  # the embedding model truncates long input anyway

# Must match derive_mood() in scripts/build_catalog.py, so the LLM's idea of a
# mood lines up with how moods were assigned in the catalog.
MOOD_DEFINITIONS = {
    "euphoric": "very high energy, positive, very danceable",
    "happy": "energetic and positive",
    "angry": "very high energy, very dark/negative",
    "intense": "energetic but dark or tense",
    "relaxed": "moderate-low energy, positive",
    "chill": "low energy, positive, calm",
    "melancholic": "low energy, somewhat sad or wistful",
    "sad": "low energy, very sad",
}


class ExtractedPreferences(BaseModel):
    """Schema Gemini must fill in. Field descriptions are sent to the model."""
    genres: List[str] = Field(description="Genres from the allowed list that fit the request. Empty if none fit.")
    moods: List[str] = Field(description="Moods from the allowed list that fit the request. Empty if none fit.")
    target_energy: float = Field(description="0.0 = very calm, 1.0 = very intense. Use 0.5 if not implied.")
    target_valence: float = Field(description="0.0 = dark/sad, 1.0 = bright/happy. Use 0.5 if not implied.")
    target_acousticness: float = Field(
        description="0.0 = electronic/heavily produced, 1.0 = fully acoustic/unplugged. "
        "Use 0.5 if not mentioned. Scale with intensity: 'not too acoustic' ~0.3, "
        "'no acoustic at all' ~0.05, 'a bit acoustic' ~0.65, 'only acoustic' ~0.95."
    )
    num_songs: int = Field(description=f"How many songs the user asked for. Use {DEFAULT_NUM_SONGS} if not stated.")
    reference_artists: List[str] = Field(description="Artists the user mentions as examples of what they like.")
    exclude_artists: List[str] = Field(description="Artists the user explicitly does NOT want.")
    discover_new_artists: bool = Field(
        description="True if the user wants to discover new or different artists (so the reference "
        "artists themselves should not appear in the playlist). False otherwise."
    )
    unmatched_requests: List[str] = Field(
        description="Genres, moods, or other wishes the user mentioned that are NOT in the allowed lists."
    )
    search_description: str = Field(
        description="One or two sentences describing the SOUND the user wants, written like a music "
        "catalog entry: genre style, mood, energy, instrumentation, production. Translate artist, album "
        "and activity references into sonic terms (e.g. \"like Frank Ocean's Blonde\" -> \"atmospheric "
        "alternative r&b, dreamy and mellow, sparse production, layered soulful vocals\"; \"for studying\" "
        "-> \"calm, mostly instrumental, non-distracting background music\"). Used for semantic search."
    )


@dataclass
class ParsedRequest:
    """Validated output of the extraction step, ready for the recommender/agent."""
    profile: UserProfile
    num_songs: int
    reference_artists: List[str] = field(default_factory=list)
    exclude_artists: List[str] = field(default_factory=list)
    discover_new_artists: bool = False  # True -> agent leaves reference artists out of the playlist
    unmatched: List[str] = field(default_factory=list)  # things we can't satisfy from the catalog
    search_description: str = ""  # sonic rewrite of the request, used as the RAG search query
    raw_llm_output: dict = field(default_factory=dict)  # kept for transparency / debugging


class ExtractionError(Exception):
    """Raised when the LLM response can't be turned into valid preferences."""


def build_vocabulary(catalog: List[Song]) -> dict:
    return {
        "genres": sorted({s.genre for s in catalog}),
        "moods": sorted({s.mood for s in catalog}),
    }


def build_system_prompt(vocab: dict) -> str:
    moods = "\n".join(f"  - {m}: {MOOD_DEFINITIONS.get(m, '')}" for m in vocab["moods"])
    return f"""You convert a music request into structured listening preferences.

Allowed genres (use these exact strings):
  {", ".join(vocab["genres"])}

Allowed moods (use these exact strings):
{moods}

Rules:
- Only put allowed genres/moods in `genres` and `moods`. Pick the closest allowed
  ones when the user's wording differs (e.g. "rap" -> "hip hop", "upbeat" -> "happy").
- If the user asks for a genre or mood with no reasonable match in the allowed
  lists, put their wording in `unmatched_requests` instead of guessing.
- Infer energy/valence from the activity and mood (studying -> lower energy,
  workout -> high energy, heartbreak -> low valence).
- Do not invent preferences the user did not express or imply.
"""


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def validate_preferences(extracted: ExtractedPreferences, vocab: dict) -> ParsedRequest:
    """Guardrail: never trust the LLM output blindly. Pure function, no API calls."""
    allowed_genres, allowed_moods = set(vocab["genres"]), set(vocab["moods"])

    genres_in = [g.strip().lower() for g in extracted.genres]
    moods_in = [m.strip().lower() for m in extracted.moods]

    genres = _dedupe([g for g in genres_in if g in allowed_genres])
    moods = _dedupe([m for m in moods_in if m in allowed_moods])
    rejected = [x for x in genres_in if x not in allowed_genres] + [x for x in moods_in if x not in allowed_moods]
    if rejected:
        logger.warning("LLM returned values outside the catalog vocabulary: %s", rejected)

    num_songs = extracted.num_songs if extracted.num_songs > 0 else DEFAULT_NUM_SONGS

    return ParsedRequest(
        profile=UserProfile(
            genres=genres,
            moods=moods,
            target_energy=_clamp(extracted.target_energy),
            target_valence=_clamp(extracted.target_valence),
            target_acousticness=_clamp(extracted.target_acousticness),
        ),
        num_songs=min(num_songs, MAX_NUM_SONGS),
        reference_artists=_dedupe([a.strip() for a in extracted.reference_artists]),
        exclude_artists=_dedupe([a.strip() for a in extracted.exclude_artists]),
        discover_new_artists=bool(extracted.discover_new_artists),
        unmatched=_dedupe([u.strip() for u in extracted.unmatched_requests] + rejected),
        search_description=extracted.search_description.strip()[:MAX_SEARCH_DESCRIPTION_CHARS],
        raw_llm_output=extracted.model_dump(),
    )


def make_client():
    """Create a Gemini client. Reads GEMINI_API_KEY from the environment (or .env)."""
    from google import genai  # imported here so tests don't need the SDK configured
    from google.genai import types

    if not os.environ.get("GEMINI_API_KEY"):
        raise ExtractionError("GEMINI_API_KEY is not set. Add it to a .env file in the project root.")
    return genai.Client(http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


def extract_preferences(query: str, catalog: List[Song], client=None, model: Optional[str] = None) -> ParsedRequest:
    """Natural-language request -> validated ParsedRequest.

    `client` can be injected (tests pass a fake), otherwise a real Gemini client is created.
    """
    from google.genai import types

    if not query or not query.strip():
        raise ExtractionError("Empty request.")

    client = client or make_client()
    model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    vocab = build_vocabulary(catalog)

    thinking_level = os.environ.get("GEMINI_THINKING", DEFAULT_THINKING_LEVEL).upper()

    try:
        response = client.models.generate_content(
            model=model,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(vocab),
                response_mime_type="application/json",
                response_schema=ExtractedPreferences,
                temperature=0.0,  # same request -> same profile, as far as the model allows
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            ),
        )
    except Exception as e:  # timeouts, auth errors, unknown model, network...
        raise ExtractionError(f"Gemini request failed ({type(e).__name__}): {e}") from e

    extracted = response.parsed
    if not isinstance(extracted, ExtractedPreferences):
        # SDK couldn't parse it for us; try the raw text before giving up.
        try:
            extracted = ExtractedPreferences.model_validate(json.loads(response.text or ""))
        except (json.JSONDecodeError, ValidationError) as e:
            raise ExtractionError(f"Could not parse LLM response: {e}") from e

    parsed = validate_preferences(extracted, vocab)
    logger.info("Extracted preferences for %r: %s", query, parsed.raw_llm_output)
    return parsed
