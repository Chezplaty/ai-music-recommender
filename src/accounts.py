"""
Local user accounts and taste profiles for the Streamlit app.

Accounts live in a JSON file (default: app_data/users.json, git-ignored). Passwords
are never stored: only a random salt and a PBKDF2-SHA256 hash. This is fine for a
local demo, but it is not a production auth system (no rate limiting, no sessions
server-side, single file).
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from preferences import DEFAULT_NUM_SONGS, ParsedRequest
from recommender import UserProfile
from rag.ingestion import describe_acousticness, describe_energy, describe_valence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = PROJECT_ROOT / "app_data" / "users.json"
PBKDF2_ITERATIONS = 200_000
USERNAME_RE = re.compile(r"^[a-z0-9_]{3,30}$")
MIN_PASSWORD_LENGTH = 6


class AccountError(Exception):
    """A user-facing problem: bad username, taken name, wrong password..."""


@dataclass
class TasteProfile:
    """What a listener told us about themselves at sign-up. Used as defaults for every request."""
    display_name: str
    genres: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    energy: float = 0.5
    valence: float = 0.5
    acousticness: float = 0.5
    favorite_artists: List[str] = field(default_factory=list)
    avoid_artists: List[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """How the profile is described to Gemini when it interprets a request."""
        lines = [
            f"favorite genres: {', '.join(self.genres) or 'no preference'}",
            f"favorite moods: {', '.join(self.moods) or 'no preference'}",
            f"usual energy: {self.energy:.2f} ({describe_energy(self.energy)})",
            f"usual positivity (valence): {self.valence:.2f} ({describe_valence(self.valence)})",
            f"usual acousticness: {self.acousticness:.2f} ({describe_acousticness(self.acousticness)})",
        ]
        if self.favorite_artists:
            lines.append(f"favorite artists: {', '.join(self.favorite_artists)}")
        if self.avoid_artists:
            lines.append(f"never play: {', '.join(self.avoid_artists)}")
        return "\n".join(lines)

    def to_request(self, num_songs: int = DEFAULT_NUM_SONGS) -> ParsedRequest:
        """A request built straight from the profile, with no LLM call ("Quick mix")."""
        sound = [f"{', '.join(self.genres)} music" if self.genres else "music",
                 f"{', '.join(self.moods)} mood" if self.moods else "",
                 describe_energy(self.energy), describe_valence(self.valence),
                 describe_acousticness(self.acousticness)]
        return ParsedRequest(
            profile=UserProfile(genres=list(self.genres), moods=list(self.moods), target_energy=self.energy,
                                target_valence=self.valence, target_acousticness=self.acousticness),
            num_songs=num_songs,
            reference_artists=list(self.favorite_artists),
            exclude_artists=list(self.avoid_artists),
            search_description="; ".join(s for s in sound if s),
        )


def parse_artist_list(text: str) -> List[str]:
    """'Frank Ocean, SZA\\nThe Weeknd' -> ['Frank Ocean', 'SZA', 'The Weeknd'] (deduped, order kept)."""
    out = []
    for part in re.split(r"[,\n]", text or ""):
        part = part.strip()
        if part and part.lower() not in {a.lower() for a in out}:
            out.append(part)
    return out


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS).hex()


class UserStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or os.environ.get("MUSIC_APP_USERS", DEFAULT_STORE))

    # --- file I/O ---
    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        """Write atomically, so a crash mid-write can't corrupt every account."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    # --- accounts ---
    @staticmethod
    def normalize(username: str) -> str:
        return (username or "").strip().lower()

    def create_user(self, username: str, password: str) -> str:
        username = self.normalize(username)
        if not USERNAME_RE.match(username):
            raise AccountError("Usernames are 3-30 characters: letters, numbers and underscores.")
        if len(password or "") < MIN_PASSWORD_LENGTH:
            raise AccountError(f"Passwords need at least {MIN_PASSWORD_LENGTH} characters.")
        data = self._load()
        if username in data:
            raise AccountError("That username is taken.")
        salt = secrets.token_bytes(16)
        data[username] = {
            "salt": salt.hex(),
            "password_hash": _hash_password(password, salt),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profile": None,
        }
        self._save(data)
        return username

    def authenticate(self, username: str, password: str) -> str:
        username = self.normalize(username)
        record = self._load().get(username)
        # Same message for unknown user and wrong password, so names can't be probed.
        if record is None or not hmac.compare_digest(
                record["password_hash"], _hash_password(password or "", bytes.fromhex(record["salt"]))):
            raise AccountError("Wrong username or password.")
        return username

    # --- profiles ---
    def get_profile(self, username: str) -> Optional[TasteProfile]:
        record = self._load().get(self.normalize(username))
        if not record or not record.get("profile"):
            return None
        return TasteProfile(**record["profile"])

    def save_profile(self, username: str, profile: TasteProfile) -> None:
        data = self._load()
        username = self.normalize(username)
        if username not in data:
            raise AccountError("Unknown user.")
        data[username]["profile"] = asdict(profile)
        self._save(data)
