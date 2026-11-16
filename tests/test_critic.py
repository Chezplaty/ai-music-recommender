"""Tests for the AI critic (src/agent/critic.py) and how the workflow uses it.

A ScriptedReviewer stands in for Gemini: it returns pre-written ReviewResponses in
order and records what it was asked. So these test OUR side of the contract:
when the critic is called, what it is shown, how its answer is validated, and how
its plan drives the next attempt. Not whether Gemini gives good advice.
"""

from types import SimpleNamespace

import pytest

from preferences import LLMError, ParsedRequest, build_vocabulary
from recommender import UserProfile
from agent.critic import (CriticError, GeminiReviewer, ReviewContext, ReviewResponse, apply_review,
                          build_review_prompt)
from agent.evaluator import MAX_CANDIDATES, evaluate
from agent.planner import make_plan
from agent.generator import build_playlist
from agent.workflow import MAX_REVISIONS, run_agent


def make_request(**overrides) -> ParsedRequest:
    data = dict(
        profile=UserProfile(genres=["metal"], moods=["intense", "angry"], target_energy=0.9,
                            target_valence=0.3, target_acousticness=0.1),
        num_songs=5,
        search_description="heavy distorted guitars, dark and intense metal, loud",
    )
    data.update(overrides)
    return ParsedRequest(**data)


def response(approved=False, **overrides) -> ReviewResponse:
    """A critic answer that keeps the default metal plan unless overridden."""
    data = dict(
        approved=approved, issues=[] if approved else ["not quite right"], reasoning="because",
        next_search_text="heavy distorted guitars, dark and intense metal, loud",
        next_genres=["metal"], next_moods=["intense", "angry"], next_target_energy=0.9,
        next_target_valence=0.3, next_target_acousticness=0.1, remove_song_ids=[],
        next_n_candidates=30, search_within_genres=False,
    )
    data.update(overrides)
    return ReviewResponse(**data)


class ScriptedReviewer:
    """Returns the scripted responses in order (an Exception in the script is raised)."""

    def __init__(self, *script):
        self.script = list(script)
        self.contexts = []

    def __call__(self, ctx):
        self.contexts.append(ctx)
        item = self.script.pop(0) if self.script else response(approved=True)
        if isinstance(item, Exception):
            raise item
        return item


def run(catalog, retriever, reviewer, **request_overrides):
    return run_agent("loud angry metal for the gym", catalog, retriever,
                     request=make_request(**request_overrides), reviewer=reviewer)


def ids(playlist):
    return [item.song.id for item in playlist]


@pytest.fixture
def vocab(catalog):
    return build_vocabulary(catalog)


def first_attempt_context(catalog, retriever, vocab, mode="critique", request=None):
    request = request or make_request()
    strategy, _ = make_plan(request, catalog)
    hits = retriever.search(strategy.search_text, n_results=30)
    playlist = build_playlist([h.song for h in hits], request, strategy, {h.song.id: h.similarity for h in hits})
    return ReviewContext(mode=mode, query="loud angry metal for the gym", request=request, strategy=strategy,
                         playlist=playlist, evaluation=evaluate(playlist, request, strategy, catalog),
                         attempt_number=1, revisions_left=2, vocab=vocab)


# --- the AI controls the loop ------------------------------------------------------

def test_ai_approval_ends_the_loop_after_one_attempt(catalog, retriever):
    reviewer = ScriptedReviewer(response(approved=True))
    result = run(catalog, retriever, reviewer)
    assert len(result.attempts) == 1 and result.attempts[0].review.approved
    assert len(reviewer.contexts) == 1 and reviewer.contexts[0].mode == "critique"


def test_ai_rejection_drives_the_next_attempt(catalog, retriever):
    """Rules pass, the AI rejects and changes the plan; the next attempt uses the AI's plan."""
    reviewer = ScriptedReviewer(
        response(next_search_text="fast punk with shouted vocals", next_genres=["punk"], next_target_energy=0.8),
        response(approved=True),
    )
    result = run(catalog, retriever, reviewer)
    assert len(result.attempts) == 2
    second = result.attempts[1].strategy
    assert second.search_text == "fast punk with shouted vocals"
    assert second.profile.genres == ["punk"] and second.profile.target_energy == 0.8
    assert result.request.profile.genres == ["metal"]  # the user's original request is kept as-is
    assert any("revision 1 (AI)" in t["detail"] for t in result.trace)
    assert ids(result.playlist) == ids(result.attempts[1].playlist)


def test_songs_removed_by_the_ai_never_come_back(catalog, retriever):
    first = run(catalog, retriever, ScriptedReviewer(response(approved=True)))
    to_remove = [item.song.id for item in first.playlist[:2]]

    reviewer = ScriptedReviewer(response(remove_song_ids=to_remove), response(approved=True))
    result = run(catalog, retriever, reviewer)
    assert not {item.song.id for item in result.playlist} & set(to_remove)
    assert set(to_remove) <= set(result.attempts[1].strategy.banned_song_ids)


def test_ai_writes_the_fix_when_rules_fail(catalog, retriever):
    """Search text drifts to folk while the profile wants metal: rules fail, AI is asked to replan."""
    reviewer = ScriptedReviewer(response(search_within_genres=True), response(approved=True))
    result = run(catalog, retriever, reviewer, search_description="gentle acoustic folk guitar, soft vocals")
    assert not result.attempts[0].evaluation.passed
    assert reviewer.contexts[0].mode == "revise"
    assert result.attempts[1].strategy.genre_targeted
    assert result.attempts[-1].accepted


def test_ai_cannot_approve_a_playlist_that_failed_the_rules(catalog, retriever, vocab):
    ctx = first_attempt_context(catalog, retriever, vocab, mode="revise")
    review = apply_review(ctx, response(approved=True))
    assert not review.approved and review.revision is not None


def test_loop_stops_at_max_revisions_even_if_ai_keeps_rejecting(catalog, retriever):
    plans = [response(next_n_candidates=40 + 10 * i) for i in range(10)]
    result = run(catalog, retriever, ScriptedReviewer(*plans))
    assert len(result.attempts) == MAX_REVISIONS + 1
    assert result.playlist
    assert any("AI critic still had concerns" in n for n in result.notes)


def test_rejection_without_any_change_stops_instead_of_repeating(catalog, retriever):
    result = run(catalog, retriever, ScriptedReviewer(response()))  # rejects, copies the current plan
    assert len(result.attempts) == 1
    assert any("proposed no changes" in t["detail"] for t in result.trace)


def test_ai_approved_attempt_is_preferred_over_higher_scoring_rejected_one(catalog, retriever):
    """Attempt 1 (rejected) vs attempt 2 (approved): the approved one is returned."""
    reviewer = ScriptedReviewer(response(next_genres=["punk"], next_search_text="punk"), response(approved=True))
    result = run(catalog, retriever, reviewer)
    assert ids(result.playlist) == ids(result.attempts[1].playlist)


# --- Gemini unavailable -> fall back to rules ---------------------------------------

def test_unavailable_critic_accepts_when_rules_pass(catalog, retriever):
    result = run(catalog, retriever, ScriptedReviewer(CriticError("503 overloaded")))
    assert len(result.attempts) == 1 and result.attempts[0].accepted
    assert any("unavailable" in n for n in result.notes)


def test_unavailable_critic_falls_back_to_rule_revision_when_rules_fail(catalog, retriever):
    reviewer = ScriptedReviewer(CriticError("503 overloaded"), CriticError("503 overloaded"))
    result = run(catalog, retriever, reviewer, search_description="gentle acoustic folk guitar, soft vocals")
    assert any("rule fallback" in t["detail"] for t in result.trace)
    assert result.attempts[1].strategy.genre_targeted


# --- apply_review: the guardrail on the critic ---------------------------------------

def test_invalid_ai_values_are_sanitized(catalog, retriever, vocab):
    ctx = first_attempt_context(catalog, retriever, vocab)
    review = apply_review(ctx, response(
        next_genres=["metal", "vaporwave"], next_moods=["spooky"], next_target_energy=1.7,
        next_target_valence=-0.2, remove_song_ids=[ctx.playlist[0].song.id, 999999], next_n_candidates=10_000,
        next_search_text="   ",
    ))
    s = review.revision.strategy
    assert s.profile.genres == ["metal"]                   # unknown genre dropped
    assert s.profile.moods == ctx.strategy.profile.moods   # all-invalid -> keep current, not "any mood"
    assert s.profile.target_energy == 1.0 and s.profile.target_valence == 0.0
    assert s.banned_song_ids == [ctx.playlist[0].song.id]  # id not in the playlist ignored
    assert s.n_candidates == MAX_CANDIDATES
    assert s.search_text == ctx.strategy.search_text       # empty -> keep current


def test_revision_lists_human_readable_changes(catalog, retriever, vocab):
    ctx = first_attempt_context(catalog, retriever, vocab)
    review = apply_review(ctx, response(next_target_energy=0.6, next_genres=["metal", "punk"]))
    changes = " ".join(review.revision.changes)
    assert "energy 0.90 -> 0.60" in changes and "genres ['metal'] -> ['metal', 'punk']" in changes


def test_reordering_moods_is_not_reported_as_a_change(catalog, retriever, vocab):
    ctx = first_attempt_context(catalog, retriever, vocab)
    review = apply_review(ctx, response(next_moods=["angry", "intense"]))
    assert not any(c.startswith("moods") for c in review.revision.changes)


# --- what Gemini is shown ------------------------------------------------------------

def test_prompt_shows_original_request_playlist_and_checks(catalog, retriever, vocab):
    ctx = first_attempt_context(catalog, retriever, vocab)
    prompt = build_review_prompt(ctx)
    assert 'ORIGINAL REQUEST: "loud angry metal for the gym"' in prompt
    assert f"[{ctx.playlist[0].song.id}] {ctx.playlist[0].song.title}" in prompt
    assert "GUARDRAIL CHECKS" in prompt and "PASS in_catalog" in prompt
    assert "revisions left: 2" in prompt


def test_gemini_reviewer_uses_structured_output_and_wraps_errors(catalog, retriever, vocab):
    ctx = first_attempt_context(catalog, retriever, vocab)

    class FakeClient:
        def __init__(self, parsed=None, error=None):
            self.models, self.calls, self.parsed, self.error = self, [], parsed, error

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if self.error:
                raise self.error
            return SimpleNamespace(parsed=self.parsed, text="")

    ok = FakeClient(parsed=response(approved=True))
    assert GeminiReviewer(client=ok, model="m")(ctx).approved
    config = ok.calls[0]["config"]
    assert config.response_schema is ReviewResponse
    assert "metal" in config.system_instruction  # allowed genres are in the system prompt

    with pytest.raises(CriticError):
        GeminiReviewer(client=FakeClient(error=RuntimeError("503")), model="m")(ctx)
