"""
The music recommendation agent, as a LangGraph state machine.

    analyze -> plan -> retrieve -> generate -> evaluate --accepted--> finalize -> explain -> END
                          ^                       |
                          +------- revise <-------+  rejected (at most MAX_REVISIONS times)

  analyze   Gemini turns the request into a ParsedRequest            (preferences.py)
  plan      initial search text, pool size, artist rules             (agent/planner.py)
  retrieve  ChromaDB semantic search -> candidate songs              (rag/retrieval.py)
  generate  original scorer ranks; playlist rules applied            (agent/generator.py)
  evaluate  1. guardrail rules check the playlist                    (agent/evaluator.py)
            2. Gemini critic reviews it and, if it rejects, writes   (agent/critic.py)
               the next plan
               - rules passed -> "critique": AI approves or rejects
               - rules failed -> "revise":   AI diagnoses and replans
  revise    apply the AI's plan (rule-based fallback only if Gemini is unreachable)
  finalize  keep the best attempt, run guardrails, list limitations
  explain   explain each song from code-computed evidence (no LLM)    (agent/explainer.py)

An attempt is accepted only when the rules pass AND the AI approves (if Gemini is
unreachable, passing the rules is enough). Every node appends to `trace`, so the
UI can show exactly what the agent did and why.
"""

import logging
from typing import Callable, Iterator, List, Optional, Tuple

from langgraph.graph import END, START, StateGraph

from preferences import ParsedRequest, build_vocabulary, extract_preferences
from recommender import Song
from guardrails.validation import duplicate_songs, invalid_songs
from rag.retrieval import MusicRetriever, is_excluded
from dataclasses import replace

from agent.critic import GeminiReviewer, ReviewContext, run_review
from agent.evaluator import evaluate, revise as rule_revise
from agent.explainer import explain_playlist
from agent.generator import build_playlist
from agent.planner import make_plan
from agent.state import AgentResult, AgentState, Attempt, Revision

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2
PER_GENRE_RESULTS = 12  # = songs per genre in the catalog, so a genre search can return all of them


def _attempt_rank(attempt: Attempt):
    """Best attempt: accepted (rules + AI) first, then rules passed, fewest failed checks, highest avg score."""
    scores = [item.score for item in attempt.playlist]
    return (attempt.accepted, attempt.evaluation.passed, -len(attempt.evaluation.failed),
            sum(scores) / len(scores) if scores else 0.0)


def _review_trace(review) -> List[dict]:
    if review.error:
        return [{"step": "ai_review", "detail": f"AI critic unavailable ({review.error[:120]}); "
                                                "falling back to rule checks"}]
    if review.mode == "critique":
        verdict = "APPROVED" if review.approved else "REJECTED"
    else:
        verdict = "DIAGNOSED rule failures"
    trace = [{"step": "ai_review", "detail": f"{verdict}: {review.reasoning}"}]
    trace += [{"step": "ai_review", "detail": f"issue: {issue}"} for issue in review.issues]
    return trace


def build_agent(catalog: List[Song], retriever: MusicRetriever, extractor: Optional[Callable] = None,
                reviewer: Optional[Callable] = None):
    """Compile the graph.

    extractor(query, catalog)  -> ParsedRequest        (default: Gemini)
    reviewer(ReviewContext)    -> ReviewResponse       (None = no AI critic, rules only)
    All are injectable so tests run without Gemini.
    """
    extractor = extractor or extract_preferences
    vocab = build_vocabulary(catalog)

    def analyze(state: AgentState) -> dict:
        if state.get("request") is not None:  # already parsed (tests, or a UI that edits the profile)
            return {"trace": [{"step": "analyze", "detail": "using provided preferences"}]}
        request = extractor(state["query"], catalog)
        p = request.profile
        detail = (f"genres={p.genres or 'any'}, moods={p.moods or 'any'}, energy={p.target_energy:.2f}, "
                  f"valence={p.target_valence:.2f}, acousticness={p.target_acousticness:.2f}, "
                  f"songs={request.num_songs}")
        return {"request": request, "trace": [{"step": "analyze", "detail": detail}]}

    def plan(state: AgentState) -> dict:
        strategy, notes = make_plan(state["request"], catalog, state.get("query", ""))
        return {"strategy": strategy, "notes": notes, "revisions": 0,
                "trace": [{"step": "plan", "detail": step} for step in strategy.steps]}

    def retrieve(state: AgentState) -> dict:
        strategy = state["strategy"]
        results = retriever.search(strategy.search_text, n_results=strategy.n_candidates,
                                   exclude_artists=strategy.exclude_artists)
        detail = f"{len(results)} candidates from semantic search"
        if strategy.genre_targeted and strategy.profile.genres:
            extra = 0
            for genre in strategy.profile.genres:
                genre_results = retriever.search(strategy.search_text, n_results=PER_GENRE_RESULTS,
                                                 exclude_artists=strategy.exclude_artists, genres=[genre])
                extra += len(genre_results)
                results += genre_results
            detail += f" + {extra} from genre-filtered searches"
        # Merge, keeping each song once with its best similarity.
        merged = {}
        for r in results:
            if r.song.id not in merged or r.similarity > merged[r.song.id].similarity:
                merged[r.song.id] = r
        retrieved = sorted(merged.values(), key=lambda r: r.similarity, reverse=True)
        return {"retrieved": retrieved, "trace": [{"step": "retrieve", "detail": f"{detail} ({len(retrieved)} unique)"}]}

    def generate(state: AgentState) -> dict:
        retrieved = state["retrieved"]
        playlist = build_playlist([r.song for r in retrieved], state["request"], state["strategy"],
                                  similarity={r.song.id: r.similarity for r in retrieved})
        return {"playlist": playlist,
                "trace": [{"step": "generate", "detail": f"ranked {len(retrieved)} candidates, picked {len(playlist)}"}]}

    def evaluate_node(state: AgentState) -> dict:
        evaluation = evaluate(state["playlist"], state["request"], state["strategy"], catalog)
        trace = [{"step": "evaluate", "detail": f"{'PASS' if c.passed else 'FAIL'} {c.name}: {c.detail}"}
                 for c in evaluation.checks]

        revisions_left = MAX_REVISIONS - state["revisions"]
        review = None
        # Critique when the rules pass; when they fail, ask the AI to replan only if a retry is still possible.
        if reviewer is not None and (evaluation.passed or revisions_left > 0):
            ctx = ReviewContext(mode="critique" if evaluation.passed else "revise", query=state.get("query", ""),
                                request=state["request"], strategy=state["strategy"], playlist=state["playlist"],
                                evaluation=evaluation, attempt_number=len(state.get("attempts", [])) + 1,
                                revisions_left=revisions_left, vocab=vocab)
            review = run_review(reviewer, ctx)
            trace += _review_trace(review)

        attempt = Attempt(number=len(state.get("attempts", [])) + 1, strategy=state["strategy"],
                          playlist=state["playlist"], evaluation=evaluation, review=review)
        return {"evaluation": evaluation, "attempts": [attempt], "trace": trace}

    def route_after_evaluate(state: AgentState) -> str:
        if state["attempts"][-1].accepted or state["revisions"] >= MAX_REVISIONS:
            return "finalize"
        return "revise"

    def revise_node(state: AgentState) -> dict:
        attempt = state["attempts"][-1]
        review = attempt.review
        n = state["revisions"] + 1

        if review is not None and review.revision is not None:
            revision = review.revision  # the AI's plan
            if not revision.changes:
                return {"gave_up": True,
                        "trace": [{"step": "revise", "detail": "stop: the AI critic rejected but proposed no changes"}]}
        else:
            # Gemini unreachable or disabled: rule-based fallback.
            new_strategy, reason = rule_revise(state["strategy"], attempt.evaluation, state["request"])
            if new_strategy is None:
                return {"gave_up": True, "trace": [{"step": "revise", "detail": f"stop: {reason}"}]}
            revision = Revision(strategy=new_strategy, changes=[reason], source="rules")

        who = "AI" if revision.source == "ai" else "rule fallback"
        trace = [{"step": "revise", "detail": f"revision {n} ({who}): {change}"} for change in revision.changes]
        return {"strategy": revision.strategy, "revisions": n, "trace": trace}

    def route_after_revise(state: AgentState) -> str:
        return "finalize" if state.get("gave_up") else "retrieve"

    def finalize(state: AgentState) -> dict:
        attempts = state["attempts"]
        best = max(attempts, key=_attempt_rank)

        # Final guardrail: strip anything that should never reach the user.
        bad_ids = {s.id for s in invalid_songs([i.song for i in best.playlist], catalog)}
        bad_ids |= {s.id for s in duplicate_songs([i.song for i in best.playlist])}
        playlist = [i for i in best.playlist
                    if i.song.id not in bad_ids and not is_excluded(i.song, best.strategy.exclude_artists)]
        removed = len(best.playlist) - len(playlist)

        notes = list(state.get("notes", []))
        if removed:
            notes.append(f"Guardrail removed {removed} song(s) that failed validation.")
        for check in best.evaluation.failed:
            notes.append(f"Still short after {len(attempts) - 1} revision(s): {check.name} ({check.detail}).")
        if best.review and not best.review.approved and not best.review.error and best.review.issues:
            notes.append("The AI critic still had concerns: " + " | ".join(best.review.issues))
        if any(a.review and a.review.error for a in attempts):
            notes.append("The AI critic was unavailable for at least one attempt; rule checks were used instead.")

        verdict = ("AI-approved" if best.review and best.review.approved
                   else "rules passed" if best.evaluation.passed else "best effort")
        detail = (f"returning attempt {best.number} of {len(attempts)} ({verdict}; "
                  f"{len(best.evaluation.checks) - len(best.evaluation.failed)}/{len(best.evaluation.checks)} "
                  f"rule checks passed)")
        result = AgentResult(request=state["request"], playlist=playlist, evaluation=best.evaluation,
                             chosen_attempt=best.number,
                             attempts=attempts, notes=notes,
                             trace=state["trace"] + [{"step": "finalize", "detail": detail}])
        return {"result": result, "final_strategy": best.strategy,
                "trace": [{"step": "finalize", "detail": detail}]}

    def explain(state: AgentState) -> dict:
        result = state["result"]
        playlist, summary = explain_playlist(state["request"], state["final_strategy"], result.playlist)
        trace = [{"step": "explain", "detail": f"explained {len(playlist)} songs from their evidence"}]
        result = replace(result, playlist=playlist, summary=summary, trace=result.trace + trace)
        return {"result": result, "trace": trace}

    graph = StateGraph(AgentState)
    graph.add_node("analyze", analyze)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize)
    graph.add_node("explain", explain)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate, {"revise": "revise", "finalize": "finalize"})
    graph.add_conditional_edges("revise", route_after_revise, {"retrieve": "retrieve", "finalize": "finalize"})
    graph.add_edge("finalize", "explain")
    graph.add_edge("explain", END)
    return graph.compile()


def iter_agent(query: str, catalog: List[Song], retriever: MusicRetriever,
               extractor: Optional[Callable] = None, request: Optional[ParsedRequest] = None,
               reviewer: Optional[Callable] = None, ai_review: bool = True) -> Iterator[Tuple[str, object]]:
    """Run the workflow step by step, for live progress in a UI.

    Yields ("step", trace_entries) after each node finishes, then ("result", AgentResult) last.
    ai_review=True with no `reviewer` uses the Gemini critic; False runs rule checks only.
    """
    if ai_review and reviewer is None:
        reviewer = GeminiReviewer()
    agent = build_agent(catalog, retriever, extractor, reviewer if ai_review else None)
    result = None
    for update in agent.stream({"query": query, "request": request, "attempts": [], "trace": []},
                               stream_mode="updates"):
        for node, changes in update.items():
            changes = changes or {}
            entries = changes.get("trace", [])
            for entry in entries:
                logger.info("[%s] %s", entry["step"], entry["detail"])
            yield "step", entries
            if node == "explain":
                result = changes["result"]
    yield "result", result


def run_agent(query: str, catalog: List[Song], retriever: MusicRetriever,
              extractor: Optional[Callable] = None, request: Optional[ParsedRequest] = None,
              reviewer: Optional[Callable] = None, ai_review: bool = True) -> AgentResult:
    """Run the whole workflow and return the final result (with its full trace)."""
    result = None
    for kind, payload in iter_agent(query, catalog, retriever, extractor, request, reviewer, ai_review):
        if kind == "result":
            result = payload
    return result
