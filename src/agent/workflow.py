"""
The music recommendation agent, as a LangGraph state machine.

    analyze -> plan -> retrieve -> generate -> evaluate --pass--> finalize -> END
                          ^                       |
                          +------- revise <--fail-+   (at most MAX_REVISIONS times)

  analyze   Gemini turns the request into a ParsedRequest   (the only LLM call)
  plan      choose search text, pool size, artist rules     (agent/planner.py)
  retrieve  ChromaDB semantic search -> candidate songs     (rag/retrieval.py)
  generate  original scorer ranks; playlist rules applied   (agent/generator.py)
  evaluate  critic checks the playlist against the request  (agent/evaluator.py)
  revise    pick a fix for what failed, then retrieve again
  finalize  keep the best attempt, run guardrails, list limitations

Every node appends to `trace`, so the UI can show exactly what the agent did.
"""

import logging
from typing import Callable, List, Optional

from langgraph.graph import END, START, StateGraph

from preferences import ParsedRequest, extract_preferences
from recommender import Song
from guardrails.validation import duplicate_songs, invalid_songs
from rag.retrieval import MusicRetriever, is_excluded
from agent.evaluator import evaluate, revise
from agent.generator import build_playlist
from agent.planner import make_plan
from agent.state import AgentResult, AgentState, Attempt

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2
PER_GENRE_RESULTS = 12  # = songs per genre in the catalog, so a genre search can return all of them


def _attempt_rank(attempt: Attempt):
    """Best attempt = fewest failed checks, then highest average score."""
    scores = [item.score for item in attempt.playlist]
    return (-len(attempt.evaluation.failed), sum(scores) / len(scores) if scores else 0.0)


def build_agent(catalog: List[Song], retriever: MusicRetriever, extractor: Optional[Callable] = None):
    """Compile the graph. `extractor(query, catalog) -> ParsedRequest` is injectable for tests."""
    extractor = extractor or extract_preferences

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
        strategy, request = state["strategy"], state["request"]
        results = retriever.search(strategy.search_text, n_results=strategy.n_candidates,
                                   exclude_artists=strategy.exclude_artists)
        detail = f"{len(results)} candidates from semantic search"
        if strategy.genre_targeted:
            extra = 0
            for genre in request.profile.genres:
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
        attempt = Attempt(number=len(state.get("attempts", [])) + 1, strategy=state["strategy"],
                          playlist=state["playlist"], evaluation=evaluation)
        trace = [{"step": "evaluate", "detail": f"{'PASS' if c.passed else 'FAIL'} {c.name}: {c.detail}"}
                 for c in evaluation.checks]
        return {"evaluation": evaluation, "attempts": [attempt], "trace": trace}

    def route_after_evaluate(state: AgentState) -> str:
        if state["evaluation"].passed or state["revisions"] >= MAX_REVISIONS:
            return "finalize"
        return "revise"

    def revise_node(state: AgentState) -> dict:
        new_strategy, reason = revise(state["strategy"], state["evaluation"], state["request"])
        if new_strategy is None:
            # Nothing would help; stop (finalize reports the remaining issues).
            return {"gave_up": True, "trace": [{"step": "revise", "detail": f"stop: {reason}"}]}
        return {"strategy": new_strategy, "revisions": state["revisions"] + 1,
                "trace": [{"step": "revise", "detail": f"revision {state['revisions'] + 1}: {reason}"}]}

    def route_after_revise(state: AgentState) -> str:
        return "finalize" if state.get("gave_up") else "retrieve"

    def finalize(state: AgentState) -> dict:
        attempts = state["attempts"]
        best = max(attempts, key=_attempt_rank)
        request = state["request"]

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

        detail = (f"returning attempt {best.number} of {len(attempts)} "
                  f"({len(best.evaluation.checks) - len(best.evaluation.failed)}/{len(best.evaluation.checks)} checks passed)")
        result = AgentResult(request=request, playlist=playlist, evaluation=best.evaluation,
                             attempts=attempts, notes=notes,
                             trace=state["trace"] + [{"step": "finalize", "detail": detail}])
        return {"result": result, "trace": [{"step": "finalize", "detail": detail}]}

    graph = StateGraph(AgentState)
    graph.add_node("analyze", analyze)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", route_after_evaluate, {"revise": "revise", "finalize": "finalize"})
    graph.add_conditional_edges("revise", route_after_revise, {"retrieve": "retrieve", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent(query: str, catalog: List[Song], retriever: MusicRetriever,
              extractor: Optional[Callable] = None, request: Optional[ParsedRequest] = None) -> AgentResult:
    """Run the whole workflow and return the final result (with its full trace)."""
    agent = build_agent(catalog, retriever, extractor)
    state = agent.invoke({"query": query, "request": request, "attempts": [], "trace": []})
    result = state["result"]
    for entry in result.trace:
        logger.info("[%s] %s", entry["step"], entry["detail"])
    return result
