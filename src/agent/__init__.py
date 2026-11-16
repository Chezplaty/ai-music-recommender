"""
Agentic playlist generation (LangGraph).

  state.py      - data passed between steps (Strategy, PlaylistItem, Evaluation, AgentState)
  planner.py    - request -> Strategy (how to search, how many candidates, artist rules)
  generator.py  - candidates -> playlist (existing scorer ranks; diversity/exclusion rules applied)
  evaluator.py  - the critic: checks the playlist against the request; proposes a revision
  workflow.py   - the LangGraph graph: analyze -> plan -> retrieve -> generate -> evaluate -> (revise | finalize)
"""
