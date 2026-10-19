"""Causal Articulation Theory (CAT) — articulation rules + recursion.

Faithful re-implementation of the articulation engine from Zečević, Dhami &
Kersting, "Causal Articulation Theory (CAT): Articulating Static and Temporal
Causal Models Beyond LLM Rationalizations", CLeaR 2026 (PMLR 323), for the
static linear-SCM case (their Sec. 3).

CAT takes a *given* causal structure plus the *sign and magnitude of causal
effects* and produces a natural-language answer to a why-question ("why is V_i
below/above its population average?"). It is NOT an LLM method — LLMs appear in
that paper only as a motivating contrast. This module is deliberately
provider-agnostic: the per-edge causal effect is supplied by a callable
``effect(child, parent) -> float``. That is the hook that lets us drive CAT's
articulation from *FADO's graph-free effect estimates* instead of from a given
SCM's structural coefficients, and compare against oracle-graph estimators
(do-Shapley, Causal Shapley) and the ground-truth SCM.

Rules (Def. 6 + Table 1, read operationally per the paper's Sec. 3 "How to read
the articulation rules"): for a directed edge V_j -> V_i with signed causal
effect alpha, let the observed deviations be d_j = sign(v_j - mu_j) and
d_i = sign(v_i - mu_i).

  * ER1 excitation : sign(alpha) * d_j == d_i
        -> "V_i because V_j being {low|high}"
  * ER2 inhibition : sign(alpha) * d_j == -d_i
        -> "V_i although V_j is {low|high}"
  * ER3 preference : the parent with the largest |alpha| is prefixed "mostly".

The recursion (Def. 7) is depth-first over parents; roots (no parents) are the
base case. A cause already used on the current path is not re-explained
(CAT's "non-repeating causes" property).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class EdgeArticulation:
    parent: str
    rule: str          # "excitation" | "inhibition"
    parent_state: str  # "low" | "high"
    effect: float
    dominant: bool     # ER3: the strongest-|effect| parent


@dataclass
class NodeArticulation:
    node: str
    state: str                     # "low" | "high"
    edges: list[EdgeArticulation] = field(default_factory=list)
    children_expl: dict = field(default_factory=dict)   # parent -> NodeArticulation


def _state(v: float, mu: float, tol: float = 1e-9) -> Optional[str]:
    if v > mu + tol:
        return "high"
    if v < mu - tol:
        return "low"
    return None  # exactly at the mean: no valid why-question direction


def _dir(state: Optional[str]) -> int:
    return {"high": 1, "low": -1}.get(state, 0)


def articulate(
    node: str,
    parents: Callable[[str], list[str]],
    effect: Callable[[str, str], float],
    obs: dict[str, float],
    mu: dict[str, float],
    *,
    _path: Optional[frozenset] = None,
    min_effect: float = 1e-6,
) -> Optional[NodeArticulation]:
    """Articulate why ``node`` is below/above its mean, recursively.

    ``parents(node)`` returns the structural parents of ``node`` (the graph is
    given, exactly as CAT assumes an SCM estimate). ``effect(child, parent)``
    returns the signed causal effect of ``parent`` on ``child`` — this is the
    only place a provider (FADO / oracle / ground-truth) enters.
    """
    _path = _path or frozenset()
    st = _state(obs[node], mu[node])
    if st is None:
        return None
    art = NodeArticulation(node=node, state=st)
    d_i = _dir(st)

    pa = [p for p in parents(node) if p not in _path]
    scored = []
    for p in pa:
        a = float(effect(node, p))
        if abs(a) < min_effect:
            continue
        sp = _state(obs[p], mu[p])
        if sp is None:
            continue
        rule = "excitation" if (int(_sign(a)) * _dir(sp)) == d_i else "inhibition"
        scored.append(EdgeArticulation(parent=p, rule=rule, parent_state=sp,
                                       effect=a, dominant=False))
    # ER3 (the "mostly" modifier) fires only when there is more than one
    # parental effect (the paper's |alpha^pa| > 1 condition); it tags the
    # strongest absolute parental effect.
    if len(scored) > 1:
        k = max(range(len(scored)), key=lambda j: abs(scored[j].effect))
        scored[k].dominant = True
    # Stable, readable order: dominant first, then remaining by |effect| desc.
    scored.sort(key=lambda e: (not e.dominant, -abs(e.effect)))
    art.edges = scored

    for e in scored:
        sub = articulate(e.parent, parents, effect, obs, mu,
                         _path=_path | {node}, min_effect=min_effect)
        if sub is not None:
            art.children_expl[e.parent] = sub
    return art


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


# --------------------------------------------------------------------------
# Rendering + structural fingerprint
# --------------------------------------------------------------------------

def render(art: NodeArticulation, names: Optional[dict[str, str]] = None) -> str:
    """Render a NodeArticulation as a natural-language sentence."""
    nm = (lambda x: names.get(x, x)) if names else (lambda x: x)
    return _render_clause(art, nm, top=True).strip().rstrip(",") + "."


def _render_clause(art: NodeArticulation, nm, *, top: bool) -> str:
    head = f"{nm(art.node)} is {art.state}"
    if not art.edges:
        return head
    exc = [e for e in art.edges if e.rule == "excitation"]
    inh = [e for e in art.edges if e.rule == "inhibition"]
    parts = [head]
    # "because of <excitation parents>", dominant tagged "mostly"
    if exc:
        segs = []
        for e in exc:
            pref = "mostly " if e.dominant else ""
            sub = art.children_expl.get(e.parent)
            if sub is not None and sub.edges:
                segs.append(f"{pref}because {nm(e.parent)} is {e.parent_state} "
                            f"({_render_clause(sub, nm, top=False)})")
            else:
                segs.append(f"{pref}because {nm(e.parent)} is {e.parent_state}")
        parts.append(" and ".join(segs))
    # "although <inhibition parents>"
    for e in inh:
        pref = "mostly " if e.dominant else ""
        parts.append(f"{pref}although {nm(e.parent)} is {e.parent_state}")
    return ", ".join(parts)


def fingerprint(art: NodeArticulation) -> dict:
    """Structural summary used for provider-vs-provider agreement.

    Captures what CAT's *rules* decided, independent of surface wording:
    per-parent rule (excitation/inhibition) and which parent is dominant.
    Recurses so multi-hop structure is included.
    """
    fp = {
        "node": art.node,
        "state": art.state,
        "edges": {e.parent: {"rule": e.rule, "dominant": e.dominant}
                  for e in art.edges},
    }
    fp["children"] = {p: fingerprint(sub) for p, sub in art.children_expl.items()}
    return fp
