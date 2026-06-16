from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DISPLAYED_OPTION_RE = re.compile(r'"selected_option_id"\s*:\s*"?(\d+)"?')


def sorted_dimension_codes(weight_map: Dict[str, Any]) -> List[str]:
    return [
        str(dim)
        for dim, _ in sorted(
            ((str(dim), float(val)) for dim, val in weight_map.items()),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def stakeholder_utilities_for_scenario(scenario: Dict[str, Any]) -> Dict[int, Dict[int, float]]:
    out: Dict[int, Dict[int, float]] = {}
    options = {int(opt["option_id"]): dict(opt["dimension_weights"]) for opt in scenario["options"]}
    for stakeholder in scenario["stakeholders"]:
        role_id = int(stakeholder["role_id"])
        pref = {str(k): float(v) for k, v in stakeholder["dimension_weights"].items()}
        util_by_option: Dict[int, float] = {}
        for option_id, opt_weights in options.items():
            util_by_option[int(option_id)] = sum(pref[dim] * float(opt_weights[dim]) for dim in pref)
        out[role_id] = util_by_option
    return out


def ranking_from_scores(scores_by_option: Dict[int, float]) -> List[int]:
    return sorted((int(opt) for opt in scores_by_option.keys()), key=lambda opt: (-float(scores_by_option[opt]), int(opt)))


def borda_winner_set(rankings: Dict[int, Sequence[int]]) -> List[int]:
    any_ranking = next(iter(rankings.values()), None)
    if any_ranking is None:
        raise ValueError("borda_winner_set requires at least one ranking")
    option_ids = [int(x) for x in any_ranking]
    n = len(option_ids)
    scores = {int(opt): 0 for opt in option_ids}
    for ranking in rankings.values():
        if len(ranking) != n:
            raise ValueError("All rankings must have the same length")
        for idx, opt in enumerate(ranking):
            scores[int(opt)] += (n - 1 - idx)
    best = max(scores.values())
    return sorted(int(opt) for opt, val in scores.items() if val == best)


def utilitarian_winner_set(utilities: Dict[int, Dict[int, float]]) -> List[int]:
    option_ids = sorted({int(opt) for per_st in utilities.values() for opt in per_st.keys()})
    totals = {opt: sum(float(per_st[opt]) for per_st in utilities.values()) for opt in option_ids}
    best = max(totals.values())
    return sorted(int(opt) for opt, val in totals.items() if val == best)


def utilitarian_scores(utilities: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    option_ids = sorted({int(opt) for per_st in utilities.values() for opt in per_st.keys()})
    return {opt: sum(float(per_st[opt]) for per_st in utilities.values()) for opt in option_ids}


def maximin_winner_set(utilities: Dict[int, Dict[int, float]]) -> List[int]:
    option_ids = sorted({int(opt) for per_st in utilities.values() for opt in per_st.keys()})
    floors = {opt: min(float(per_st[opt]) for per_st in utilities.values()) for opt in option_ids}
    best = max(floors.values())
    return sorted(int(opt) for opt, val in floors.items() if val == best)


def maximin_scores(utilities: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    option_ids = sorted({int(opt) for per_st in utilities.values() for opt in per_st.keys()})
    return {opt: min(float(per_st[opt]) for per_st in utilities.values()) for opt in option_ids}


def ordinal_alignment_score(priority_order: Sequence[str], delivery_order: Sequence[str]) -> int:
    delivery_pos = {str(dim): idx for idx, dim in enumerate(delivery_order)}
    n = len(priority_order)
    score = 0
    for p_rank, dim in enumerate(priority_order):
        delivery_rank = delivery_pos[str(dim)]
        priority_weight = n - p_rank
        delivery_strength = n - delivery_rank
        score += priority_weight * delivery_strength
    return int(score)


def ordinal_rankings_for_scenario(scenario: Dict[str, Any]) -> Dict[int, List[int]]:
    option_orders = {
        int(opt["option_id"]): list(opt.get("dimension_rank_order") or sorted_dimension_codes(opt["dimension_weights"]))
        for opt in scenario["options"]
    }
    out: Dict[int, List[int]] = {}
    for stakeholder in scenario["stakeholders"]:
        role_id = int(stakeholder["role_id"])
        priorities = list(stakeholder.get("dimension_rank_order") or sorted_dimension_codes(stakeholder["dimension_weights"]))
        scores = {
            opt_id: ordinal_alignment_score(priorities, order)
            for opt_id, order in option_orders.items()
        }
        out[role_id] = ranking_from_scores(scores)
    return out


def maximin_ordinal_winner_set(scenario: Dict[str, Any]) -> List[int]:
    floors = maximin_ordinal_scores(scenario)
    best = max(floors.values())
    return sorted(int(opt) for opt, val in floors.items() if val == best)


def maximin_ordinal_scores(scenario: Dict[str, Any]) -> Dict[int, int]:
    option_orders = {
        int(opt["option_id"]): list(opt.get("dimension_rank_order") or sorted_dimension_codes(opt["dimension_weights"]))
        for opt in scenario["options"]
    }
    option_ids = sorted(option_orders.keys())
    floors: Dict[int, int] = {}
    for opt_id in option_ids:
        per_stakeholder = []
        for stakeholder in scenario["stakeholders"]:
            priorities = list(stakeholder.get("dimension_rank_order") or sorted_dimension_codes(stakeholder["dimension_weights"]))
            per_stakeholder.append(ordinal_alignment_score(priorities, option_orders[opt_id]))
        floors[int(opt_id)] = min(per_stakeholder)
    return floors


def nash_winner_set(utilities: Dict[int, Dict[int, float]], epsilon: float = 1.0e-12) -> List[int]:
    option_ids = sorted({int(opt) for per_st in utilities.values() for opt in per_st.keys()})
    scores: Dict[int, float] = {}
    for opt in option_ids:
        total = 0.0
        for per_st in utilities.values():
            total += math.log(float(per_st[opt]) + float(epsilon))
        scores[int(opt)] = total
    best = max(scores.values())
    return sorted(int(opt) for opt, val in scores.items() if abs(val - best) <= 1.0e-12)


def copeland_winner_set(utilities: Dict[int, Dict[int, float]]) -> List[int]:
    scores = copeland_scores(utilities)
    best = max(scores.values())
    return sorted(int(opt) for opt, val in scores.items() if abs(val - best) <= 1.0e-12)


def copeland_scores(utilities: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    option_ids = sorted({int(opt) for per_st in utilities.values() for opt in per_st.keys()})
    scores = {opt: 0.0 for opt in option_ids}
    for i, a in enumerate(option_ids):
        for b in option_ids[i + 1 :]:
            votes_a = 0.0
            votes_b = 0.0
            for per_st in utilities.values():
                ua = float(per_st[a])
                ub = float(per_st[b])
                if ua > ub:
                    votes_a += 1.0
                elif ub > ua:
                    votes_b += 1.0
                else:
                    votes_a += 0.5
                    votes_b += 0.5
            if votes_a > votes_b:
                scores[a] += 1.0
            elif votes_b > votes_a:
                scores[b] += 1.0
            else:
                scores[a] += 0.5
                scores[b] += 0.5
    return scores


def pareto_optimal_set(utilities: Dict[int, Dict[int, float]]) -> List[int]:
    option_ids = sorted({int(opt) for per_st in utilities.values() for opt in per_st.keys()})
    winners: List[int] = []
    for candidate in option_ids:
        cand_values = [float(utilities[role_id][candidate]) for role_id in sorted(utilities.keys())]
        dominated = False
        for other in option_ids:
            if other == candidate:
                continue
            other_values = [float(utilities[role_id][other]) for role_id in sorted(utilities.keys())]
            if all(o >= c for o, c in zip(other_values, cand_values)) and any(o > c for o, c in zip(other_values, cand_values)):
                dominated = True
                break
        if not dominated:
            winners.append(int(candidate))
    return sorted(winners)


def kendall_tau_distance(order_a: Sequence[int], order_b: Sequence[int]) -> float:
    if len(order_a) != len(order_b):
        raise ValueError("kendall_tau_distance requires same length orders")
    pos_a = {int(opt): idx for idx, opt in enumerate(order_a)}
    pos_b = {int(opt): idx for idx, opt in enumerate(order_b)}
    options = [int(x) for x in order_a]
    discordant = 0
    total_pairs = 0
    n = len(options)
    for i in range(n):
        for j in range(i + 1, n):
            oi = options[i]
            oj = options[j]
            total_pairs += 1
            if (pos_a[oi] - pos_a[oj]) * (pos_b[oi] - pos_b[oj]) < 0:
                discordant += 1
    return float(discordant) / float(total_pairs) if total_pairs else 0.0


def scenario_properties(scenario: Dict[str, Any]) -> Dict[str, Any]:
    utilities = stakeholder_utilities_for_scenario(scenario)
    cardinal_rankings = {role_id: ranking_from_scores(scores) for role_id, scores in utilities.items()}
    ordinal_rankings = ordinal_rankings_for_scenario(scenario)

    borda = borda_winner_set(cardinal_rankings)
    utilitarian_score_map = utilitarian_scores(utilities)
    utilitarian = sorted(int(opt) for opt, val in utilitarian_score_map.items() if val == max(utilitarian_score_map.values()))
    maximin_score_map = maximin_scores(utilities)
    maximin = sorted(int(opt) for opt, val in maximin_score_map.items() if val == max(maximin_score_map.values()))
    maximin_ordinal_score_map = maximin_ordinal_scores(scenario)
    maximin_ord = sorted(int(opt) for opt, val in maximin_ordinal_score_map.items() if val == max(maximin_ordinal_score_map.values()))
    nash = nash_winner_set(utilities)
    copeland_score_map = copeland_scores(utilities)
    copeland = sorted(int(opt) for opt, val in copeland_score_map.items() if abs(val - max(copeland_score_map.values())) <= 1.0e-12)
    pareto = pareto_optimal_set(utilities)

    ranking_list = list(cardinal_rankings.values())
    taus: List[float] = []
    for i, left in enumerate(ranking_list):
        for right in ranking_list[i + 1 :]:
            taus.append(kendall_tau_distance(left, right))
    tau_avg = sum(taus) / len(taus) if taus else 0.0
    top_choices = {int(ranking[0]) for ranking in cardinal_rankings.values()}
    rule_winners = [tuple(borda), tuple(utilitarian), tuple(maximin), tuple(maximin_ord), tuple(nash), tuple(copeland)]
    winner_entropy_counts = Counter(rule_winners)
    entropy = 0.0
    total = float(sum(winner_entropy_counts.values()))
    for count in winner_entropy_counts.values():
        p = float(count) / total
        entropy -= p * math.log(p)

    conflict_score = float(tau_avg)
    conflict_level = "low" if conflict_score < (1.0 / 3.0) else ("medium" if conflict_score < (2.0 / 3.0) else "high")
    return {
        "scenario_id": int(scenario["scenario_id"]),
        "scenario_code": str(scenario.get("scenario_code") or f"scenario_{scenario['scenario_id']}"),
        "borda_winner_id": int(min(borda)),
        "borda_tie_size": len(borda),
        "borda_winner_set": borda,
        "utilitarian_winner_id": int(min(utilitarian)),
        "utilitarian_winner_set": utilitarian,
        "utilitarian_tie_size": len(utilitarian),
        "maximin_winner_id": int(min(maximin)),
        "maximin_winner_set": maximin,
        "maximin_tie_size": len(maximin),
        "maximin_ordinal_winner_id": int(min(maximin_ord)),
        "maximin_ordinal_winner_set": maximin_ord,
        "maximin_ordinal_tie_size": len(maximin_ord),
        "nash_winner_set": nash,
        "copeland_winner_set": copeland,
        "pareto_optimal_set": pareto,
        "pareto_set_size": len(pareto),
        "borda_winner_is_pareto": int(min(borda)) in set(pareto),
        "is_divergent": (set(borda) != set(utilitarian)) or (set(borda) != set(maximin)),
        "conflict_score": conflict_score,
        "conflict_level": conflict_level,
        "conflict_kendall_tau_avg": tau_avg,
        "conflict_top_choice_distinct_count": len(top_choices),
        "conflict_rules5_entropy": entropy,
        "stakeholder_rankings": cardinal_rankings,
        "ordinal_stakeholder_rankings": ordinal_rankings,
        "utilities": utilities,
    }


def canonical_from_displayed(displayed_option_id: int, displayed_order: Sequence[int]) -> int:
    idx = int(displayed_option_id) - 1
    if idx < 0 or idx >= len(displayed_order):
        raise ValueError(f"Displayed option id out of range: {displayed_option_id}")
    return int(displayed_order[idx])


def displayed_from_canonical(canonical_option_id: int, displayed_order: Sequence[int]) -> int:
    try:
        return int(list(displayed_order).index(int(canonical_option_id)) + 1)
    except ValueError as exc:
        raise ValueError(f"Canonical option id not present in displayed order: {canonical_option_id}") from exc


def parse_selected_option_id(response_text: str) -> Tuple[Optional[int], Optional[str]]:
    raw = str(response_text or "").strip()
    if not raw:
        return None, "empty_response"
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "selected_option_id" in parsed:
            return int(parsed["selected_option_id"]), None
    except Exception:
        pass
    match = DISPLAYED_OPTION_RE.search(raw)
    if match:
        return int(match.group(1)), None
    return None, "selected_option_id_not_found"


def agreement_flag(canonical_selected: Optional[int], canonical_winner_set: Sequence[int]) -> Optional[int]:
    if canonical_selected is None:
        return None
    winners = {int(x) for x in canonical_winner_set}
    return 1 if int(canonical_selected) in winners else 0


def veto_violation(
    canonical_selected: Optional[int],
    scenario: Dict[str, Any],
    vetoes_by_scenario: Dict[int, Dict[str, int]],
    *,
    dimension_id_to_code: Optional[Dict[int, str]] = None,
) -> Optional[int]:
    if canonical_selected is None:
        return None
    scenario_veto = vetoes_by_scenario.get(int(scenario["scenario_id"])) or scenario.get("scenario_veto")
    if not scenario_veto:
        return None
    dim_id = int(scenario_veto["dimension_id"])
    option_map = {int(opt["option_id"]): opt for opt in scenario["options"]}
    option = option_map[int(canonical_selected)]
    ordered = list(option.get("dimension_rank_order") or sorted_dimension_codes(option["dimension_weights"]))
    dim_code = None
    if dimension_id_to_code is not None:
        dim_code = dimension_id_to_code.get(int(dim_id))
    if dim_code is None:
        dim_codes = list(option["dimension_weights"].keys())
        dim_code = dim_codes[dim_id - 1] if dim_id - 1 < len(dim_codes) else None
    if dim_code is None:
        return None
    weakest = ordered[-1]
    return 1 if str(weakest) == str(dim_code) else 0
