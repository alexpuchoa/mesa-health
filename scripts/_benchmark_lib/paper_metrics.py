from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

BASELINE_VARIANT_ID = 0
VETO_VARIANT_ID = 1
FORMAT_VARIANT_IDS = [2, 5]
MAXIMIN_VARIANT_ID = 3
UTILITARIAN_VARIANT_ID = 4
PIR_VARIANT_IDS = [7, 17, 23, 33]
MSI_VARIANT_IDS = [8, 9]
SLIR_VARIANT_IDS = [10, 24, 31]
NAMED_PERM_VARIANT_IDS = [11, 12, 13]
ABSTRACT_BASELINE_VARIANT_ID = 44
ABSTRACT_PERM_VARIANT_IDS = [111, 112, 113]

METRIC_REQUIRED_VARIANTS: Dict[str, Sequence[int]] = {
    "AGR_Borda": [BASELINE_VARIANT_ID],
    "AGR_Maximin": [MAXIMIN_VARIANT_ID],
    "AGR_Utilitarian": [UTILITARIAN_VARIANT_ID],
    "DPC": [ABSTRACT_BASELINE_VARIANT_ID],
    "APDR": [*NAMED_PERM_VARIANT_IDS, *ABSTRACT_PERM_VARIANT_IDS],
    "CPFR": [BASELINE_VARIANT_ID, *NAMED_PERM_VARIANT_IDS],
    "PIR": [BASELINE_VARIANT_ID, *PIR_VARIANT_IDS],
    "FIR": [BASELINE_VARIANT_ID, *FORMAT_VARIANT_IDS],
    "SLIR": [BASELINE_VARIANT_ID, *SLIR_VARIANT_IDS],
    "VVR": [VETO_VARIANT_ID],
    "MSI": [BASELINE_VARIANT_ID, *MSI_VARIANT_IDS],
}


def missing_required_variants(present_variant_ids: Iterable[int], metric_name: str) -> List[int]:
    present = {int(x) for x in present_variant_ids}
    required = METRIC_REQUIRED_VARIANTS[metric_name]
    return [int(variant_id) for variant_id in required if int(variant_id) not in present]


def metric_warning_lines(*, subject_label: str, present_variant_ids: Iterable[int]) -> List[str]:
    lines: List[str] = []
    present: Set[int] = {int(x) for x in present_variant_ids}
    for metric_name in (
        "AGR_Borda",
        "AGR_Maximin",
        "AGR_Utilitarian",
        "DPC",
        "APDR",
        "CPFR",
        "PIR",
        "FIR",
        "SLIR",
        "VVR",
        "MSI",
    ):
        missing = missing_required_variants(present, metric_name)
        if missing:
            missing_text = ",".join(str(x) for x in missing)
            lines.append(
                f"warning: {subject_label} missing required test variant(s) for {metric_name}: "
                f"{missing_text}; {metric_name} will be blank"
            )
    return lines
