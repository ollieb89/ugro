"""HPO Advanced Analysis - Compatibility Alias.

Re-exports from ugro.hpo.analysis for backward compatibility
with documentation examples that reference analysis_advanced.

Example:
    from ugro.hpo.analysis_advanced import analyze_hpo_results
"""

from ugro.hpo.analysis import (
    analyze_hpo_results,
    compare_studies,
    export_best_config,
)

__all__ = [
    "analyze_hpo_results",
    "export_best_config",
    "compare_studies",
]
