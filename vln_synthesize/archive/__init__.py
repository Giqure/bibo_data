"""syn_utils — navigation primitives for VLN data synthesis.

Importing this package binds search/sampling extensions to NavGrid
and query extensions to NavMeshWrapper so that downstream code can
use e.g. ``grid.astar_distance(...)``
"""
from syn_utils.nav_grid import NavGrid                       # noqa: F401
from syn_utils import nav_grid_search as _ngs                # noqa: F401  (binds methods)
from syn_utils.goal_region import GoalRegion, CircleRegion, RectRegion  # noqa: F401

# NavMeshWrapper + query extensions require Isaac Sim runtime;
# import lazily so the package is still usable for grid-only work.
try:
    from vln_synthesize.syn_utils.nav_mesh_wrap import NavMeshWrapper            # noqa: F401
    from syn_utils import nav_mesh_query as _nmq             # noqa: F401  (binds methods)
except Exception:
    pass
