"""BoT-SORT baseline with explicit GMC and no appearance extractor."""
from pathlib import Path
from traffic_mot.tracking.backend_bridge import UltralyticsTracker


class BoTSORTTracker(UltralyticsTracker):
    def __init__(self, config: Path):
        super().__init__(config)
        if self.args.tracker_type != "botsort": raise ValueError("Expected botsort configuration")
