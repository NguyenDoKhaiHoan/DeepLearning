"""ByteTrack baseline adapter (stock pinned Ultralytics implementation)."""
from pathlib import Path
from traffic_mot.tracking.backend_bridge import UltralyticsTracker


class ByteTrackTracker(UltralyticsTracker):
    def __init__(self, config: Path):
        super().__init__(config)
        if self.args.tracker_type != "bytetrack": raise ValueError("Expected bytetrack configuration")
