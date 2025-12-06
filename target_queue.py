import uuid
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Tuple
import numpy as np

# Using an Enum for status makes state transitions explicit and easier to debug
# than string comparisons.
class TargetStatus(Enum):
    PENDING = "pending"
    PICKED_UP = "picked_up"
    DROPPED = "dropped"

@dataclass
class Target:
    """
    Represents a single pick-and-place task.
    """
    position: np.ndarray
    # Dropoff can be a coordinate array or a named location (e.g., 'hamper')
    dropoff: Union[np.ndarray, str]
    source: str  # 'user' or 'ai'
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TargetStatus = TargetStatus.PENDING

    def distance_to(self, other_pos: np.ndarray) -> float:
        return float(np.linalg.norm(self.position - other_pos))

class TargetQueue:
    def __init__(self, duplicate_threshold: float = 0.05):
        """
        :param duplicate_threshold: The distance in meters under which two targets
                                    are considered the same object.
        """
        self._queue: List[Target] = []
        self._duplicate_threshold = duplicate_threshold
        # Since UI, AI, and Robot are independent threads/processes, we need a lock
        # to prevent race conditions (e.g., UI removing a target while Robot is selecting it).
        self._lock = threading.RLock()

    def add_user_target(self, position: Union[Tuple, List, np.ndarray], dropoff: Union[Tuple, List, np.ndarray, str]) -> str:
        """
        Adds a high-priority target from the UI.
        User targets bypass de-duplication checks (user knows best) and
        are inserted at the front of the queue.

        Returns the id of the newly added target.
        """
        # Ensure position is a numpy array
        pos_array = np.array(position, dtype=np.float64)
        
        # Handle dropoff conversion if it's a coordinate
        if not isinstance(dropoff, str):
            dropoff_val = np.array(dropoff, dtype=np.float64)
        else:
            dropoff_val = dropoff

        target = Target(position=pos_array, dropoff=dropoff_val, source='user')
        
        with self._lock:
            self._queue.insert(0, target)
            return target.id

    def add_ai_targets(self, targets_data: List[dict]):
        """
        Batch processes AI suggestions with specific reconciliation logic:
        1. Syncs with existing targets:
           - If match found (dist < threshold):
             - If existing is USER: Keep User data, ignore AI update.
             - If existing is AI: Update position/dropoff to new AI data, keep existing ID.
           - If no match: Add as new AI target.
        2. Prunes stale AI targets:
           - Any existing AI target not matched in this batch is removed.

        All targets known to the model must be submitted at once in a single call.
        Absense of a target is taken as proof it has slipped into the interdimensional space where socks
        go sometimes betweneen the wash and dry cycle.
        """
        with self._lock:
            matched_ids = set()
            new_targets = []

            # 1. Process all incoming AI suggestions
            for data in targets_data:
                pos = np.array(data['position'], dtype=np.float64)
                
                dropoff_raw = data.get('dropoff', 'default_drop')
                if not isinstance(dropoff_raw, str):
                    dropoff_val = np.array(dropoff_raw, dtype=np.float64)
                else:
                    dropoff_val = dropoff_raw

                # Find best matching existing target (closest within threshold)
                best_match = None
                min_dist = self._duplicate_threshold

                for t in self._queue:
                    # Enforce 1-to-1 matching: don't match something already claimed by this batch
                    if t.id in matched_ids:
                        continue
                    
                    dist = t.distance_to(pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = t

                if best_match:
                    # Match found - mark as kept
                    matched_ids.add(best_match.id)
                    
                    # If it was an AI target, update it with the fresher sensor data
                    # If it was a User target, we do nothing (User data is ground truth)
                    if best_match.source == 'ai':
                        best_match.position = pos
                        best_match.dropoff = dropoff_val
                else:
                    # No match found - create new
                    new_target = Target(
                        position=pos,
                        dropoff=dropoff_val,
                        source='ai'
                    )
                    new_targets.append(new_target)

            # 2. Rebuild queue: Keep Users + Matched AI + New AI
            # This logic effectively deletes unmatched (stale) AI targets
            # while preserving the order of existing items.
            self._queue = [
                t for t in self._queue 
                if t.source == 'user' or t.id in matched_ids
            ]
            self._queue.extend(new_targets)

    def remove_target(self, target_id: str) -> bool:
        """
        Removes a target by ID. Returns True if found and removed.
        """
        with self._lock:
            for i, target in enumerate(self._queue):
                if target.id == target_id:
                    del self._queue[i]
                    return True
            return False

    def reorder_target(self, target_id: str, new_index: int):
        """
        Moves a specific target to a new index in the queue.
        """
        with self._lock:
            # Locate the target first
            target_index = next((i for i, t in enumerate(self._queue) if t.id == target_id), None)
            
            if target_index is not None:
                target = self._queue.pop(target_index)
                # Clamp index to valid bounds to prevent errors
                safe_index = max(0, min(new_index, len(self._queue)))
                self._queue.insert(safe_index, target)

    def get_best_target(self) -> Optional[Target]:
        """
        Selects the best target for the robot.
        Proximity logic removed: simply returns the first PENDING target in the queue.
        """
        with self._lock:
            return next((t for t in self._queue if t.status == TargetStatus.PENDING), None)

    def set_target_status(self, target_id: str, status: TargetStatus) -> bool:
        """
        Updates the status of a target. 
        If status is DROPPED, the target is removed from the queue.
        """
        with self._lock:
            if status == TargetStatus.DROPPED:
                return self.remove_target(target_id)
            
            target = self._get_by_id(target_id)
            if target:
                target.status = status
                return True
            return False

    def get_queue_snapshot(self) -> List[dict]:
        """
        Returns the whole queue as a list of dicts for UI visualization.
        """
        snapshot = []
        with self._lock:
            for target in self._queue:
                # Convert numpy arrays to lists for JSON serialization
                pos_list = target.position.tolist()
                dropoff_val = target.dropoff
                if isinstance(dropoff_val, np.ndarray):
                    dropoff_val = dropoff_val.tolist()

                snapshot.append({
                    "id": target.id,
                    "position": pos_list,
                    "dropoff": dropoff_val,
                    "status": target.status.value,
                    "source": target.source
                })
        return snapshot

    def get_target_info(self, target_id: str) -> Optional[dict]:
        """
        Robot may query this to check if a target it was pursuing was deleted from the queue.
        """
        with self._lock:
            target = self._get_by_id(target_id)
            if target:
                return {
                    "id": target.id,
                    "position": target.position,
                    "dropoff": target.dropoff,
                    "status": target.status.value
                }
            return None

    def _is_duplicate(self, pos: np.ndarray) -> bool:
        """
        Internal helper. Checks if a position is effectively identical to any 
        target currently in the queue (Pending or Picked).
        """
        for target in self._queue:
            if target.distance_to(pos) < self._duplicate_threshold:
                return True
        return False

    def _get_by_id(self, target_id: str) -> Optional[Target]:
        return next((t for t in self._queue if t.id == target_id), None)