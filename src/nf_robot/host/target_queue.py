import uuid
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Tuple
import numpy as np

from nf_robot.generated.nf import telemetry, common

@dataclass
class Target:
    """
    Represents a single target. idendified by AI or user, with a given drop point.
    """
    position: np.ndarray
    # Dropoff can be a coordinate array or a named location (e.g., 'hamper')
    dropoff: Union[np.ndarray, str]
    source: str  # 'user' or 'ai'
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: telemetry.TargetStatus = telemetry.TargetStatus.SEEN

    def distance_to(self, other_pos: np.ndarray) -> float:
        return float(np.linalg.norm(self.position - other_pos))

    def as_proto(self) -> telemetry.OneTarget:
        tp = telemetry.OneTarget(
            id = self.id,
            position = common.Vec3(*self.position),
            status = self.status,
            source = self.source,
        )
        if isinstance(self.dropoff, np.ndarray):
            tp.coords = common.Vec3(*self.dropoff)
        else:
            tp.tag = self.dropoff
        return tp

class TargetQueue:
    def __init__(self, duplicate_threshold: float = 0.1):
        """
        :param duplicate_threshold: The distance in meters under which two targets
                                    are considered the same object.
        """
        self._queue: List[Target] = []
        self._false_targets: List[Target] = []
        self._duplicate_threshold = duplicate_threshold
        # to prevent race conditions (e.g., UI removing a target while Robot is selecting it).
        self._lock = threading.RLock()

    def add_user_target(self, position: Union[Tuple, List, np.ndarray], dropoff: Union[Tuple, List, np.ndarray, str]) -> str:
        """
        Adds a high-priority target from the UI.
        User targets bypass de-duplication checks (user knows best) and
        are inserted at the front of the queue.

        Returns the id of the newly added target.
        """
        # Ensure position is a numpy array representing a 3d coordinate
        pos_array = np.array(position, dtype=np.float64)
        if len(pos_array) == 2:
            pos_array = np.pad(pos_array, (0, 1))
        
        # Handle dropoff conversion if it's a coordinate
        if not isinstance(dropoff, str):
            dropoff_val = np.array(dropoff, dtype=np.float64)
        else:
            dropoff_val = dropoff

        target = Target(position=pos_array, dropoff=dropoff_val, source='user')
        
        with self._lock:
            self._queue.insert(0, target)
            return target.id

    def _deduplicate_batch(self, targets_data: List[dict]) -> List[dict]:
        """
        Merges pairs within the batch that are closer than duplicate_threshold.
        Greedy approach: Only one pair is merged per pass to avoid collapsing chains.
        """
        merged_targets = []
        skip_indices = set()
        n = len(targets_data)
        
        # Helper to safely extract numpy position for distance calc
        def _extract_pos(d):
            p = d['position']
            return np.array(p, dtype=np.float64)

        for i in range(n):
            if i in skip_indices:
                continue
            
            pos_i = _extract_pos(targets_data[i])
            
            for j in range(i + 1, n):
                if j in skip_indices:
                    continue
                
                pos_j = _extract_pos(targets_data[j])
                dist = float(np.linalg.norm(pos_i - pos_j))
                
                if dist < self._duplicate_threshold:
                    # Found a duplicate in the batch.
                    # We keep 'i' and drop 'j'.
                    skip_indices.add(j)
                    # We stop looking for matches for 'i' to satisfy "only pairs can combine in one pass"
                    break
            
            merged_targets.append(targets_data[i])
            
        return merged_targets

    def add_ai_targets(self, targets_data: List[dict]):
        """
        Batch processes AI suggestions with specific reconciliation logic:
        0. Deduplicate batch
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
            targets_to_process = self._deduplicate_batch(targets_data)
            matched_ids = set()
            new_targets = []

            # Process all incoming AI suggestions
            for data in targets_to_process:
                pos = np.array(data['position'], dtype=np.float64)

                dropoff_raw = data.get('dropoff', 'default_drop')
                if not isinstance(dropoff_raw, str):
                    dropoff_val = np.array(dropoff_raw, dtype=np.float64)
                else:
                    dropoff_val = dropoff_raw

                # If this looks like something the user previously deleted, skip it
                best_false_match = None
                smalled_observed_false_dist = self._duplicate_threshold
                for ft in self._false_targets:
                    dist = ft.distance_to(pos)
                    if dist < smalled_observed_false_dist:
                        smalled_observed_false_dist = dist
                        best_false_match = ft
                if best_false_match:
                    continue # Discard this target

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

            # Rebuild queue: Keep Users + Matched AI + New AI
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
                    # we don't want it in the work queue anymore, but we don't want to forget about it.
                    # wouldn't be any good if it just came back on its own after you delete it.
                    # move it to another list so it can still sync with AI targets
                    self._false_targets.append(self._queue[i])
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
            return next((
                t for t in self._queue
                if (t.status == telemetry.TargetStatus.SELECTED or t.status == telemetry.TargetStatus.SEEN)
                ), None)

    def set_target_status(self, target_id: str, status: telemetry.TargetStatus) -> bool:
        """
        Updates the status of a target. 
        If status is DROPPED, the target is removed from the queue.
        """
        with self._lock:
            if status == telemetry.TargetStatus.DROPPED:
                return self.remove_target(target_id)
            
            target = self._get_by_id(target_id)
            if target:
                target.status = status
                return True
            return False

    def set_target_position(self, target_id: str, pos2d: np.ndarray):
        """
        Updates the position of a target.
        """
        with self._lock:
            target = self._get_by_id(target_id)
            if target:
                target.position = np.pad(pos2d, (0, 1))
                target.source = 'user'

    def get_queue_snapshot(self) -> telemetry.TargetList:
        """
        Returns the whole queue as a telemetry.TargetList for UI visualization.
        """
        targets = []
        with self._lock:
            for target in self._queue:
                targets.append(target.as_proto())
        return telemetry.TargetList(targets=targets)

    def get_targets_as_array(self) -> np.ndarray:
        """
        Returns all targets as an array of 3D coordinates.
        """
        targets = []
        with self._lock:
            for target in self._queue:
                targets.append(target.position)
        return np.array(targets)


    def get_target_info(self, target_id: str) -> Optional[telemetry.OneTarget]:
        """
        Robot may query this to check if a target it was pursuing was deleted from the queue.
        """
        with self._lock:
            target = self._get_by_id(target_id)
            if target:
                return target.as_proto()
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