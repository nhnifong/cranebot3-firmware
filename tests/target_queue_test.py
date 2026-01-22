import unittest
import numpy as np

from nf_robot.host.target_queue import TargetQueue
from nf_robot.generated.nf.telemetry import TargetStatus
from nf_robot.generated.nf.common import Vec3
from nf_robot.common.util import *

class TestTargetQueue(unittest.TestCase):
    def setUp(self):
        # Initialize a fresh queue before every test to ensure isolation
        self.queue = TargetQueue(duplicate_threshold=0.5)

    def test_add_user_target_priority(self):
        # User targets should always be inserted at the front of the queue
        id1 = self.queue.add_user_target([10, 0, 0], "bin_a")
        id2 = self.queue.add_user_target([20, 0, 0], "bin_b")

        snapshot = self.queue.get_queue_snapshot().targets
        
        # Expecting LIFO behavior for user additions based on "insert(0, ...)" logic
        self.assertEqual(snapshot[0].id, id2)
        self.assertEqual(snapshot[1].id, id1)
        self.assertEqual(snapshot[0].source, 'user')

    def test_ai_target_addition_and_pruning(self):
        # Add an initial batch of AI targets
        batch_1 = [
            {'position': [10, 10, 0], 'dropoff': 'bin_a'},
            {'position': [20, 20, 0], 'dropoff': 'bin_b'}
        ]
        self.queue.add_ai_targets(batch_1)
        
        snapshot_1 = self.queue.get_queue_snapshot().targets
        self.assertEqual(len(snapshot_1), 2)
        ids_1 = {t.id for t in snapshot_1}

        # Add a second batch that is completely different
        # The previous targets should be pruned because they are not in this new batch
        batch_2 = [
            {'position': [30, 30, 0], 'dropoff': 'bin_c'}
        ]
        self.queue.add_ai_targets(batch_2)
        
        snapshot_2 = self.queue.get_queue_snapshot().targets
        self.assertEqual(len(snapshot_2), 1)
        
        # Verify that the old IDs are gone and we have a new ID
        self.assertNotIn(snapshot_2[0].id, ids_1)
        np.testing.assert_array_equal(tonp(snapshot_2[0].position), [30, 30, 0])

    def test_ai_reconciliation_update_existing(self):
        # Test that if AI submits a target close to an existing AI target, we update it rather than replace it
        
        # Initial AI target
        self.queue.add_ai_targets([
            {'position': [10.0, 10.0, 0.0], 'dropoff': 'old_drop'}
        ])
        original_id = self.queue.get_queue_snapshot().targets[0].id

        # New batch with a target very close (within threshold 0.5)
        self.queue.add_ai_targets([
            {'position': [10.1, 10.0, 0.0], 'dropoff': 'new_drop'}
        ])
        
        snapshot = self.queue.get_queue_snapshot().targets
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0].id, original_id, "ID should be preserved for updates")
        self.assertEqual(snapshot[0].tag, 'new_drop', "Dropoff should be updated")
        # Position should reflect the fresh sensor data
        np.testing.assert_array_almost_equal(tonp(snapshot[0].position), [10.1, 10.0, 0.0])

    def test_ai_cannot_override_user_target(self):
        # If a User target exists, AI seeing the same object should not modify it or remove it
        
        user_pos = [5.0, 5.0, 0.0]
        user_id = self.queue.add_user_target(user_pos, "user_drop")

        # AI submits a target very close to the user target
        ai_pos = [5.1, 5.0, 0.0] # Within 0.5 threshold
        self.queue.add_ai_targets([
            {'position': ai_pos, 'dropoff': 'ai_drop'}
        ])

        snapshot = self.queue.get_queue_snapshot().targets
        
        self.assertEqual(len(snapshot), 1, "Should reconcile to a single target")
        self.assertEqual(snapshot[0].id, user_id, "User ID must be preserved")
        self.assertEqual(snapshot[0].source, 'user', "Source must remain 'user'")
        self.assertEqual(snapshot[0].tag, 'user_drop', "User dropoff preference must be preserved")
        
        # Crucially, the position should NOT update to the AI's noisy reading
        np.testing.assert_array_equal(tonp(snapshot[0].position), user_pos)

    def test_user_targets_survive_ai_pruning(self):
        # AI batches prune unmatched AI targets, but they should never touch User targets
        
        self.queue.add_user_target([100, 100, 0], "safe_zone")
        
        # Empty AI batch implies "I see nothing". 
        # This should clear any AI targets (none here) but leave User targets alone.
        self.queue.add_ai_targets([])
        
        snapshot = self.queue.get_queue_snapshot().targets
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0].source, 'user')

    def test_batch_deduplication(self):
        # Threshold is 0.5
        # Scenario: 
        # A and B are duplicates (dist 0.04). 
        # C is distinct (dist > 0.5 from others).
        
        batch = [
            {'position': [10.0, 0.0, 0.0], 'dropoff': 'drop'},   # A
            {'position': [10.04, 0.0, 0.0], 'dropoff': 'drop'},  # B (dist 0.04 < 0.5)
            {'position': [20.0, 0.0, 0.0], 'dropoff': 'drop'}    # C
        ]
        
        # Call the private method directly
        deduped_batch = self.queue._deduplicate_batch(batch)
        
        # Should result in 2 targets (A/B merged, C separate)
        self.assertEqual(len(deduped_batch), 2)
        
        # Extract positions as numpy arrays for comparison
        positions = [np.array(t['position'], dtype=np.float64) for t in deduped_batch]
        
        # Check for C
        has_c = any(np.allclose(p, [20.0, 0.0, 0.0]) for p in positions)
        self.assertTrue(has_c, "Target C should be present")
        
        # Check for A (or B merged into A)
        # Since logic keeps first, we expect ~10.0
        has_a = any(np.allclose(p, [10.0, 0.0, 0.0]) for p in positions)
        self.assertTrue(has_a, "Target A should be present (absorbing B)")

    def test_get_best_target_logic(self):
        # Should return first PENDING target
        id1 = self.queue.add_user_target([1, 1, 1], "drop") # Index 1
        id2 = self.queue.add_user_target([2, 2, 2], "drop") # Index 0 (User adds to front)

        # id2 is at front
        best = self.queue.get_best_target()
        self.assertEqual(best.id, id2)

        # Mark id2 as PICKED_UP
        self.queue.set_target_status(id2, TargetStatus.PICKED_UP)

        # best should now be id1, because id2 is no longer PENDING
        best_new = self.queue.get_best_target()
        self.assertEqual(best_new.id, id1)

    def test_status_transitions_and_dropping(self):
        t_id = self.queue.add_user_target([0,0,0], "drop")
        
        # Test transition to PICKED_UP
        self.queue.set_target_status(t_id, TargetStatus.PICKED_UP)
        info = self.queue.get_target_info(t_id)
        self.assertEqual(info.status, TargetStatus.PICKED_UP)

        # Test transition to DROPPED (should remove from queue)
        self.queue.set_target_status(t_id, TargetStatus.DROPPED)
        
        self.assertIsNone(self.queue.get_target_info(t_id))
        self.assertEqual(len(self.queue.get_queue_snapshot().targets), 0)

    def test_reordering(self):
        # Setup: [C, B, A] (User adds to front)
        id_a = self.queue.add_user_target([1,0,0], "A")
        id_b = self.queue.add_user_target([2,0,0], "B")
        id_c = self.queue.add_user_target([3,0,0], "C")

        initial_snapshot = self.queue.get_queue_snapshot().targets
        self.assertEqual(initial_snapshot[0].id, id_c)
        self.assertEqual(initial_snapshot[1].id, id_b)
        self.assertEqual(initial_snapshot[2].id, id_a)

        # Move A (currently index 2) to front (index 0)
        self.queue.reorder_target(id_a, 0)
        
        new_snapshot = self.queue.get_queue_snapshot().targets
        self.assertEqual(new_snapshot[0].id, id_a)
        self.assertEqual(new_snapshot[1].id, id_c)
        self.assertEqual(new_snapshot[2].id, id_b)

if __name__ == '__main__':
    unittest.main()