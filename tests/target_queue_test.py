import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from target_queue import TargetQueue, TargetStatus

class TestTargetQueue(unittest.TestCase):
    def setUp(self):
        # Initialize a fresh queue before every test to ensure isolation
        self.queue = TargetQueue(duplicate_threshold=0.5)

    def test_add_user_target_priority(self):
        # User targets should always be inserted at the front of the queue
        id1 = self.queue.add_user_target([10, 0, 0], "bin_a")
        id2 = self.queue.add_user_target([20, 0, 0], "bin_b")

        snapshot = self.queue.get_queue_snapshot()
        
        # Expecting LIFO behavior for user additions based on "insert(0, ...)" logic
        self.assertEqual(snapshot[0]['id'], id2)
        self.assertEqual(snapshot[1]['id'], id1)
        self.assertEqual(snapshot[0]['source'], 'user')

    def test_numpy_conversion_and_handling(self):
        # Verify that inputs (lists/tuples) are correctly converted to numpy arrays internally
        pos_input = [1.5, 2.5, 3.5]
        drop_input = (10.0, 10.0, 10.0)
        
        t_id = self.queue.add_user_target(pos_input, drop_input)
        
        info = self.queue.get_target_info(t_id)
        self.assertIsInstance(info['position'], np.ndarray)
        self.assertIsInstance(info['dropoff'], np.ndarray)
        np.testing.assert_array_equal(info['position'], np.array(pos_input))

    def test_ai_target_addition_and_pruning(self):
        # 1. Add an initial batch of AI targets
        batch_1 = [
            {'position': [10, 10, 0], 'dropoff': 'bin_a'},
            {'position': [20, 20, 0], 'dropoff': 'bin_b'}
        ]
        self.queue.add_ai_targets(batch_1)
        
        snapshot_1 = self.queue.get_queue_snapshot()
        self.assertEqual(len(snapshot_1), 2)
        ids_1 = {t['id'] for t in snapshot_1}

        # 2. Add a second batch that is completely different
        # The previous targets should be pruned because they are not in this new batch
        batch_2 = [
            {'position': [30, 30, 0], 'dropoff': 'bin_c'}
        ]
        self.queue.add_ai_targets(batch_2)
        
        snapshot_2 = self.queue.get_queue_snapshot()
        self.assertEqual(len(snapshot_2), 1)
        
        # Verify that the old IDs are gone and we have a new ID
        self.assertNotIn(snapshot_2[0]['id'], ids_1)
        np.testing.assert_array_equal(snapshot_2[0]['position'], [30, 30, 0])

    def test_ai_reconciliation_update_existing(self):
        # Test that if AI submits a target close to an existing AI target, we update it rather than replace it
        
        # Initial AI target
        self.queue.add_ai_targets([
            {'position': [10.0, 10.0, 0.0], 'dropoff': 'old_drop'}
        ])
        original_id = self.queue.get_queue_snapshot()[0]['id']

        # New batch with a target very close (within threshold 0.5)
        self.queue.add_ai_targets([
            {'position': [10.1, 10.0, 0.0], 'dropoff': 'new_drop'}
        ])
        
        snapshot = self.queue.get_queue_snapshot()
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0]['id'], original_id, "ID should be preserved for updates")
        self.assertEqual(snapshot[0]['dropoff'], 'new_drop', "Dropoff should be updated")
        # Position should reflect the fresh sensor data
        np.testing.assert_array_almost_equal(snapshot[0]['position'], [10.1, 10.0, 0.0])

    def test_ai_cannot_override_user_target(self):
        # If a User target exists, AI seeing the same object should not modify it or remove it
        
        user_pos = [5.0, 5.0, 0.0]
        user_id = self.queue.add_user_target(user_pos, "user_drop")

        # AI submits a target very close to the user target
        ai_pos = [5.1, 5.0, 0.0] # Within 0.5 threshold
        self.queue.add_ai_targets([
            {'position': ai_pos, 'dropoff': 'ai_drop'}
        ])

        snapshot = self.queue.get_queue_snapshot()
        
        self.assertEqual(len(snapshot), 1, "Should reconcile to a single target")
        self.assertEqual(snapshot[0]['id'], user_id, "User ID must be preserved")
        self.assertEqual(snapshot[0]['source'], 'user', "Source must remain 'user'")
        self.assertEqual(snapshot[0]['dropoff'], 'user_drop', "User dropoff preference must be preserved")
        
        # Crucially, the position should NOT update to the AI's noisy reading
        np.testing.assert_array_equal(snapshot[0]['position'], user_pos)

    def test_user_targets_survive_ai_pruning(self):
        # AI batches prune unmatched AI targets, but they should never touch User targets
        
        self.queue.add_user_target([100, 100, 0], "safe_zone")
        
        # Empty AI batch implies "I see nothing". 
        # This should clear any AI targets (none here) but leave User targets alone.
        self.queue.add_ai_targets([])
        
        snapshot = self.queue.get_queue_snapshot()
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0]['source'], 'user')

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
        self.assertEqual(info['status'], 'picked_up')

        # Test transition to DROPPED (should remove from queue)
        self.queue.set_target_status(t_id, TargetStatus.DROPPED)
        
        self.assertIsNone(self.queue.get_target_info(t_id))
        self.assertEqual(len(self.queue.get_queue_snapshot()), 0)

    def test_reordering(self):
        # Setup: [C, B, A] (User adds to front)
        id_a = self.queue.add_user_target([1,0,0], "A")
        id_b = self.queue.add_user_target([2,0,0], "B")
        id_c = self.queue.add_user_target([3,0,0], "C")

        initial_snapshot = self.queue.get_queue_snapshot()
        self.assertEqual(initial_snapshot[0]['id'], id_c)
        self.assertEqual(initial_snapshot[1]['id'], id_b)
        self.assertEqual(initial_snapshot[2]['id'], id_a)

        # Move A (currently index 2) to front (index 0)
        self.queue.reorder_target(id_a, 0)
        
        new_snapshot = self.queue.get_queue_snapshot()
        self.assertEqual(new_snapshot[0]['id'], id_a)
        self.assertEqual(new_snapshot[1]['id'], id_c)
        self.assertEqual(new_snapshot[2]['id'], id_b)

    def test_snapshot_json_compatibility(self):
        # Ensure the snapshot doesn't contain numpy types which break standard json.dumps
        self.queue.add_user_target(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        
        snapshot = self.queue.get_queue_snapshot()
        target_data = snapshot[0]
        
        self.assertIsInstance(target_data['position'], list)
        self.assertIsInstance(target_data['dropoff'], list)
        self.assertIsInstance(target_data['position'][0], float)
        
        # Ensure we aren't leaking numpy scalars
        self.assertNotIsInstance(target_data['position'][0], np.float64)

if __name__ == '__main__':
    unittest.main()