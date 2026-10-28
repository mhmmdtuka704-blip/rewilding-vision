import unittest

from rewilding_vision.splits import assign_group_splits


class SplitTest(unittest.TestCase):
    def test_deterministic_and_group_safe(self):
        groups = [f"sequence_{i:02d}" for i in range(18) for _ in range(5)]
        first = assign_group_splits(groups)
        second = assign_group_splits(reversed(groups))
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "validation", "test"})
        self.assertEqual(len(first), 18)

    def test_requires_three_groups(self):
        with self.assertRaises(ValueError):
            assign_group_splits(["one", "two"])

    def test_minimum_three_groups_all_receive_a_split(self):
        assignments = assign_group_splits(["one", "two", "three"])
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
