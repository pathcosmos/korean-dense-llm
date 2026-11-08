import unittest
from data.download import select_entries


class DownloadSelectionTest(unittest.TestCase):
    def test_uniform_and_prefix(self):
        items = list(range(1126))
        sample = select_entries(items, 64, True)
        self.assertEqual(len(set(sample)), 64)
        self.assertEqual((sample[0], sample[-1]), (0, 1125))
        self.assertEqual(sample, sorted(sample))
        self.assertEqual(select_entries(items, 3), [0, 1, 2])
        self.assertEqual(select_entries([1, 2], 10, True), [1, 2])

    def test_invalid_selection(self):
        with self.assertRaises(ValueError):
            select_entries([1], None, True)
        with self.assertRaises(ValueError):
            select_entries([1], 0)
