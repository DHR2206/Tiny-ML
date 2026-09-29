import unittest

from buddy.nlu.normalize import normalize
from buddy.nlu.slots import count_slots, math_slots, number_range


def cs(text):
    return count_slots(normalize(text)).as_dict()


class TestSlots(unittest.TestCase):
    def test_ranges(self):
        for text in ["count from 1 to 10", "count 1 to 10", "count 1-10", "count 1–10", "count one through ten",
                     "count between 1 and 10", "count from one two ten", "count from 1 till 10", "count 1 10"]:
            self.assertEqual(cs(text), {"start": 1, "end": 10}, text)

    def test_single_number(self):
        self.assertEqual(cs("count to 20"), {"end": 20})
        self.assertEqual(cs("count from 5"), {"start": 5})
        self.assertEqual(cs("count backwards from 20"), {"start": 20, "backwards": True})

    def test_steps(self):
        self.assertEqual(cs("count by 2 from 2 to 20"), {"start": 2, "end": 20, "step": 2})
        self.assertEqual(cs("count in fives"), {"step": 5})
        self.assertEqual(cs("count in 10s to 100"), {"end": 100, "step": 10})

    def test_number_range_only_when_explicit(self):
        self.assertEqual(number_range(normalize("from 5 to 15")), (5, 15))
        self.assertIsNone(number_range(normalize("i am 5")))

    def test_math(self):
        self.assertEqual(math_slots(normalize("take away 2 from 5")), {"a": 5, "b": 2, "op": "minus"})
        self.assertEqual(math_slots(normalize("what is 5-3")), {"a": 5, "b": 3, "op": "minus"})
        self.assertEqual(math_slots(normalize("how much is 5 and 2")), {"a": 5, "b": 2, "op": "plus"})
        self.assertIsNone(math_slots(normalize("count 1-10")))
        self.assertIsNone(math_slots(normalize("five times table")))


if __name__ == "__main__":
    unittest.main()
