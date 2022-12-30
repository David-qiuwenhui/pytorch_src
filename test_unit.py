import unittest


def my_div(a, b):
    return a / b


class TestFunc(unittest.TestCase):
    def test_div(self):
        self.assertEqual(2, my_div(2, 1))
        self.assertEqual(-2, my_div(2, -1))
        # self.assertEqual(1, my_div(2, 0))


if __name__ == "__main__":
    unittest.main()
