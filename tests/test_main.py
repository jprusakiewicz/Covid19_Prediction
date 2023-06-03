from unittest import TestCase

from src.main import run


class TestMain(TestCase):
    def test_run(self):
        # when
        results = run()
        # assert
        self.assertIsInstance(results, dict)

    def test_test_accuracy_higher_than_zero(self):
        # when
        results = run()
        # assert
        self.assertGreater(results['RMSE'], 0)

