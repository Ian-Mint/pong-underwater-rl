import unittest

from utils.dashboard.dashboard.data_loader import *

experiments = ['4-t2', '5-t2']


class TestUtils(unittest.TestCase):
    def test_get_rewards_history_df_columns_are_experiments(self):
        df = get_rewards_history_df('usr', experiments)
        self.assertEqual(experiments, list(df.columns))

    def test_get_experiments_returns_a_list_of_dicts_with_keys_label_and_value(self):
        exp = get_experiments_for_dropdown('usr')
        self.assertIsInstance(exp, list)

        for e in exp:
            self.assertIsInstance(e, dict)
            self.assertEqual(['label', 'value'], list(e.keys()))

    def test_get_all_grid_search_params_returns_a_dict(self):
        param_dict = get_all_grid_search_params('usr')
        self.assertIsInstance(param_dict, dict)


if __name__ == '__main__':
    unittest.main()
