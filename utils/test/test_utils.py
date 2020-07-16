import unittest

from utils.dashboard.dashboard.data_loader import *
from utils.dashboard.dashboard.dashboard import monitor_memoized, cache, PreventUpdate

experiments = ['4-t2', '5-t2']


class TestDashboard(unittest.TestCase):
    def test_monitor_memoized_raises_PreventUpdate_if_no_change(self):
        @cache.memoize()
        def foo(value):
            foo.count += 1
            if foo.count < 3:
                return value
            else:
                foo.count = 0
                return None
        foo.count = 0

        @cache.memoize()
        def bar(value):
            pass
        self.assertRaises(PreventUpdate, monitor_memoized, foo, bar, 'arg')
        
    def test_monitor_memoized_returns_updated_value(self):
        @cache.memoize()
        def foo(value):
            foo.count += 1
            if foo.count < 2:
                return 'wrong'
            else:
                return 'right'
        foo.count = 0

        @cache.memoize()
        def bar(value):
            bar.count += 1
            if bar.count < 2:
                return 'wrong'
            else:
                return 'right'
        bar.count = 0

        self.assertEqual('wrong', foo('arg'))
        self.assertEqual('wrong', bar('arg'))

        result = monitor_memoized(foo, bar, 'arg')
        self.assertEqual('right', result)



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
