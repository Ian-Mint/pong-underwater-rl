import glob
import math
import os
import shutil
import unittest

from gym_dynamic_pong.envs import DynamicPongEnv
from utils import Line, Point
from utils.sprites import Ball


class TestLine(unittest.TestCase):
    def test_angle_between_up_line_and_ray_from_bottom_left_is_positive(self):
        l1 = Line((0, -1), (0, 1))
        l2 = Line((-1, -1), (1, 1))
        angle = l1.angle_to_normal(l2)
        self.assertGreater(angle, 0)

    def test_angle_between_up_line_and_ray_from_top_left_is_negative(self):
        l1 = Line((0, -1), (0, 1))
        l2 = Line((-1, 1), (1, -1))
        angle = l1.angle_to_normal(l2)
        self.assertLess(angle, 0)

    def test_angle_between_down_line_and_ray_from_bottom_left_is_negative(self):
        l1 = Line((0, 1), (0, -1))
        l2 = Line((-1, -1), (1, 1))
        angle = l1.angle_to_normal(l2)
        self.assertLess(angle, 0)

    def test_angle_between_down_line_and_ray_from_top_left_is_positive(self):
        l1 = Line((0, 1), (0, -1))
        l2 = Line((-1, 1), (1, -1))
        angle = l1.angle_to_normal(l2)
        self.assertGreater(angle, 0)

    def test_10_10_20_20_and_10_20_20_10_intersect_at_15_15(self):
        line1 = Line((10, 10), (20, 20))
        line2 = Line((10, 20), (20, 10))
        intersection = line1.get_intersection(line2)
        self.assertEqual(Point(15., 15.), intersection)

    def test_same_lines_do_not_intersect(self):
        line1 = Line((10, 10), (20, 20))
        line2 = line1
        intersection = line1.get_intersection(line2)
        self.assertEqual(None, intersection)

    def test_non_parallel_non_intersecting_segments_do_not_intersect_1(self):
        line1 = Line((10, 10), (20, 20))
        line2 = Line((10, 20), (14, 16))
        intersection = line1.get_intersection(line2)
        self.assertEqual(None, intersection)

    def test_non_parallel_non_intersecting_segments_do_not_intersect_2(self):
        line1 = Line((10, 10), (20, 20))
        line2 = Line((1, 5), (20, 5))
        intersection = line1.get_intersection(line2)
        self.assertEqual(None, intersection)

    def test_close_non_parallel_paddles_do_not_intersect(self):
        line1 = Line((10, 10), (20, 20))
        line2 = Line((10.1, 11), (10.5, 11.1))
        intersection = line1.get_intersection(line2)
        self.assertEqual(None, intersection)

    def test_vertical_line(self):
        line1 = Line((10., 10.), (10., 20.))
        line2 = Line((5., 15.), (15., 15.))
        intersection = line1.get_intersection(line2)
        self.assertEqual(Point(10., 15.), intersection)

    def test_top_of_screen_trajectory(self):
        trajectory = Line((349.824288866177, 298.7636286943965), (351.9456092097367, 300.8849490379562))
        border = Line((0, 300), (400, 300))
        intersection = trajectory.get_intersection(border)
        dx = (300. - trajectory.start.y) / (
                    (trajectory.end.y - trajectory.start.y) / (trajectory.end.x - trajectory.start.x))
        self.assertEqual(Point(dx + trajectory.start.x, 300.), intersection)


class TestPoint(unittest.TestCase):
    def test_point_iter(self):
        point1 = Point(0, 1)
        v = [p for p in point1]
        self.assertEqual([0, 1], v)

    def test_point_tuple_conversion(self):
        point1 = Point(0, 1)
        self.assertEqual((0, 1), tuple(point1))


class TestRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 160
        self.height = 160
        self.default_speed = 2
        self.snell_speed = 2
        self.paddle_speed = 3
        self.paddle_height = 30
        pong_env = DynamicPongEnv(max_score=5, width=self.width,
                                  height=self.height,
                                  default_speed=self.default_speed,
                                  snell_speed=self.snell_speed,
                                  our_paddle_speed=self.paddle_speed,
                                  their_paddle_speed=self.paddle_speed,
                                  our_paddle_height=self.paddle_height,
                                  their_paddle_height=self.paddle_height, )
        self.env = pong_env
        self.env.step(0)

        self.save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'artifacts'))

    def test_png_render_creates_directory_if_it_doesnt_exist(self):
        self.env.render('png', self.save_dir)
        self.assertTrue(os.path.exists(self.save_dir))

    def test_png_render_creates_png_file(self):
        self.env.render('png', self.save_dir)
        match = glob.glob(os.path.join(self.save_dir, "**", "*.png"), recursive=True)
        self.assertGreater(len(match), 0)

    def tearDown(self) -> None:
        shutil.rmtree(self.save_dir)


class TestBall(unittest.TestCase):
    def setUp(self) -> None:
        self.ball = Ball(100, 100)

    def test_unit_velocity_sets_angle_to_0_for_1_0(self):
        self.ball.unit_velocity = Point(1., 0.)
        self.assertAlmostEqual(0, self.ball.angle)

    def test_unit_velocity_sets_angle_to_30_for_sqrt3_1(self):
        self.ball.unit_velocity = Point(math.sqrt(3), 1)
        self.assertAlmostEqual(math.pi / 6, self.ball.angle)

    def test_unit_velocity_sets_angle_to_45_for_1_1(self):
        self.ball.unit_velocity = Point(1., 1.)
        self.assertAlmostEqual(math.pi / 4, self.ball.angle)

    def test_unit_velocity_sets_angle_to_135_for_neg1_1(self):
        self.ball.unit_velocity = Point(-1., 1.)
        self.assertAlmostEqual(math.pi * 3 / 4, self.ball.angle)

    def test_unit_velocity_sets_angle_to_150_for_neg_sqrt3_1(self):
        self.ball.unit_velocity = Point(-math.sqrt(3), 1)
        self.assertAlmostEqual(math.pi * 5 / 6, self.ball.angle)

    def test_unit_velocity_sets_angle_to_180_for_neg1_0(self):
        self.ball.unit_velocity = Point(-1., 0.)
        self.assertAlmostEqual(math.pi, self.ball.angle)

    def test_unit_velocity_sets_angle_to_225_for_neg1_neg1(self):
        self.ball.unit_velocity = Point(-1., -1.)
        self.assertAlmostEqual(math.pi * 5 / 4, self.ball.angle)

    def test_unit_velocity_sets_angle_to_315_for_1_neg1(self):
        self.ball.unit_velocity = Point(1., -1.)
        self.assertAlmostEqual(math.pi * 7 / 4, self.ball.angle)