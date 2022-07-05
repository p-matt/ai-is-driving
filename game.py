import math
import random
import pygame as pg
import numpy as np

from numpy import array
from utils import scale_image, blit_rotate_center, get_distance


class Gui:

    def __init__(self):
        self.pg = pg
        self.pg.font.init()
        self.pg.display.set_caption("AI learn to drive")

        self.grass = scale_image(self.pg.image.load("data/imgs/grass.jpg"), 2.5)
        self.track = scale_image(self.pg.image.load("data/imgs/track.png"), 0.9)
        self.track_border = scale_image(self.pg.image.load("data/imgs/track-border.png"), 0.9)
        self.track_border_mask = self.pg.mask.from_surface(self.track_border)

        self.finish = self.pg.image.load("data/imgs/finish.png")
        self.finish_mask = self.pg.mask.from_surface(self.finish)
        self.finish_position = (130, 250)

        self.red_car = scale_image(self.pg.image.load("data/imgs/red-car.png"), 0.35)
        self.green_car = scale_image(self.pg.image.load("data/imgs/green-car.png"), 0.35)
        self.grey_car = scale_image(self.pg.image.load("data/imgs/grey-car.png"), 0.35)
        self.car_shape = self.red_car.get_size()
        self.car_mask = self.pg.mask.Mask((8, 8), True)

        self.width, self.height = self.track.get_width(), self.track.get_height()
        self.window = self.pg.display.set_mode((self.width, self.height))

        self.main_font = self.pg.font.SysFont("comicsans", 44)
        self.sec_font = self.pg.font.SysFont("arial", 22)

        self.fps_in_game, self.fps_in_menu = 10 ** 5, 30
        self.clock = self.pg.time.Clock()
        self.images = [(self.grass, (0, 0)), (self.track, (0, 0)), (self.finish, self.finish_position),
                       (self.track_border, (0, 0))]
        self.path = [(178, 235), (178, 225), (177, 220), (177, 210), (177, 200), (176, 193), (176, 188), (177, 178),
                     (177, 165),
                     (177, 159),
                     (176, 146), (176, 136), (174, 123), (169, 110), (166, 103), (153, 92), (145, 86), (135, 79),
                     (123, 75),
                     (112, 75),
                     (103, 77), (92, 81), (87, 83), (78, 90), (70, 102), (66, 109), (62, 118), (60, 127), (58, 139),
                     (60, 151),
                     (62, 163), (62, 177), (60, 195), (60, 206), (58, 227), (58, 234), (59, 251), (60, 265), (60, 277),
                     (58, 289),
                     (57, 301), (56, 321), (56, 338), (58, 355), (58, 367), (58, 376), (58, 391), (57, 403), (56, 410),
                     (58, 427),
                     (61, 444), (66, 461), (72, 476), (79, 488), (84, 493), (93, 501), (100, 510), (108, 517),
                     (114, 521),
                     (119, 527),
                     (126, 534), (138, 546), (152, 559), (158, 567), (168, 579), (178, 591), (182, 597), (191, 604),
                     (201, 613),
                     (213, 625), (221, 633), (232, 646), (240, 656), (251, 665), (257, 672), (270, 683), (282, 694),
                     (296, 702),
                     (308, 711), (323, 720), (334, 722), (354, 716), (374, 707), (389, 698), (393, 694), (400, 687),
                     (407, 677),
                     (409, 668), (409, 654), (409, 626), (408, 597), (408, 589), (410, 562), (413, 540), (417, 523),
                     (426, 508),
                     (438, 500), (449, 492), (477, 480), (489, 478), (507, 476), (521, 478), (537, 489), (557, 500),
                     (567, 507),
                     (581, 526), (588, 537), (595, 564), (596, 580), (596, 589), (596, 608), (595, 626), (598, 646),
                     (601, 663),
                     (601, 674), (603, 689), (605, 697), (615, 710), (636, 715), (665, 716), (701, 714), (716, 709),
                     (729, 705),
                     (738, 689), (738, 666), (739, 636), (737, 614), (736, 588), (736, 577), (737, 551), (737, 533),
                     (738, 512),
                     (738, 491), (737, 464), (737, 454), (736, 439), (736, 415), (727, 393), (714, 374), (669, 359),
                     (646, 357),
                     (617, 355), (576, 357), (553, 357), (526, 357), (493, 358), (451, 357), (435, 355), (419, 351),
                     (407, 341),
                     (400, 320), (401, 297), (408, 280), (432, 261), (448, 258), (475, 257), (519, 257), (545, 257),
                     (564, 257),
                     (594, 256), (619, 257), (650, 258), (675, 258), (700, 254), (714, 251), (735, 232), (739, 214),
                     (745, 192),
                     (744, 167), (742, 152), (736, 129), (732, 113), (719, 96), (701, 86), (677, 83), (646, 79),
                     (628, 78),
                     (597, 76),
                     (584, 76), (564, 75), (543, 75), (520, 77), (500, 77), (476, 77), (452, 75), (432, 75), (409, 75),
                     (391, 74),
                     (360, 73), (329, 73), (326, 73), (309, 76), (298, 83), (289, 96), (284, 114), (283, 125),
                     (280, 160),
                     (280, 175),
                     (278, 196), (276, 221), (274, 237), (275, 262), (275, 283), (275, 305), (275, 322), (274, 341),
                     (275, 358),
                     (274, 369), (270, 380), (255, 390), (242, 397), (219, 400), (189, 400), (179, 394), (170, 384),
                     (167, 364), (166, 349), (162, 332)]

    def draw_main_overlay(self, pop, gen, car_alive, speed):
        _ = [self.window.blit(img, pos) for img, pos in self.images]

        _ = [ai.draw() for ai in pop]

        gen_text = self.sec_font.render(f"{'Generation': <22} {gen: >4}", 1, (255, 255, 255))
        self.window.blit(gen_text, (5, self.height - gen_text.get_height() - 70))

        alive_text = self.sec_font.render(f"{'Car alive': <22} {car_alive: >4}", 1, (255, 255, 255))
        self.window.blit(alive_text, (5, self.height - alive_text.get_height() - 40))

        speed_text = self.sec_font.render(f"{'Avg. speed top 3': <22} {speed[0]: >4}", 1, (255, 255, 255))
        self.window.blit(speed_text, (5, self.height - speed_text.get_height() - 10))
        self.pg.display.flip()
        self.clock.tick(self.fps_in_game)


class AbstractCar:
    angles_ray = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]
    ai_possible_moves = [{97: False, 122: True, 114: False}, {97: True, 122: True, 114: False},
                         {97: False, 122: True, 114: True}, {97: False, 122: False, 114: False}]

    def __init__(self, gui, max_vel=math.inf, rotation_vel=15, is_parent=False):
        self.gui = gui
        self.img = self.gui.red_car if is_parent else self.gui.grey_car
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.start_position = (random.randint(176, 180), random.randint(239, 241))  # (178, 240)
        self.x, self.y = self.start_position
        self.acceleration = 0.4
        self.current_lap = 1
        self.distance_from_start = self.get_distance_from_start()
        self.start = False
        self.end = False
        self.has_collided = False
        self.n_step = 0

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def get_context(self, draw_ray=False):
        self.n_step += 1
        raycast_distances = []

        for angle_ray in AbstractCar.angles_ray:
            vector = self.gui.pg.Vector2()  # A zero vector.
            collision = False
            index = 1
            while collision is False:
                vector_height = 2 * index
                vector.from_polar((vector_height, -self.angle - angle_ray))
                target_pos = (self.x, self.y) + vector
                try:
                    collision = (self.gui.track_border_mask.get_at(target_pos) == 1)
                except IndexError:
                    collision = None
                if collision:
                    if draw_ray and self.has_collided is False:
                        self.gui.pg.draw.line(self.gui.window, (124, 252, 0), (self.x, self.y), target_pos, 1)
                        self.gui.pg.draw.circle(self.gui.window, (255, 0, 0), [target_pos[0], target_pos[1]], 2)
                    raycast_distances.append(1 / vector_height)

                index += 1
        if draw_ray:
            self.gui.pg.display.update()

        return np.array(raycast_distances + [1 / (self.vel if self.vel != 0 else 1)]).reshape(1, -1)

    def draw(self):
        blit_rotate_center(self.gui.window, self.img,
                           (self.x - self.gui.car_shape[0] / 2, self.y - self.gui.car_shape[1] / 2), self.angle)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.update_position()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel / 2)
        self.update_position()

    def update_position(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0, finish_mask=False):
        offset = (int(self.x - (self.gui.car_shape[0] / 4) - x), int(self.y - (self.gui.car_shape[1] / 8) - y))
        poi = mask.overlap(self.gui.car_mask, offset)
        if finish_mask:
            if poi:
                if self.n_step < 30:  # workaround to be able to detect if the colision with the fish_mask means this is a new lap or just a colision from the start
                    return True
                self.current_lap += 1
            return False
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0

    def get_distance_from_start(self):
        index = array([math.hypot(self.x - x, self.y - y) for x, y in self.gui.path]).argmin()
        path_from_start = self.gui.path[:index + 1]
        path_from_start.insert(0, self.start_position)

        distance_from_start = get_distance(path_from_start) * self.current_lap

        self.distance_from_start = distance_from_start
        return distance_from_start


class PlayerCar(AbstractCar):

    def reduce_speed(self):
        self.vel = max(self.vel - (self.acceleration * self.vel), 0)
        self.update_position()

    def stop(self):
        self.vel = 0
        self.update_position()

    def bounce(self):
        self.vel = -self.vel
        self.update_position()

    def move(self, next_move):
        reduce_speed = True

        if next_move[self.gui.pg.K_a]:
            self.rotate(left=True)

        if next_move[self.gui.pg.K_r]:
            self.rotate(right=True)

        if next_move[self.gui.pg.K_z]:
            reduce_speed = False
            self.move_forward()

        if reduce_speed:
            self.reduce_speed()

    def handle_collision(self):
        if any([self.collide(self.gui.track_border_mask),
                self.collide(self.gui.finish_mask, x=self.gui.finish_position[0], y=self.gui.finish_position[1],
                             finish_mask=True)]):
            self.stop()
            self.has_collided = True
