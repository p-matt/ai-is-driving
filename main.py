import statistics

from statistics import mean
import pickle

from model import *
from utils import *
from game import Gui, PlayerCar, AbstractCar
from evolution import GA

import numpy as np


def get_numbers_of_car_alive(car_pop):
    return sum([1 for car in car_pop if not car.has_collided])


def get_numbers_of_car_stuck(car_pop, dist):
    return sum([1 for car in car_pop if car.get_distance_from_start() < dist and not car.has_collided])


def get_average_speed(car_pop):
    try:
        vel_sorted = [ai.vel for ai in sorted([ai for ai in car_pop if ai.has_collided is not True and ai.vel > 0], key=lambda x: x.distance_from_start)]
        return round(mean(vel_sorted[-3:]), 2), round(mean(vel_sorted[-10:]), 2)
    except statistics.StatisticsError:
        return -1, -1


def get_car_population(ga, n_gen):
    if n_gen == 1:
        return [PlayerCar(gui, is_parent=True) for i in range(ga.population_size)]

    return [PlayerCar(gui, is_parent=True if i < ga.parents_size else False) for i in range(ga.population_size)]


def pre_process(cnet, stats, car_population, draw_raycast, n_steps):
    user_skip_the_generation = False
    for event in gui.pg.event.get():
        if event.type == gui.pg.MOUSEBUTTONDOWN:
            draw_raycast = not draw_raycast
        if event.type == gui.pg.KEYDOWN:
            if event.key == gui.pg.K_SPACE:
                user_skip_the_generation = True
            if event.key == gui.pg.K_ESCAPE:
                with open(f'data/stats/{len(car_population)}/unlimited/results.pickle', 'wb') as handle:
                    pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
                cnet.save()
    car_alive = get_numbers_of_car_alive(car_population)
    game_skip = car_alive < 1
    if n_steps > 50:
        car_stucks = get_numbers_of_car_stuck(car_population, 200)
        game_skip = game_skip or (car_stucks == car_alive)
    elif n_steps > 40:
        car_stucks = get_numbers_of_car_stuck(car_population, 100)
        game_skip = game_skip or (car_stucks == car_alive)
    return user_skip_the_generation, car_alive, game_skip, draw_raycast


def wait_in_the_menu():
    blit_text_center(gui.window, gui.main_font, f"Press any key to start !")
    gui.pg.display.update()
    for event in gui.pg.event.get():
        if event.type == gui.pg.QUIT:
            gui.pg.quit()
            break
        if event.type == gui.pg.KEYDOWN or event.type == gui.pg.MOUSEBUTTONDOWN:
            return False
        gui.clock.tick(gui.fps_in_menu)
    return True


def step(car_pop, cnet, draw_ray=False):
    X = [car.get_context(draw_ray) for car in car_pop]
    y_pred_index = [np.argmax(y_pred_sample, axis=1)[0] for y_pred_sample in cnet.make_prediction(X)]
    for car, y_pred_sample_index in zip(car_pop, y_pred_index):
        if car.has_collided:
            continue

        move = AbstractCar.ai_possible_moves[y_pred_sample_index]

        car.move(next_move=move)
        car.handle_collision()

        if not car.has_collided:
            car.get_distance_from_start()

        gui.pg.event.pump()


def combine_weights_parents_descendants(weights_l1, weights_l2, biases_l1, biases_l2, wp_ng, bp_ng):
    weights_l1 = np.concatenate((wp_ng[0], weights_l1))
    weights_l2 = np.concatenate((wp_ng[1], weights_l2))
    weights = np.concatenate((weights_l1, weights_l2))

    biases_l1 = np.concatenate((bp_ng[0], biases_l1))
    biases_l2 = np.concatenate((bp_ng[1], biases_l2))
    biases = np.concatenate((biases_l1, biases_l2))
    return weights, biases


def learn(n_turns, ga, cnet):
    stats = {"best_fitness": [], "mean_fitness": [], "mean_top3_speed": [], "mean_top10_speed": []}

    for n_generation in range(1, ga.generation):
        print(n_generation)
        speed_history = []
        draw_raycast = False
        car_population = get_car_population(ga, n_generation)

        for n_turn in range(n_turns):
            user_skip, n_car_alive, game_skip, draw_raycast = pre_process(cnet, stats, car_population, draw_raycast,
                                                                          n_turn)
            if any(x for x in (user_skip, game_skip)):
                break

            step(car_population, cnet, draw_raycast)

            avg_speed = get_average_speed(car_population)
            gui.draw_main_overlay(car_population, n_generation, n_car_alive, avg_speed)
            speed_history.append(avg_speed)
            gui.clock.tick(gui.fps_in_game)
            # print(gui.clock.get_fps())

        fitness = ga.get_fitness(car_population)

        wp_co, bp_co, wp_ng, bp_ng = ga.natural_selection(fitness, *cnet.get_weights_and_biases())

        weights, biases = combine_weights_parents_descendants(*ga.mutation(*ga.crossover(wp_co, bp_co)), wp_ng, bp_ng)

        cnet.set_weights_and_biases(weights, biases)

        stats["best_fitness"].append(max(fitness))
        stats["mean_fitness"].append(mean(fitness))
        stats["mean_top3_speed"].append(np.array(speed_history)[:, 0])
        stats["mean_top10_speed"].append(np.array(speed_history)[:, 1])


def start():
    run = True
    in_the_menu = True
    n_turns = 10000
    n_gen, pop_size, selection_rate, mutation_rate = 1000, 100, .15, .01

    cnet = CarNet(pop_size)
    ga = GA(n_gen, pop_size, selection_rate, mutation_rate)

    while run:
        if in_the_menu:
            in_the_menu = wait_in_the_menu()
            continue

        learn(n_turns, ga, cnet)

    gui.pg.quit()


if __name__ == "__main__":
    gui = Gui()
    start()
