import copy
import random
from heapq import nlargest
from math import ceil

import numpy as np

class GA:

    def __init__(self, generation, population_size, selection_rate, mutation_rate=.1, cnet=None):
        self.generation = generation
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate # 1 / ((cnet.pbc * INPUT_SIZE) + (cnet.pbc * OUTPUT_SIZE))
        self.parents_size = ceil(population_size * selection_rate)  # parents to generate new pop from
        self.preserved_parents_size = ceil(self.parents_size * .5) if ceil(self.parents_size * .5) % 2 == 0 else ceil(self.parents_size * .5) + 1  # parents to keep for generation n+1 (even number)
        self.descendants_size = population_size - self.preserved_parents_size
        self.child_by_parents = 2

    @staticmethod
    def get_fitness(population):
        fitness = [car.distance_from_start for car in population]
        print("average fitness", sum(fitness) / len(fitness))
        print("best fitness", max(fitness))
        return fitness

    def natural_selection(self, fitness, weights, biases):

        parents_for_crossover_index = nlargest(self.parents_size, list(range(self.population_size)),
                                               key=lambda i: fitness[i])
        weight_parents_co = [weights[0][parents_for_crossover_index], weights[1][parents_for_crossover_index]]
        bias_parents_co = [biases[0][parents_for_crossover_index], biases[1][parents_for_crossover_index]]

        parents_for_next_generation = nlargest(self.preserved_parents_size, list(range(self.population_size)),
                                               key=lambda i: fitness[i])

        weight_parents_ng = np.array([weights[0][parents_for_next_generation], weights[1][parents_for_next_generation]])
        bias_parents_ng = np.array([biases[0][parents_for_next_generation], biases[1][parents_for_next_generation]])

        return weight_parents_co, bias_parents_co, weight_parents_ng, bias_parents_ng

    def crossover(self, _weights, _biases):
        pair_weights = []
        pair_biases = []
        weights, biases = copy.deepcopy(_weights), copy.deepcopy(_biases)
        while len(pair_weights) != ceil(self.descendants_size / self.child_by_parents):
            indexes = list(range(self.parents_size))
            idx1 = random.choice(indexes)
            indexes.remove(idx1)
            idx2 = random.choice(indexes)
            w1_l1, w1_l2, b1_l1, b1_l2 = weights[0][idx1], weights[1][idx1], biases[0][idx1], biases[1][idx1]
            w2_l1, w2_l2, b2_l1, b2_l2 = weights[0][idx2], weights[1][idx2], biases[0][idx2], biases[1][idx2]
            pair_weights.append((w1_l1, w1_l2, w2_l1, w2_l2))
            pair_biases.append((b1_l1, b1_l2, b2_l1, b2_l2))

        pair_weights = np.array(pair_weights, dtype="object")
        pair_biases = np.array(pair_biases, dtype="object")

        for i, (w1_l1, w1_l2, w2_l1, w2_l2) in enumerate(pair_weights):  # iterate parent weights
            for j, w1_perceptron in enumerate(w1_l1):  # iterate on each perceptron weights for hidden layer 1
                for k, _ in enumerate(w1_perceptron):  # iterate on each weight values
                    if random.random() > .5:
                        w1_l1[j][k], w2_l1[j][k] = w2_l1[j][k], w1_l1[j][k]
                        pair_biases[i][0][k], pair_biases[i][2][k] = pair_biases[i][2][k], pair_biases[i][0][k]
            for j, w2_perceptron in enumerate(w1_l2):  # iterate on each perceptron weights for output layer
                for k, _ in enumerate(w2_perceptron):  # iterate on each weight values
                    if random.random() > .5:
                        w1_l2[j][k], w2_l2[j][k] = w2_l2[j][k], w1_l2[j][k]
                        pair_biases[i][1][k], pair_biases[i][3][k] = pair_biases[i][3][k], pair_biases[i][1][k]

        weights_l1 = np.concatenate((pair_weights[:, 0], pair_weights[:, 2]))
        weights_l2 = np.concatenate((pair_weights[:, 1], pair_weights[:, 3]))
        biases_l1 = np.concatenate((pair_biases[:, 0], pair_biases[:, 2]))
        biases_l2 = np.concatenate((pair_biases[:, 1], pair_biases[:, 3]))
        # weights_l1 = pair_weights[:, 0]
        # weights_l2 = pair_weights[:, 1]
        # biases_l1 = pair_biases[:, 0]
        # biases_l2 = pair_biases[:, 1]

        return weights_l1, weights_l2, biases_l1, biases_l2

    def mutation(self, weights_l1, weights_l2, biases_l1, biases_l2):
        for i, (w_l1, w_l2, b_l1, b_l2) in enumerate(zip(weights_l1, weights_l2, biases_l1, biases_l2)):
            for j, w1_perceptron in enumerate(w_l1):  # iterate on each perceptron weights for hidden layer 1
                for k, _ in enumerate(w1_perceptron):  # iterate on each weight values
                    if random.random() > (1 - self.mutation_rate):
                        r = random.random()
                        b = random.uniform(.95, 1.05)
                        if r < .2:
                            w_l1[j][k] *= random.uniform(1.25, 1.35) if random.random() < .9 else random.uniform(-1.35, -1.25)
                            b_l1[k] *= b
                        elif r < .4:
                            w_l1[j][k] *= random.uniform(1.35, 1.55) if random.random() < .5 else random.uniform(-1.55, -1.35)
                            b_l1[k] *= b
                        elif r < .8:
                            w_l1[j][k] *= random.uniform(1.75, 2.5) if random.random() < .5 else random.uniform(-2.5, -1.75)
                            b_l1[k] *= b
                        elif r < .9:
                            w_l1[j][k] *= random.randint(-2, 2)
            for j, w2_perceptron in enumerate(w_l2):  # iterate on each perceptron weights for hidden layer 1
                for k, _ in enumerate(w2_perceptron):  # iterate on each weight values
                    if random.random() > (1 - self.mutation_rate):
                        r = random.random()
                        b = random.uniform(.95, 1.05)
                        if r < .2:
                            w_l2[j][k] *= random.uniform(1.25, 1.35) if random.random() < .9 else random.uniform(-1.35, -1.25)
                            b_l2[k] *= b
                        elif r < .4:
                            w_l2[j][k] *= random.uniform(1.35, 1.55) if random.random() < .5 else random.uniform(-1.55, -1.35)
                            b_l2[k] *= b
                        elif r < .8:
                            w_l2[j][k] *= random.uniform(1.75, 2.5) if random.random() < .5 else random.uniform(-2.5, -1.75)
                            b_l2[k] *= b
                        elif r < .9:
                            w_l2[j][k] *= random.randint(-2, 2)

        return weights_l1, weights_l2, biases_l1, biases_l2
