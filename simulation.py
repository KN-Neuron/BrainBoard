import random

from instance import Instance


class Simulation(object):
    def __init__(self, number_of_instances, steps, cross_rate, mutation_rate, chars, text):
        self.number_of_instances = number_of_instances
        self.steps = steps
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.chars = chars
        self.text = text
        self.population = []
        self.generate_population()

    def generate_population(self):
        self.population = []
        for i in range(self.number_of_instances):
            self.population.append(Instance(self.chars, self.text))

    def find_parent(self):
        i1 = self.population[random.randint(0,self.number_of_instances - 1)]
        i2 = self.population[random.randint(0,self.number_of_instances - 1)]
        return i1 if i1.fitness < i2.fitness else i2

    def crossover(self, i1, i2):
        if random.random() > self.cross_rate:
            return list(i1.combination)

        size = len(i1.combination)
        start, end = sorted(random.sample(range(size), 2))

        child_combination = [-1] * size

        child_combination[start:end] = i1.combination[start:end]

        i2_filtered = [x for x in i2.combination if x not in child_combination]

        idx = 0
        for i in range(size):
            if child_combination[i] == -1:
                child_combination[i] = i2_filtered[idx]
                idx += 1
        return child_combination

    def mutate(self, combination):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(combination)), 2)
            combination[idx1], combination[idx2] = combination[idx2], combination[idx1]
        return combination

    def run(self):
        for i in range(self.steps):
            new_population = []
            current_best = self.find_best()
            new_population.append(Instance(self.chars, self.text, current_best.combination, current_best.fitness))
            num_immigrants = int(self.number_of_instances * 0.05)
            for _ in range(num_immigrants):
                new_population.append(Instance(self.chars, self.text))
            while len(new_population) < self.number_of_instances:
                p1 = self.find_parent()
                p2 = self.find_parent()
                child_combination = self.crossover(p1, p2)
                child_combination = self.mutate(child_combination)
                if child_combination == p1.combination:
                    new_population.append(Instance(self.chars, self.text, child_combination, p1.fitness))
                else:
                    new_population.append(Instance(self.chars, self.text, child_combination))
            self.population = new_population
            if i % 50 == 0:
                print(f"Generacja: {i} | Najlepszy fitness: {self.find_best().fitness}")

    def find_best(self):
        best = self.population[0]
        for i in range(self.number_of_instances):
            if best.fitness > self.population[i].fitness:
                best = self.population[i]
        return best