import random

def dict_init(chars,combination):
    dictionary = {}
    counter = 0
    for i in range(6):
        for j in range(5):
            dictionary[chars[combination[counter]]] = (i, j-2)
            counter += 1
    return dictionary

def evaluate(dictionary, string):
    current_set = 0
    total_steps = 0
    for char in list(string.upper()):
        set = dictionary[char][0]
        pos = dictionary[char][1]
        if set == current_set:
            total_steps += abs(pos) + 2
        else:
            total_steps += min(abs(set - current_set), 6 - abs(set - current_set)) + 1
            current_set = set
            total_steps += abs(pos) + 1
    return total_steps

class Instance(object):
    def __init__(self, chars, string, combination = None, fitness = None):
        self.chars = chars
        if combination is None:
            self.combination = random.sample(range(len(chars)), len(chars))
        else:
            self.combination = list(combination)
        self.dictionary = dict_init(chars, self.combination)
        if fitness is None:
            self.fitness = evaluate(self.dictionary, string)
        else:
            self.fitness = fitness

    def calculate(self, string):
        self.fitness = evaluate(self.dictionary, string)
