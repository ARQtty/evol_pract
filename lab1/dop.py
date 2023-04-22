import random
from pathlib import Path
import math
from statistics import mean
from tqdm import tqdm


class Chromosome:
    def __init__(self, bits: list[int]):
        self._bits = bits

    def get_bits(self) -> list[int]:
        return self._bits

    @staticmethod
    def number_to_bits(a: int) -> list[int]:
        return list(map(int, str(bin(a))[2:]))

    @staticmethod
    def bits_to_number(a: list[int]) -> int:
        return int('0b' + ''.join(list(map(str, a))), 2)

    @classmethod
    def from_random(cls, n_genes: int):
        return cls([round(random.random()) for _ in range(n_genes)])


class Individual:
    def __init__(self, chromosome: Chromosome):
        self._chromosome = chromosome

    def get_x(self) -> float:
        # 0..2^N_GENES = 2^17
        x = self._chromosome.bits_to_number(self._chromosome.get_bits())

        # to 0..17
        x = math.log2(x)

        # to -20..-3
        x -= 20
        return x

    def fitness(self) -> float:
        x = self.get_x()
        fitness = math.sin(2*x) / x**2
        return fitness

    def mutate(self, g1: Chromosome, g2: Chromosome) -> Chromosome:
        g1 = g1.get_bits()
        g2 = g2.get_bits()

        new_g = list(g1) if random.random() > 0.5 else list(g2)
        if random.random() < MUTATION_PROB:
            gene_ix = random.randint(0, len(g1) - 1)

            for i in range(gene_ix, len(g1)):
                new_g[i] = g1[i] if random.random() > 0.5 else g2[i]
        return Chromosome(new_g)


def main():
    random.seed(27)
    population: list[Individual] = [Individual(Chromosome.from_random(N_GENES)) for _ in range(POPULATION_SIZE)]
    fitnesses = []

    for epoch in tqdm(range(20)):

        sum_fitness = sum([ind.fitness() for ind in population])
        probs = [ind.fitness() / sum_fitness for ind in population]

        for ind in population[:3]:
            print(ind.get_x(), ind.fitness())
        print()

        for i in range(POPULATION_SIZE):
            p1, p2 = random.choices(population[:POPULATION_SIZE], weights=probs, k=2)
            new_chromosome = p1.mutate(p1._chromosome, p2._chromosome)
            population.append(Individual(new_chromosome))
        population = population[-POPULATION_SIZE:]

        fitnesses.append(mean([ind.fitness() for ind in population]))

        if epoch % 5 == 0:
            plt.Figure()

            x = np.arange(-20, -3.1, 0.01)
            y = np.sin(2*x)/x**2
            plt.plot(x, y)

            x = np.array([ind.get_x() for ind in population])
            plt.scatter(x, np.sin(2*x)/x**2, s=10, c='red')
            plt.title(f'итерация {epoch}')

            path = Path(f'pics/convergence_mutation_{MUTATION_PROB}')
            path.mkdir(exist_ok=True, parents=True)
            plt.savefig(path / f'epoch_{epoch}.png')
            plt.show()


    return fitnesses


if __name__ == '__main__':
    import numpy as np
    # import itertools
    #
    # for arg in range(1, 100):
    #     MUTATION_PROB = arg / 100
    #
    import matplotlib.pyplot as plt

    MUTATION_PROB = 0.9
    N_GENES = 17
    POPULATION_SIZE = 40
    for MUTATION_PROB in range(10, 100, 25):
        MUTATION_PROB /= 100
        main()

    # plt.Figure()
    # plt.plot(main())
    # plt.show()
    #
    # x = np.arange(-20, -3.1, 0.01)
    # y = np.sin(2*x)/x**2
    #
    # plt.Figure()
    # plt.plot(x, y)
    # plt.show()
