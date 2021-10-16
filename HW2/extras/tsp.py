import numpy as np 
import mlrose_hiive
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, datasets
import time
import random
import warnings

np.random.seed(50)

# def return_max(input):
#     return max(input)

# def generate_edges(num_edges,max_value):
# 	edges = []
# 	while len(edges) < num_edges:
# 		node_1 = np.random.randint(max_value)
# 		node_2 = np.random.randint(max_value)
# 		if node_1 == node_2:
# 			continue
# 		edges.append((node_1, node_2))
# 		edges = list(set(edges))
# 	edges.sort(key = return_max, reverse = False)
# 	return edges

# edges = generate_edges(100,16)
# print(edges)

dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), \
             (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), \
             (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), \
             (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), \
             (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
             (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), \
             (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]

coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

coords_list = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4),(1, 2), (2, 4), (1, 5), (1, 6), (1, 4), (3, 4), (4, 5), (2, 3), (2, 4), (2, 6), (3, 5), (4, 2), (4, 5), (5, 6), (7, 6), (2, 7), (4, 1), (4, 1)]

fitness_simulated_annealing = []
fitness_random_hill_climb = []
fitness_genetic_algorithm = []
fitness_mimic = []

time_simulated_annealing = []
time_random_hill_climb = []
time_genetic_algorithm = []
time_mimic = []

## Plot effect of increasing problem size

# range_values = range(4,8,1)

# for value in range_values:
# 	fitness = mlrose_hiive.TravellingSales(coords = coords_list[:value])
# 	problem = mlrose_hiive.TSPOpt(length = value, fitness_fn = fitness, maximize = False)
# 	problem.set_mimic_fast_mode(True)
# 	init_state = np.random.choice(value, size = value, replace = False)
# 	start = time.time()
# 	_, best_fitness_sa, _ = mlrose_hiive.simulated_annealing(problem, schedule = mlrose_hiive.ExpDecay(), max_attempts = 100, max_iters = 1000, init_state = init_state, curve = True)
# 	end = time.time()
# 	sa_time = end - start
# 	print("SA:", sa_time, value)

# 	start = time.time()
# 	_, best_fitness_rhc, _ = mlrose_hiive.random_hill_climb(problem, max_attempts = 100, max_iters = 1000, init_state = init_state, curve = True)
# 	end = time.time()
# 	rhc_time = end - start
# 	print("RHC:", rhc_time, value)

# 	start = time.time()
# 	_, best_fitness_ga, _ = mlrose_hiive.genetic_alg(problem, max_attempts = 100, max_iters = 1000, curve = True)
# 	end = time.time()
# 	ga_time = end - start
# 	print("GA:", ga_time, value)

# 	start = time.time()
# 	_, best_fitness_mimic, _ = mlrose_hiive.mimic(problem, pop_size = 200, max_attempts = 100, max_iters = 1000, curve = True)
# 	end = time.time()
# 	mimic_time = end - start
# 	print("MIMIC:", mimic_time, value)

# 	fitness_simulated_annealing.append(best_fitness_sa)
# 	fitness_random_hill_climb.append(best_fitness_rhc)
# 	fitness_genetic_algorithm.append(best_fitness_ga)
# 	fitness_mimic.append(best_fitness_mimic)

# 	time_simulated_annealing.append(sa_time)
# 	time_random_hill_climb.append(rhc_time)
# 	time_genetic_algorithm.append(ga_time)
# 	time_mimic.append(mimic_time)

# fitness_simulated_annealing = np.array(fitness_simulated_annealing)
# fitness_random_hill_climb = np.array(fitness_random_hill_climb)
# fitness_genetic_algorithm = np.array(fitness_genetic_algorithm)
# fitness_mimic = np.array(fitness_mimic)

# time_simulated_annealing = np.array(time_simulated_annealing)
# time_random_hill_climb = np.array(time_random_hill_climb)
# time_genetic_algorithm = np.array(time_genetic_algorithm)
# time_mimic = np.array(time_mimic)

# plt.figure()
# plt.plot(range_values, fitness_simulated_annealing, label = 'Simulated Annealing')
# plt.plot(range_values, fitness_random_hill_climb, label = 'Randomized Hill Climb')
# plt.plot(range_values, fitness_genetic_algorithm, label = 'Genetic Algorithm')
# plt.plot(range_values, fitness_mimic, label = 'MIMIC')
# plt.title('Fitness vs. Problem Size (TSP)')
# plt.xlabel('Problem Size')
# plt.ylabel('Fitness')
# plt.legend()
# plt.show()
# plt.savefig('tsp_fitness.png')

# plt.figure()
# plt.plot(range_values, time_simulated_annealing, label = 'Simulated Annealing')
# plt.plot(range_values, time_random_hill_climb, label = 'Randomized Hill Climb')
# plt.plot(range_values, time_genetic_algorithm, label = 'Genetic Algorithm')
# plt.plot(range_values, time_mimic, label = 'MIMIC')
# plt.title('Time Efficiency vs. Problem Size (TSP)')
# plt.legend()
# plt.xlabel('Problem Size')
# plt.ylabel('Computation Time (s)')
# plt.savefig('tsp_computation.png')

## Plot change with respect to iterations

problem_length = 20
fitness = mlrose_hiive.TravellingSales(coords = coords_list[:problem_length])
problem = mlrose_hiive.TSPOpt(length = problem_length, fitness_fn = fitness, maximize = False)
problem.set_mimic_fast_mode(True)
init_state = np.random.choice(problem_length, size = problem_length, replace = False)
_, _, fitness_curve_sa = mlrose_hiive.simulated_annealing(problem, init_state = init_state, curve = True)
print("Done with SA iterations!")
_, _, fitness_curve_rhc = mlrose_hiive.random_hill_climb(problem, init_state = init_state, curve = True)
print("Done with RHC iterations!")
_, _, fitness_curve_ga = mlrose_hiive.genetic_alg(problem, curve = True)
print("Done with GA iterations!")
_, _, fitness_curve_mimic = mlrose_hiive.mimic(problem, pop_size = 1000, curve = True)
print("Done with MIMIC iterations!")

plt.figure()
plt.plot(fitness_curve_sa[:,0], label = 'Simulated Annealing')
plt.plot(fitness_curve_rhc[:,0], label = 'Randomized Hill Climb')
plt.plot(fitness_curve_ga[:,0], label = 'Genetic Algorithm')
plt.plot(fitness_curve_mimic[:,0], label = 'MIMIC')
plt.title('Fitness Curve (TSP)')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.savefig('tsp_iterations.png')
