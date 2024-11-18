import random
import time
import matplotlib.pyplot as plt
from deap import base, creator, tools
import multiprocessing

# Define necessary classes and functions
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Fitness function (evaluation function)
def onemax(individual):
    start_time = time.perf_counter()
    time.sleep(toolbox.sleep_duration)
    fitness = sum(individual)
    end_time = time.perf_counter()
    eval_time = end_time - start_time
    return fitness, eval_time  # Return fitness and evaluation time

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
IND_SIZE = 100
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

toolbox.register("evaluate", onemax)

# Parallel evaluation function
def evaluate_range(individuals):
    total_eval_time = 0
    results = []
    for individual in individuals:
        fitness, eval_time = toolbox.evaluate(individual)
        total_eval_time += eval_time
        results.append((fitness,))
    return results, total_eval_time

def parallel_evaluate(pop, num_workers):
    overhead_time = 0

    # Measure process creation time
    process_creation_start = time.perf_counter()
    pool = multiprocessing.Pool(processes=num_workers)
    process_creation_end = time.perf_counter()
    process_creation_time = process_creation_end - process_creation_start
    overhead_time += process_creation_time

    # Divide the population among workers
    chunk_size = len(pop) // num_workers
    chunks = [pop[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
    if len(pop) % num_workers != 0:
        chunks[-1].extend(pop[num_workers * chunk_size:])

    # Measure task submission time
    task_submission_start = time.perf_counter()
    async_result = pool.map_async(evaluate_range, chunks)
    task_submission_end = time.perf_counter()
    task_submission_time = task_submission_end - task_submission_start
    overhead_time += task_submission_time

    # Close the pool to prevent new tasks
    pool.close()

    # Wait for all tasks to complete (do not measure this time)
    async_result.wait()

    # Get results (do not measure this time)
    results = async_result.get()

    # Assign fitness to individuals
    fitness_times = []
    idx = 0
    for res, eval_time in results:
        for fit in res:
            pop[idx].fitness.values = fit
            idx += 1
        fitness_times.append(eval_time)

    # Maximum fitness function execution time among workers
    total_onemax_time = max(fitness_times)

    # Overhead time is the sum of process creation time and task submission time
    return total_onemax_time, overhead_time

def main():
    # Parameter settings
    random.seed(64)
    NGEN = 100   # Number of generations
    POP = 50     # Population size
    CXPB = 0.7   # Crossover probability
    MUTPB = 0.2  # Mutation probability

    num_workers = 4  # Fix the number of workers to 4
    sleep_duration_list = [0.0, 0.000005, 0.00001, 0.00005,
                           0.0001, 0.0002]  # Various sleep durations

    total_execution_times = []
    total_fitness_times = []
    total_overhead_times = []
    total_other_ga_times = []

    print(f"\nRunning GA with the number of workers fixed at {num_workers}")
    for sleep_duration in sleep_duration_list:
        toolbox.sleep_duration = sleep_duration  # Set sleep duration
        total_fitness_time = 0  # Total execution time of the fitness function
        total_overhead_time = 0  # Total overhead time
        total_other_ga_time = 0  # Total execution time of other GA operations

        start_time = time.perf_counter()
        pop = toolbox.population(n=POP)

        for gen in range(NGEN):
            # Evaluate the fitness function
            total_onemax_time, overhead_time = parallel_evaluate(pop, num_workers)
            total_fitness_time += total_onemax_time
            total_overhead_time += overhead_time

            # Other GA operations (selection, crossover, mutation)
            other_ga_start = time.perf_counter()

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            other_ga_end = time.perf_counter()
            other_ga_time = other_ga_end - other_ga_start
            total_other_ga_time += other_ga_time

            # Evaluate individuals that need evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                total_onemax_time, overhead_time = parallel_evaluate(invalid_ind, num_workers)
                total_fitness_time += total_onemax_time
                total_overhead_time += overhead_time

            pop[:] = offspring

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        total_execution_times.append(execution_time)
        total_fitness_times.append(total_fitness_time)
        total_overhead_times.append(total_overhead_time)
        total_other_ga_times.append(total_other_ga_time)

        print(f"Sleep duration: {sleep_duration}, Total GA execution time: {execution_time:.6f} seconds, "
              f"Total fitness function time: {total_fitness_time:.6f} seconds, "
              f"Total overhead time: {total_overhead_time:.6f} seconds, "
              f"Total other GA operations time: {total_other_ga_time:.6f} seconds")

    # Plot the stacked bar chart
    plt.figure(figsize=(10, 6))

    # Since the sleep durations are very close in value, use indices as x-axis positions
    x_positions = range(len(sleep_duration_list))

    # Calculate cumulative times
    fitness_times = total_fitness_times
    overhead_times = total_overhead_times
    other_ga_times = total_other_ga_times

    bottom_fitness = [0] * len(fitness_times)
    bottom_overhead = [sum(x) for x in zip(fitness_times)]
    bottom_other_ga = [sum(x) for x in zip(fitness_times, overhead_times)]

    bar_width = 0.9  # Make the bars wider to connect them

    # Plot stacked bar chart
    plt.bar(x_positions, fitness_times, width=bar_width, color='skyblue', label='Fitness Function Time')
    plt.bar(x_positions, overhead_times, width=bar_width, bottom=fitness_times, color='orange', label='Overhead Time')
    plt.bar(x_positions, other_ga_times, width=bar_width, bottom=bottom_other_ga, color='green', label='Other GA Operations Time')

    # Set x-axis labels to sleep durations
    plt.xticks(x_positions, [f"{sd}" for sd in sleep_duration_list])

    plt.title("Execution Time Breakdown by Sleep Duration")
    plt.xlabel("Sleep Duration (seconds)")
    plt.ylabel("Total Execution Time (seconds)")
    plt.legend()
    plt.grid(True)

    # Show the first plot
    plt.show()

    # Plot the line graph for total GA execution time and fitness function time
    plt.figure(figsize=(10, 6))

    plt.plot(sleep_duration_list, total_execution_times, marker='o', label='Total GA Execution Time')
    plt.plot(sleep_duration_list, total_fitness_times, marker='s', label='Total Fitness Function Time')

    plt.title("Total GA Execution Time and Fitness Function Time vs Sleep Duration")
    plt.xlabel("Sleep Duration (seconds)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)

    # Show the second plot
    plt.show()

if __name__ == "__main__":
    main()
