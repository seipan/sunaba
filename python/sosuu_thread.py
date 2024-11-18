import multiprocessing
import time
import matplotlib.pyplot as plt

def trial(n: int):
    assert n > 1
    i = 2
    while i < n:
        if n % i == 0:
            return False
        i += 1
    return True

def find_primes_in_range(start_end):
    start, end = start_end
    return [n for n in range(start, end + 1) if trial(n)]

if __name__ == '__main__':
    execution_times = []
    num_workers_list = []
    for num_workers in range(1, 25):
        start = 2
        end = 100000

        chunk_size = (end - start) // num_workers
        ranges = [(start + i * chunk_size, start + (i + 1) * chunk_size - 1) for i in range(num_workers)]
        ranges[-1] = (ranges[-1][0], end)  

        start_time = time.time()

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(find_primes_in_range, ranges)

        primes = [prime for sublist in results for prime in sublist]

        end_time = time.time()

        execution_times.append(end_time - start_time)
        num_workers_list.append(num_workers)

    baseline_time = execution_times[0]
    normalized_times = [t / baseline_time for t in execution_times]

    plt.figure(figsize=(1, 25))
    plt.plot(num_workers_list, normalized_times, marker='o')
    plt.title("Relative Execution Time vs Number of Workers (Range: 2 to 100,000)")
    plt.xlabel("Number of Workers")
    plt.ylabel("Relative Execution Time (Normalized to 1 Worker)")
    plt.grid(True)
    plt.show()
