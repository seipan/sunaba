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
    for num_workers in range(1, 15):
        start = 2
        end = 100000
        num_workers = num_workers

        chunk_size = (end - start) // num_workers
        ranges = [(start + i * chunk_size, start + (i + 1) * chunk_size - 1) for i in range(num_workers)]
        ranges[-1] = (ranges[-1][0], end)  

        start_time = time.time()

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(find_primes_in_range, ranges)

        primes = [prime for sublist in results for prime in sublist]

        end_time = time.time()

        execution_times.append(end_time - start_time)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 15), execution_times, marker='o')
    plt.title("Execution Time vs Number of Workers (Range: 2 to 10,000)")
    plt.xlabel("Number of Workers")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.show()
