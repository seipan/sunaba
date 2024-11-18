import pygmo as pg
import numpy as np
import time
import matplotlib.pyplot as plt

# OneMax問題を定義するクラス（遅延追加）
class OneMaxProblem:
    def __init__(self, genome_length):
        self.genome_length = genome_length
        self.bounds = ([0]*genome_length, [1]*genome_length)

    def fitness(self, x):
        time.sleep(0.001)  # 評価関数に遅延を追加（例：0.01秒）
        return [-sum(x)]  # 最小化問題として定義するため、負の値を返す

    def get_bounds(self):
        return self.bounds

    def get_nobj(self):
        return 1  # 目的関数の数

    def gradient(self, x):
        return pg.estimate_gradient(self.fitness, x)

def run_experiment(island_counts):
    genome_length = 100
    population_size = 100
    generations = 50

    times = []

    for num_islands in island_counts:
        start_time = time.time()

        # 問題の定義
        prob = pg.problem(OneMaxProblem(genome_length))

        # アルゴリズムの定義（遺伝的アルゴリズム）
        algo = pg.algorithm(pg.sga(gen=generations, cr=0.9, m=0.02))

        # アーチペラゴ（群島）の作成
        archi = pg.archipelago(n=num_islands, algo=algo, prob=prob, pop_size=population_size)

        # 進化の実行
        archi.evolve()
        archi.wait_check()  # すべての島の進化が完了するまで待機

        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        print(f"島の数: {num_islands}, 実行時間: {elapsed_time:.2f} 秒")

    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(island_counts, times, marker='o')
    plt.xlabel('島の数')
    plt.ylabel('実行時間（秒）')
    plt.title('島の数とGAの実行時間')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    island_counts = [1, 2, 4, 8]
    run_experiment(island_counts)
