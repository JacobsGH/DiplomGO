import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time

def main():
    start_time = time.time()
    
    # Параметры
    input_file = r'C:\Users\JacobsPC\Desktop\VacationClasstered\Omic.csv'
    output_dir = r'C:\Users\JacobsPC\Desktop\VacationClasstered\iterative_clustering'
    n_start = 2000  # Начальное число кластеров
    n_min = 148     # Минимальное число кластеров
    n_deductible = 100  # Шаг уменьшения числа кластеров
    max_stagnation = 3  # Максимальное число итераций без изменений
    kmeans_trials = 3   # Количество попыток кластеризации для стабильности

    # Создание директорий для сохранения результатов
    os.makedirs(output_dir, exist_ok=True)
    removed_dir = os.path.join(output_dir, 'removed_clusters')
    final_dir = os.path.join(output_dir, 'final_clusters')
    os.makedirs(removed_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Загрузка данных
    df = pd.read_csv(input_file)
    gene_names = df.iloc[:, 0]  # Первый столбец — названия генов
    data = df.iloc[:, 1:].values.astype(np.float32)  # Данные (гены x образцы)

    # Добавление синтетического образца X с нулевыми значениями
    data = np.hstack([data, np.zeros((data.shape[0], 1))])  # Добавляем новый столбец (образец)
    sample_names = list(df.columns[1:]) + ['X']  # Обновляем имена образцов

    # Инициализация
    current_data = data.copy()
    current_sample_names = sample_names.copy()
    removed_clusters = {}  # Словарь для хранения удаленных образцов по итерациям
    iteration = 0
    stagnation_counter = 0
    n_current = n_start

    # Основной цикл
    while n_current >= n_min and stagnation_counter < max_stagnation:
        iteration += 1
        
        # Адаптация числа кластеров
        n_clusters = min(n_current, current_data.shape[1] - 1)  # Не больше, чем число образцов минус X
        if n_clusters < 1: break

        # Многократная кластеризация для стабильности
        best_kmeans = None
        best_inertia = float('inf')
        for _ in range(kmeans_trials):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(current_data.T)  # Кластеризация по образцам (столбцам)
            if kmeans.inertia_ < best_inertia:
                best_kmeans = kmeans
                best_inertia = kmeans.inertia_

        # Поиск кластера с X
        x_index = current_sample_names.index('X')
        x_cluster = best_kmeans.labels_[x_index]
        cluster_mask = (best_kmeans.labels_ == x_cluster)
        
        # Фильтрация образцов
        if sum(cluster_mask) == 1:
            stagnation_counter += 1
            n_current = max(n_min, n_current - n_deductible)
        else:
            # Сохранение удаленных образцов
            removed_samples = [current_sample_names[i] for i in np.where(cluster_mask)[0] if current_sample_names[i] != 'X']
            removed_clusters[iteration] = removed_samples
            
            # Обновление данных
            keep_mask = ~cluster_mask | (np.array(current_sample_names) == 'X')
            current_data = current_data[:, keep_mask]
            current_sample_names = [current_sample_names[i] for i in np.where(keep_mask)[0]]
            stagnation_counter = 0

    # Финальная кластеризация
    final_samples_mask = (np.array(current_sample_names) != 'X')
    final_data = current_data[:, final_samples_mask]
    final_sample_names = [current_sample_names[i] for i in np.where(final_samples_mask)[0]]

    if final_data.shape[1] > 0:
        kmeans_final = KMeans(n_clusters=min(n_current, final_data.shape[1]), random_state=42)
        final_labels = kmeans_final.fit_predict(final_data.T)  # Кластеризация по образцам

        # Сохранение финальных кластеров
        for cluster_id in np.unique(final_labels):
            cluster_samples = [final_sample_names[i] for i in np.where(final_labels == cluster_id)[0]]
            pd.DataFrame({'Sample': cluster_samples}).to_csv(
                os.path.join(final_dir, f'cluster_{cluster_id}.csv'), index=False)

    # Сохранение удаленных кластеров
    for iter_num, samples in removed_clusters.items():
        pd.DataFrame({'Sample': samples}).to_csv(
            os.path.join(removed_dir, f'iteration_{iter_num}.csv'), index=False)

    print(f"Время выполнения: {time.time() - start_time:.2f} сек")

if __name__ == "__main__":
    main()