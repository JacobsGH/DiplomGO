import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import time

def main():
    start_time = time.time()
    
    # Параметры
    input_file = r'.csv'
    gene_props_file = r'.csv'
    output_dir = r'\iterative_clustering'
    name_center_mass = r'cluster_centers.csv'
    n_start = 150      # Начальное число кластеров
    n_min = 2000       # Конечное число кластеров (максимальное при n_deductible < 0)
    n_deductible = 100  # Шаг изменения кластеров (-100 = +100 за итерацию)
    kmeans_trials = 3    # Количество попыток кластеризации
    similarity_threshold = 0.5  # Порог однородности кластера
    min_clusters_above_threshold = 0.3  # Минимум кластеров выше порога

    # Создание директорий
    os.makedirs(output_dir, exist_ok=True)
    removed_dir = os.path.join(output_dir, 'removed_clusters')
    final_dir = os.path.join(output_dir, 'final_clusters')
    stats_dir = os.path.join(output_dir, 'cluster_stats')
    os.makedirs(removed_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # Загрузка данных
    df_omic = pd.read_csv(input_file)
    first_column = df_omic.iloc[:, 0]  # Сохраняем первый столбец как Series
    gene_names = df_omic.columns[1:].tolist()  # Названия генов
    data = df_omic.iloc[:, 1:].values.astype(np.float32)  # Данные (гены x образцы)
    
    # Добавление синтетического образца X
    data = np.hstack([data, np.zeros((data.shape[0], 1))])  # Новый столбец
    gene_names.append('X')

    # Загрузка свойств генов
    gene_props = pd.read_csv(gene_props_file, index_col=0)
    gene_props = gene_props.loc[gene_names[:-1]]  # Исключаем X
    scaler = MinMaxScaler()
    gene_props_normalized = scaler.fit_transform(gene_props)

    # Инициализация
    current_data = data.T.copy()  # Транспонируем для работы с образцами
    current_genes = gene_names.copy()
    removed_clusters = {}
    stats = []
    iteration = 0
    best_state = None
    n_current = n_start

    while n_current<n_min:
        iteration += 1
        print(f"Итерация {iteration}: n_clusters = {n_current}")
        
        # Кластеризация с несколькими попытками
        best_kmeans = None
        best_inertia = float('inf')
        for _ in range(kmeans_trials):
            kmeans = KMeans(n_clusters=n_current, random_state=42)
            labels = kmeans.fit_predict(current_data)
            if kmeans.inertia_ < best_inertia:
                best_kmeans = kmeans
                best_inertia = kmeans.inertia_
        
        # Поиск кластера с X
        x_index = current_genes.index('X')
        x_cluster = best_kmeans.labels_[x_index]
        cluster_mask = (best_kmeans.labels_ == x_cluster)
        
        # Удаление кластера с X (кроме самого X)
        removed_genes = [current_genes[i] for i in np.where(cluster_mask)[0] if current_genes[i] != 'X']
        if removed_genes:
            removed_clusters[iteration] = removed_genes
        
        # Обновление данных
        keep_mask = [name != 'X' and not cluster_mask[i] for i, name in enumerate(current_genes)]
        keep_mask[x_index] = True  # Сохраняем X
        current_data = current_data[keep_mask, :]
        current_genes = [current_genes[i] for i in np.where(keep_mask)[0]]

        # Оценка однородности
        valid_genes = [g for g in current_genes if g != 'X']
        if not valid_genes:
            break
            
        valid_indices = [i for i, name in enumerate(current_genes) if name != 'X']
        valid_props = gene_props_normalized[[gene_names.index(g) for g in valid_genes]]
        
        # Кластеризация для оценки
        kmeans_eval = KMeans(
            n_clusters=min(n_current, len(valid_genes)), 
            n_init=kmeans_trials
        )
        eval_labels = kmeans_eval.fit_predict(current_data[valid_indices])
        
        # Расчет однородности
        homogeneity_scores = []
        for cluster_id in np.unique(eval_labels):
            cluster_props = valid_props[eval_labels == cluster_id]
            if len(cluster_props) > 1:
                homogeneity = 1 - pairwise_distances(cluster_props).mean()
            else:
                homogeneity = 1.0
            homogeneity_scores.append(homogeneity)
        
        # Сохранение статистики
        mean_homog = np.mean(homogeneity_scores)
        percent_above = np.mean(np.array(homogeneity_scores) >= similarity_threshold)
        stats.append({
            'iteration': iteration,
            'n_clusters': n_current,
            'genes_remaining': len(valid_genes),
            'mean_homogeneity': mean_homog,
            'percent_above': percent_above
        })

        # Проверка останова
        if percent_above >= min_clusters_above_threshold:
            best_state = {
                'data': current_data[valid_indices],
                'genes': valid_genes,
                'iteration': iteration
            }
            break

        # Изменение числа кластеров
        n_current += n_deductible

    # Сохранение результатов
    pd.DataFrame(stats).to_csv(os.path.join(stats_dir, 'homogeneity_report.csv'), index=False)
    
    # Сохранение удаленных кластеров
    for iter_num, genes in removed_clusters.items():
        pd.DataFrame({'Gene': genes}).to_csv(
            os.path.join(removed_dir, f'iteration_{iter_num}.csv'), index=False)
    
    if best_state:
        final_genes = best_state['genes']
        try:
            # Получаем индексы генов, проверяя их наличие
            gene_indices = [gene_names.index(g) for g in final_genes if g in gene_names]
            final_data = gene_props_normalized[gene_indices]
        
            # Используем n_current, на котором были достигнуты параметры
            n_clusters_final = n_current
            print(f"Финальное число кластеров: {n_clusters_final}")

            kmeans_final = KMeans(n_clusters=n_clusters_final, n_init=kmeans_trials)
            final_labels = kmeans_final.fit_predict(final_data)
        
            # Сохраняем кластеры
            for cluster_id in np.unique(final_labels):
                cluster_genes = [final_genes[i] for i in np.where(final_labels == cluster_id)[0]]
                path = os.path.join(final_dir, f'cluster_{cluster_id}.csv')
                pd.DataFrame({'Gene': cluster_genes}).to_csv(path, index=False)
            
            # Создание DataFrame для центров масс кластеров и транспонирование
            centers_df = pd.DataFrame(kmeans.cluster_centers_.T, 
                          columns=[f'Cluster {i}' for i in range(n_current)])

            # Вставка первого столбца с названиями строк (первый столбец из исходного DataFrame)
            centers_df.insert(0, 'Row Names', first_column)

            # Сохранение центров масс в отдельный CSV файл
            centers_file = os.path.join(output_dir, name_center_mass)
            centers_df.to_csv(centers_file)
            print("Центры масс кластеров сохранены"
            
        except ValueError as e:
            print(f"Ошибка: {e}. Невозможно создать кластеры.")
        except Exception as e:
            print(f"Критическая ошибка: {e}")

    print(f"Выполнено итераций: {iteration}")
    print(f"Время выполнения: {time.time() - start_time:.2f} сек")

if __name__ == "__main__":
    main()