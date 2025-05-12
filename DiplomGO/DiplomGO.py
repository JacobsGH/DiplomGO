# Улучшенная программа кластеризации генов по экспрессии и GO-аннотациям
# Пятая итерация программы, на момент 12.05. 8:50 она лучшая.
# Программа кластеризует гены на основе экспрессии генов и их GO
import os
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

try:
    from goatools import obo_parser
    from goatools.associations import read_gaf
    from goatools.semantic import TermCounts, resnik_sim
    GO_AVAILABLE = True
except ImportError:
    GO_AVAILABLE = False
    print("Библиотека goatools не установлена, будут использоваться только данные экспрессии")


#Класс-Функция для сохранения логов программы, по возможности можно дополнить новой отладочной информацией
class ClusterLogger:
    """Класс для логирования выполнения программы"""
    def __init__(self, output_dir):
        self.logger = logging.getLogger('GeneClustering')
        self.logger.setLevel(logging.INFO)
        
        log_file = os.path.join(output_dir, f'clustering_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
    def log_params(self, params):
        """Логирование параметров запуска"""
        self.logger.info("ПАРАМЕТРЫ ЗАПУСКА:")
        for key, value in params.items():
            self.logger.info(f"{key}: {value}")
            
    def log_iteration(self, iteration_info):
        """Логирование информации о итерации"""
        self.logger.info(f"Итерация {iteration_info['number']}:")
        self.logger.info(f"Количество кластеров: {iteration_info['n_clusters']}")
        self.logger.info(f"Индекс инерции: {iteration_info['inertia']:.2f}")
        self.logger.info(f"Удалено генов: {iteration_info['removed_genes']}")
        self.logger.info(f"Осталось генов: {iteration_info['remaining_genes']}")
        
    def log_final_stats(self, stats):
        """Логирование финальной статистики"""
        self.logger.info("\nФИНАЛЬНАЯ СТАТИСТИКА:")
        self.logger.info(f"Всего кластеров: {stats['total_clusters']}")
        self.logger.info(f"Всего удалено генов: {stats['total_removed']}")
        self.logger.info(f"Время выполнения: {stats['execution_time']:.2f} сек")

class GOClusterAnalyzer:
    """Класс для анализа GO-аннотаций и кластеризации генов"""
    
    def __init__(self, go_obo_file=None, go_gaf_file=None):
        """Инициализация с загрузкой GO-онтологии и аннотаций"""
        self.go_dag = None
        self.term_counts = None
        self.gene_go = defaultdict(set)
        self.go_aspects = {}
        
        if GO_AVAILABLE and go_obo_file and go_gaf_file:
            self._load_go_data(go_obo_file, go_gaf_file)
    
    def _load_go_data(self, go_obo_file, go_gaf_file):
        """Загрузка данных GO с обработкой UniProtKB формата"""
        print("Загрузка онтологии GO...")
        try:
            self.go_dag = obo_parser.GODag(go_obo_file)
            
            print("Загрузка GO-аннотаций...")
            self.gene_go = defaultdict(set)
            self.go_aspects = {}
            
            # Вручную парсим GAF файл
            with open(go_gaf_file, 'r') as f:
                for line in f:
                    if line.startswith('!'):  # Пропускаем комментарии
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) < 15:  # Минимальное количество полей в GAF 2.2
                        continue
                    
                    # Извлекаем необходимые поля
                    db = parts[0]          # e.g. UniProtKB
                    db_object_id = parts[1] # e.g. A0A024RBG1
                    symbol = parts[2]       # e.g. NUDT4B
                    go_id = parts[4]        # e.g. GO:0003723
                    
                    # Используем символ гена как основной идентификатор
                    gene_symbol = symbol.upper()
                    
                    # Добавляем аннотацию
                    self.gene_go[gene_symbol].add(go_id)
                    
                    # Сохраняем аспект термина
                    if go_id in self.go_dag:
                        self.go_aspects[go_id] = self.go_dag[go_id].namespace
            
            # Создаем TermCounts (необходимо для семантических мер)
            # Создаем временный словарь ассоциаций для TermCounts
            temp_assoc = defaultdict(set)
            for gene, terms in self.gene_go.items():
                for term in terms:
                    temp_assoc[gene].add(term)
            
            self.term_counts = TermCounts(self.go_dag, temp_assoc)
            
            print(f"\nУспешно загружено:")
            print(f"- Уникальных генов: {len(self.gene_go)}")
            print(f"- GO-терминов: {sum(len(terms) for terms in self.gene_go.values())}")
            
            # Проверка аннотаций для тестовых генов
            test_genes = ['ZNF891', 'ARMC10', 'PTGER4', 'EIF1AD', 'ABCG5', 'CXCR4', 'CAPNS1']
            print("\nПроверка аннотаций для тестовых генов:")
            for gene in test_genes:
                std_gene = self.standardize_gene_name(gene)
                has_anno = std_gene in self.gene_go
                print(f"{gene} -> {std_gene}: {'ЕСТЬ' if has_anno else 'НЕТ'} аннотаций")
                
        except Exception as e:
            print(f"\nОшибка загрузки GO-аннотаций: {e}")
            import traceback
            traceback.print_exc()
            self.go_dag = None
    
    def _print_go_stats(self):
        """Вывод статистики по загруженным GO-аннотациям"""
        if not self.gene_go:
            return
            
        print("\nСтатистика GO-аннотаций:")
        print(f"Всего генов с аннотациями: {len(self.gene_go)}")
        
        # Распределение по аспектам
        aspect_counts = {'biological_process': 0, 'molecular_function': 0, 'cellular_component': 0}
        for terms in self.gene_go.values():
            for term in terms:
                aspect = self.go_aspects.get(term)
                if aspect in aspect_counts:
                    aspect_counts[aspect] += 1
        
        for aspect, count in aspect_counts.items():
            print(f"{aspect}: {count} терминов")
        
        # Среднее количество аннотаций на ген
        avg_terms = sum(len(terms) for terms in self.gene_go.values()) / len(self.gene_go)
        print(f"Среднее количество терминов на ген: {avg_terms:.1f}")
    
    def standardize_gene_name(self, gene_name):
        """Улучшенная стандартизация имени гена из файла экспрессии"""
        # Извлекаем символ гена (часть до скобки)
        base_name = re.split(r'\s*\(|\)', gene_name)[0].upper()
        
        # Удаляем возможные версии и дополнительные обозначения
        base_name = re.sub(r'\.\d+$', '', base_name)
        
        # Удаляем возможные суффиксы типа -AS1, -DT и т.д.
        base_name = re.sub(r'-\w+$', '', base_name)
        
        return base_name
    
    def get_go_vector(self, gene, aspect=None):
        """Получение вектора GO-терминов для гена"""
        standardized_gene = self.standardize_gene_name(gene)
        
        if standardized_gene not in self.gene_go or not self.go_dag:
            return []
        
        terms = self.gene_go[standardized_gene]
        if aspect:
            terms = [t for t in terms if self.go_aspects.get(t) == aspect]
        
        # Создаем более информативный вектор
        vector = []
        for term in terms:
            if term in self.go_dag:
                term_info = self.go_dag[term]
                # Добавляем несколько характеристик термина
                vector.extend([
                    term_info.depth,                  # Глубина в иерархии
                    len(term_info.children),           # Количество дочерних терминов
                    self.term_counts.get_count(term),   # Частота термина в аннотациях
                    int(term_info.is_obsolete)         # Является ли устаревшим
                ])
        
        return vector
    
    def prepare_combined_features(self, gene_names, expr_data):
        """Подготовка комбинированных признаков (экспрессия + GO)"""
        # Нормализация данных экспрессии
        expr_features = MinMaxScaler().fit_transform(expr_data.T)
        
        # Сбор GO-признаков
        go_features = []
        aspects = ['biological_process', 'molecular_function', 'cellular_component']
        
        for gene in gene_names:
            gene_vec = []
            for aspect in aspects:
                gene_vec.extend(self.get_go_vector(gene, aspect))
            
            # Если нет GO-аннотаций, используем нули
            if not gene_vec:
                gene_vec = [0] * 12  # 3 аспекта * 4 характеристики
            
            go_features.append(gene_vec)
        
        # Выравнивание размерности
        max_go_len = max(len(v) for v in go_features)
        go_features = [v + [0]*(max_go_len-len(v)) for v in go_features]
        
        # Объединение признаков
        combined_features = np.hstack([expr_features, np.array(go_features)])
        
        print(f"\nРазмерность признаков:")
        print(f"- Экспрессия: {expr_features.shape[1]}")
        print(f"- GO: {np.array(go_features).shape[1]}")
        print(f"- Всего: {combined_features.shape[1]}")
        
        return combined_features

def main():
    start_time = time.time()
    
    # Параметры
    input_file = r'C:\Users\Jacobs\Desktop\VacationClasstered2\Omic.csv'
    #go_obo_file = r'C:\Users\Jacobs\Desktop\VacationClasstered2\go-basic.obo'
    go_obo_file = r'C:\Users\Jacobs\Desktop\VacationClasstered2\go.obo'
    go_gaf_file = r'C:\Users\Jacobs\Desktop\go_clustering\goa_human.gaf'
    output_dir = r'C:\Users\Jacobs\Desktop\go_clustering\go_Relize_Program_clastering'
    
    # Параметры кластеризации
    n_start = 100               # Начальное число кластеров
    n_min = 1000                # Конечное число кластеров
    n_deductible = 100          # Шаг изменения кластеров
    kmeans_trials = 2           # Количество попыток кластеризации
    filter_iterations = 3       # Сколько раз выполнять фильтрацию (0 для отключения)
    
    params = {
        "input_file": input_file,
        "go_obo_file": go_obo_file,
        "go_gaf_file": go_gaf_file,
        "n_start": n_start,
        "n_min": n_min,
        "n_deductible": n_deductible,
        "kmeans_trials": kmeans_trials,
        "filter_iterations": filter_iterations
    }
    
    logger.log_params(params)
    
    # Создание директорий
    os.makedirs(output_dir, exist_ok=True)
    final_dir = os.path.join(output_dir, 'final_clusters')
    filtered_dir = os.path.join(output_dir, 'filtered_genes')
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(filtered_dir, exist_ok=True)
    
    logger = ClusterLogger(output_dir)
    
    print("1. Загрузка и подготовка данных...")
    # Загрузка данных экспрессии генов
    df_omic = pd.read_csv(input_file)
    first_column = df_omic.iloc[:, 0]
    
    # Обработка имен генов
    gene_names = [col for col in df_omic.columns[1:]]  # Сохраняем оригинальные имена для вывода
    data = df_omic.iloc[:, 1:].values.astype(np.float32)
    
    # Добавление синтетического образца X
    data = np.hstack([data, np.zeros((data.shape[0], 1))])
    gene_names.append('X')
    x_index = gene_names.index('X')
    
    print("\n2. Инициализация анализатора GO...")
    go_analyzer = GOClusterAnalyzer(go_obo_file, go_gaf_file)
    
    print("\n3. Подготовка комбинированных признаков...")
    # Создаем матрицу признаков [экспрессия + GO]
    if GO_AVAILABLE and go_analyzer.go_dag:
        features = go_analyzer.prepare_combined_features(gene_names[:-1], data[:, :-1])  # Исключаем X
    else:
        print("Используются только данные экспрессии (GO недоступен)")
        features = MinMaxScaler().fit_transform(data[:, :-1].T)
    
    # Добавляем синтетический ген X (все признаки = 0)
    features = np.vstack([features, np.zeros(features.shape[1])])
    
    print(f"\n4. Итеративная кластеризация с фильтрацией (n={n_start}-{n_min})...")
    current_features = features.copy()
    current_genes = gene_names.copy()
    removed_genes = []
    iteration = 0
    best_clusters = None
    best_score = -np.inf
    n_current = n_start
    filter_count = 0
    
    while n_current <= n_min and len(current_genes) > n_current:
        iteration += 1
        print(f"\nИтерация {iteration}: n_clusters = {n_current}")
        
        # Кластеризация K-means
        best_kmeans = None
        best_inertia = np.inf
        
        for trial in range(kmeans_trials):
            kmeans = KMeans(n_clusters=n_current, random_state=trial, n_init=10)
            labels = kmeans.fit_predict(current_features)
            
            if kmeans.inertia_ < best_inertia:
                best_kmeans = kmeans
                best_inertia = kmeans.inertia_
                current_labels = labels
        
        # Оценка кластеров
        cluster_score = -best_inertia
        print(f"Попытка {n_current}: inertia={best_inertia:.2f}, score={cluster_score:.2f}")
        
        if cluster_score > best_score:
            best_score = cluster_score
            best_clusters = current_labels
            print(f"Новый лучший результат с {n_current} кластерами!")
        
        # Фильтрация (исправленная часть)
        if filter_iterations > 0 and filter_count < filter_iterations:
            # Находим кластер с геном X (он всегда последний)
            x_cluster = current_labels[-1]
            cluster_mask = (current_labels == x_cluster)
            
            # Убедимся, что маска имеет правильную длину
            assert len(cluster_mask) == len(current_genes), "Несоответствие размеров маски и генов"
            
            # Создаем маску для удаления: все гены из кластера X, кроме самого X
            remove_mask = np.zeros(len(current_genes), dtype=bool)
            remove_mask[:-1] = cluster_mask[:-1]  # Не включаем сам X
            
            # Удаляем гены из этого кластера (кроме X)
            if np.any(remove_mask):
                new_removed = [g for g, m in zip(current_genes, remove_mask) if m]
                removed_genes.extend(new_removed)
                
                # Сохраняем удаленные гены
                pd.DataFrame({'Gene': new_removed}).to_csv(
                    os.path.join(filtered_dir, f'iteration_{iteration}_filtered.csv'), 
                    index=False)
                
                # Обновляем данные (оставляем только гены не из кластера X + сам X)
                keep_mask = ~remove_mask
                current_features = current_features[keep_mask]
                current_genes = [g for g, m in zip(current_genes, keep_mask) if m]
                
                print(f"Фильтрация: удалено {len(new_removed)} генов, осталось {len(current_genes)}")
                filter_count += 1
        iteration_info = {
            'number': iteration,
            'n_clusters': n_current,
            'inertia': best_inertia,
            'removed_genes': len(new_removed) if 'new_removed' in locals() else 0,
            'remaining_genes': len(current_genes)
        }
        logger.log_iteration(iteration_info)
        n_current += n_deductible
    
    print("\n5. Финальная кластеризация...")
    # Кластеризация оставшихся генов
    final_n_clusters = min(n_current, len(current_genes))
    final_kmeans = KMeans(n_clusters=final_n_clusters, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(current_features)
    
    # Сохранение результатов
    clusters = defaultdict(list)
    for gene, cluster_id in zip(current_genes, final_labels):
        clusters[cluster_id].append(gene)
    
    for cluster_id, genes in clusters.items():
        pd.DataFrame({'Gene': genes}).to_csv(
            os.path.join(final_dir, f'cluster_{cluster_id}.csv'), 
            index=False)
    
    # Сохранение центров кластеров (только экспрессия)
    centers = final_kmeans.cluster_centers_[:, :data.shape[0]]  # Только экспрессионные признаки
    centers_df = pd.DataFrame(centers.T, columns=[f'Cluster_{i}' for i in range(final_n_clusters)])
    centers_df.insert(0, 'Sample', first_column)
    centers_df.to_csv(os.path.join(output_dir, 'cluster_centers.csv'), index=False)
    
    # Сохранение удаленных генов (если есть)
    if removed_genes:
        pd.DataFrame({'Gene': removed_genes}).to_csv(
            os.path.join(output_dir, 'all_filtered_genes.csv'),
            index=False)
    
    print("\n6. Анализ результатов...")
    if GO_AVAILABLE and go_analyzer.go_dag:
        logger.logger.info("\nАНАЛИЗ GO-ТЕРМИНОВ ПО КЛАСТЕРАМ:")
        print("\nТоп GO-терминов по кластерам:")
        for cluster_id, genes in clusters.items():
            if cluster_id == 'X':
                continue
                
            print(f"\nКластер {cluster_id} ({len(genes)} генов):")
            
            # Собираем все GO-термины для кластера
            cluster_terms = defaultdict(int)
            for gene in genes:
                standardized_gene = go_analyzer.standardize_gene_name(gene)
                for term in go_analyzer.gene_go.get(standardized_gene, []):
                    cluster_terms[term] += 1
            
            # Сортируем термины по частоте
            sorted_terms = sorted(cluster_terms.items(), key=lambda x: x[1], reverse=True)
            logger.logger.info(f"\nКластер {cluster_id} ({len(genes)} генов):")
            
            # Выводим топ-3 термина для каждого аспекта
            for aspect in ['biological_process', 'molecular_function', 'cellular_component']:
                aspect_terms = [(t, cnt) for t, cnt in sorted_terms 
                               if go_analyzer.go_aspects.get(t) == aspect]
                
                if aspect_terms:
                    top_term = aspect_terms[0]
                    term_name = go_analyzer.go_dag[top_term[0]].name
                    print(f"  {aspect}: {term_name} ({top_term[1]} генов)")
    
    total_time = time.time() - start_time
    final_stats = {
        'total_clusters': len(clusters),
        'total_removed': len(removed_genes),
        'execution_time': total_time
    }
    logger.log_final_stats(final_stats)
    print(f"\nПрограмма завершена за {total_time:.2f} секунд")
    print(f"Всего кластеров: {len(clusters)}")
    print(f"Удалено генов: {len(removed_genes)}")
    print(f"Результаты сохранены в: {output_dir}")

if __name__ == "__main__":
    main()
