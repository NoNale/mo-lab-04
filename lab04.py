import numpy as np
import matplotlib.pyplot as plt

# вектор весов критериев
vector_criteria_ = [8, 4, 6, 2]
# Нормализуем ветор весов
vector_criteria_ = np.divide(vector_criteria_, sum(vector_criteria_))
print(f"Вектор весов критериев: {vector_criteria_}")
# матрица A оценок для альтернатив
A = np.array([
    [5., 5., 4., 6.],
    [7., 6., 3., 2.],
    [3., 4., 6., 4.],
    [2., 3., 7., 3.]
])
print(f"Матрица A оценок для альтернатив:\n {A}")
# Выберим в качестве главного критерия начальную цену
# Установим минимально допустимые уровни для остальных критериев:
criteria_ = np.array([None, 0.3, 0.6, 0.1])
print(f"Допустимые значения: {criteria_}")


# метод замены критериев ограничениями
class MethodReplacingCriteria:
    def __init__(self, matrix, criteria, index):
        self.matrix = matrix.copy()
        self.criteria = criteria.copy()
        self.index = index
        self.matrix_normalization()
        self.answer = self.acceptable_solutions()

    def matrix_normalization(self):
        min_ = np.min(self.matrix, axis=0)
        max_ = np.max(self.matrix, axis=0)
        for i in range(4):
            if i != self.index:
                for j in range(4):
                    self.matrix[j][i] = (self.matrix[j][i] - min_[i]) / (max_[i] - min_[i])
        print(f"Нормированная матрица A:\n {self.matrix}")

    def check_lower_thresholds(self, row):
        for col in range(4):
            if col != self.index:
                if row[col] < self.criteria[col]:
                    return False
        return True

    def acceptable_solutions(self):
        answer = None
        for row in range(4):
            if self.check_lower_thresholds(self.matrix[row]):
                if answer is None or self.matrix[answer][self.index] < self.matrix[row][self.index]:
                    answer = row
        return answer


task1 = MethodReplacingCriteria(matrix=A, criteria=criteria_, index=0)
if task1.answer is None:
    print(f"Подходящего решения нету")
else:
    print(f"При заданных условиях самым лучшим решением является: {task1.answer}")


# Формирование и сужение множества Парето
# Выберим в качестве двух основных критериев: качество пляжа (№3) и цена (№1)
class SetPareto:
    def __init__(self, matrix, criteria1, criteria3):
        self.x = matrix[:, criteria1][:]
        self.y = matrix[:, criteria3][:]
        self.max_x = np.max(self.x)
        self.max_y = np.max(self.y)
        self.answer = self.solution_set_pareto()

    def graph_set_pareto(self):
        plt.plot(self.x, self.y, 'o')
        plt.show()

    def distance_chebyshev(self, x, y):
        return np.max([np.abs(x - self.max_x), np.abs(y - self.max_y)])

    def solution_set_pareto(self):
        min_distance = self.distance_chebyshev(0., 0.)
        answer = None
        for i in range(4):
            dis = self.distance_chebyshev(self.x[i], self.y[i])
        if dis < min_distance:
            min_distance = dis
        answer = i
        return answer


task2 = SetPareto(A, 0, 2)
task2.graph_set_pareto()
print(f"Минимальное расстояние до точки {task2.answer}, следовательно решение {task2.answer} оптимально")


# Взвешивание и объединение критериев
class MethodWeighingCombiningCriteria:
    def __init__(self, vector_criteria, matrix):
        self.matrix = matrix.copy()

        self.vector_criteria = vector_criteria.copy()
        self.criteria_weights = self.method_pairwise_comparison()
        self.matrix_normalization()
        self.answer = self.get_solution()

    @staticmethod
    def compare_criteria(criteria1, criteria2):
        if criteria1 > criteria2:
            return 1
        elif criteria1 == criteria2:
            return 0.5
        return 0

    def method_pairwise_comparison(self):
        expert_grade = np.array([
            [self.compare_criteria(criteria_1, criteria_2) for criteria_2 in
             self.vector_criteria]
            for criteria_1 in self.vector_criteria
        ])
        print(f"Экспертные оценки:\n {expert_grade}")
        criteria_weights = np.array([sum(expert_grade[i]) - 0.5 for i in
                                     range(4)])
        print(f"Вектор весов критериев: {criteria_weights}")
        criteria_weights = np.divide(criteria_weights, sum(criteria_weights))
        print(f"Нормированный вектор весов критериев: {criteria_weights}")
        return criteria_weights

    def matrix_normalization(self):
        for col in range(len(self.matrix)):
            self.matrix[:, col] = np.divide(self.matrix[:, col], np.sum(self.matrix[:, col]))

        print(f"Нормированная матрица A:\n {self.matrix}")

    def get_solution(self):
        self.criteria_weights = np.transpose(np.matrix(self.vector_criteria))

        self.matrix = np.matrix(self.matrix)
        solutions = self.matrix * self.criteria_weights
        max_result = 0
        answer = None
        for i in range(4):
            if solutions[i] > max_result:
                max_result = solutions[i]
        answer = i
        return answer


task3 = MethodWeighingCombiningCriteria(matrix=A,
                                        vector_criteria=vector_criteria_)
print(f"Значения объединенного критерия для всех альтернатив:\n {task3.matrix * task3.criteria_weights}")
print(f"Наиболее приемлемой является альтернатива {task3.answer}")

# Метод анализа иерархий
matrix_rating_ = np.array([
    [3, 3, 1, 3],
    [1, 2, 1, 0],
    [0, 1, 3, 2],
    [0, 0, 3, 1]
])
vector_priorities_ = np.array([3, 1, 2, 0])


class MethodHierarchyAnalysis:
    def __init__(self, matrix_rating, vector_priorities):
        self.matrix = matrix_rating.copy()
        self.vector_priorities = vector_priorities
        self.matrix_criteria0 = self.pairwise_comparison(0)
        self.matrix_criteria1 = self.pairwise_comparison(1)
        self.matrix_criteria2 = self.pairwise_comparison(2)
        self.matrix_criteria3 = self.pairwise_comparison(3)
        self.matrix_criterias = np.transpose(
            np.matrix([self.matrix_criteria0, self.matrix_criteria1, self.matrix_criteria2, self.matrix_criteria3]))
        self.matrix_priorities = np.transpose(np.matrix(self.pairwise_comparison_matrix_priorities()))
        self.answer = self.get_solution()

    @staticmethod
    def compare_rating(rating1, rating2):
        delta = rating1 - rating2
        if delta == 3:
            return 7
        elif delta == 2:
            return 5
        elif delta == 1:
            return 3
        elif delta == 0:
            return 1
        elif delta == -1:
            return 1 / 3
        elif delta == -2:
            return 1 / 5
        else:
            return 1 / 7

    def pairwise_comparison(self, criteria_index):
        new_matrix = np.array(
            [[self.compare_rating(self.matrix[row_1][criteria_index], self.matrix[row_2][criteria_index])
              for row_2 in range(len(self.matrix))]
             for row_1 in range(len(self.matrix))])
        sum_line = np.array([np.sum(new_matrix[i]) for i in range(len(new_matrix))])
        sum_line = np.divide(sum_line, np.sum(sum_line))
        return sum_line

    def pairwise_comparison_matrix_priorities(self):
        new_matrix = np.array([
            [self.compare_rating(self.vector_priorities[row_1],
                                 self.vector_priorities[row_2])
             for row_2 in range(len(self.matrix))]
            for row_1 in range(len(self.matrix))
        ])
        sum_line = np.array([np.sum(new_matrix[i]) for i in
                             range(len(new_matrix))])
        sum_line = np.divide(sum_line, np.sum(sum_line))
        return sum_line

    @staticmethod
    def get_consistency_relation(matrix_criteria):
        sum_column = np.array([np.sum(matrix_criteria[:, i][:]) for i in
                               range(len(matrix_criteria))])

        sum_line = np.array([np.sum(matrix_criteria[i]) for i in
                             range(len(matrix_criteria))])
        sum_line = np.divide(sum_line, np.sum(sum_line))
        random_consistency_score = 1.12
        n = 4
        consistency_relation = (np.sum((sum_column * sum_line)) - n) / (n -
                                                                        1) / random_consistency_score
        print(f"Отношение согласованности {consistency_relation}")

    def get_solution(self):
        solutions = self.matrix_criterias * self.matrix_priorities

        max_result = 0
        answer = None
        for i in range(4):
            if solutions[i] > max_result:
                max_result = solutions[i]
        answer = i
        return answer


task4 = MethodHierarchyAnalysis(matrix_rating=matrix_rating_,
                                vector_priorities=vector_priorities_)
print(task4.matrix_criterias)
print(task4.matrix_priorities)
print(f"Значения объединенного критерия для всех альтернатив:\n{task4.matrix_criterias * task4.matrix_priorities}")
print(f"Наиболее приемлемой является альтернатива {task4.answer}")
