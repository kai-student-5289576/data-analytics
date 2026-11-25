# Лабораторні роботи з курсу Data Science (Аналіз даних та машинне навчання на Python)

Репозиторій містить виконані лабораторні роботи №1–4 у форматі інтерактивних Jupyter Notebook (`.ipynb`) з програмним кодом, обчисленими результатами, візуалізаціями (графіками), висновками та відповідями на контрольні запитання.

---

## Структура репозиторію

| Файл | Опис |
| :--- | :--- |
| [`lab1_data_wrangling.ipynb`](lab1_data_wrangling.ipynb) / [`.py`](lab1_data_wrangling.py) | **Лабораторна робота №1: Data Wrangling** — завантаження автомобільного датасету, виявлення та імпутація пропущених значень (`NaN`), коректування типів даних, стандартизація (`mpg` $\rightarrow$ `L/100km`), нормалізація ознак (`length`, `width`, `height`), бінінг (`horsepower`) та створення індикаторних змінних (`fuel-type`, `aspiration`). |
| [`lab2_data_analysis.ipynb`](lab2_data_analysis.ipynb) / [`.py`](lab2_data_analysis.py) | **Лабораторна робота №2: Аналіз даних за допомогою Python (EDA)** — кореляційний аналіз (`.corr()`), діаграми розсіювання з лінією регресії (`sns.regplot`), аналіз категоріальних змінних (`sns.boxplot`), описова статистика (`.describe()`, `.value_counts()`), групування (`groupby`) та теплові карти, коефіцієнт кореляції Пірсона і $P\text{-value}$, однофакторний дисперсійний аналіз (ANOVA, $F\text{-test}$). |
| [`lab3_model_development.ipynb`](lab3_model_development.ipynb) / [`.py`](lab3_model_development.py) | **Лабораторна робота №3: Model Development** — проста лінійна регресія (`highway-mpg`, `engine-size` $\rightarrow$ `price`), графіки залишків (`residplot`), множинна лінійна регресія (MLR), графіки розподілу (KDE), поліноміальна регресія (3-го та 11-го порядків), конвеєри (`Pipeline` зі `StandardScaler` та `PolynomialFeatures`), порівняння моделей за $R^2$ та $\text{MSE}$. |
| [`lab4_model_evaluation.ipynb`](lab4_model_evaluation.ipynb) / [`.py`](lab4_model_evaluation.py) | **Лабораторна робота №4: Model Evaluation and Refinement** — розбиття на тренувальну і тестову вибірки (`train_test_split`), перехресна перевірка (`cross_val_score`), моделювання та візуалізація явищ **Overfitting** і **Underfitting**, гребенева регуляризація (`Ridge`) та пошук оптимальних гіперпараметрів за допомогою `GridSearchCV`. |
| [`auto.csv`](auto.csv) | Вихідний набір даних автомобілів (*Automobile Dataset*, UCI Machine Learning Repository). |
| [`clean_df_lab1.csv`](clean_df_lab1.csv) | Очищений та нормалізований датасет після виконання Лабораторної роботи №1. |
| [`clean_auto.csv`](clean_auto.csv) | Підготовлений датасет для статистичного аналізу та побудови моделей (ЛР №2–4). |
| [`requirements.txt`](requirements.txt) | Список необхідних бібліотек Python. |

---

## Встановлення та запуск

1. Встановіть залежності:
   ```bash
   pip install -r requirements.txt
   ```
2. Запустіть Jupyter Notebook або JupyterLab:
   ```bash
   jupyter notebook
   ```
   або відкрийте файли `.ipynb` безпосередньо у VS Code / GitHub.
