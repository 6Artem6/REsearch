# 📋 ТЗ для Cursor: Движок Архитектурного Анализа (Local Knowledge Engine MVP)

## 🎯 Проблематика и Цель проекта

### Проблема

Существующие решения (RAG, обычные LLM-чаты, поиск) работают по принципу "новостной ленты" или прямого текстового поиска. Они выдают либо устаревшие базовые вещи, либо слишком сырой "хайповый" стек без проверки на точки отказа (Failure Modes). Инженеру приходится вручную фильтровать тонны информации, чтобы найти решение конкретной узкой задачи. На рынке **нет доступных автономных систем**, которые переводили бы инженерную проблему на язык фундаментальных компьютерных наук (CS-абстракций), искали решения по нескольким временным горизонтам и выдавали строго валидированную Trade-off матрицу.

### Цель

Создать локальный CLI-инструмент на базе **LangGraph** и гибридного стека (**Gemini API** + **Ollama** 7B/1.5B на Mac), который принимал бы любую бэкенд/AI задачу, раскладывал ее на фундаментальные паттерны, выполнял управляемый Re-Act поиск и формировал структурированный разбор с возможностью точечной "раскрутки" (Unraveling) любого узла.

**Актуальная реализация (2026-07):** `GRAPH_VERSION=0.4`, REST API, Domain Trust, архив ссылок, Smart Targeted Search — см. `knowledge_engine/docs/V0_6_CURRENT_SOLUTION.md`.

---

## 🛠 Технологический стек и Конкретные Модели

Никаких неопределенностей — используем только этот стек:

1. **Оркестрация и Граф:** `langgraph` (v0.2+), `langchain-core`
2. **Валидация и Схемы:** `pydantic` (v2.0+)
3. **LLM-инференс (Ollama):** `langchain-ollama`
4. **Модели (Обязательно двухуровневый подход):**
* **Мини-роутер (1B–3B):** `qwen2.5-coder:1.5b` (Задачи: роутинг, проверка условий, Pydantic-классификация, контроллер Re-Act цикла). *Быстрый, отклик ~100мс.*
* **Основная архитектурная модель (7B–14B):** `qwen2.5-coder:7b` (Задачи: глубокая декомпозиция, сборка Trade-off матрицы, финальный Unraveling).


5. **Интерфейс и Отображение:** `rich` (красивый вывод таблиц и статусов в консоли), `typer` (управление CLI).
6. **Память состояния:** `langgraph.checkpoint.memory.MemorySaver` (для реализации Human-in-the-loop паузы).

---

## 📁 Фиксированная Структура Проекта

Проект должен иметь строго следующую файловую структуру:

```text
knowledge_engine/
│
├── config.py            # Настройки Ollama (HOST, имена моделей, лимиты итераций)
├── schemas.py           # Bсе Pydantic-схемы данных и EngineState
├── nodes/               # Изолированные узлы графа
│   ├── __init__.py
│   ├── decomposition.py # Узел 1: Декомпозиция задачи на CS-абстракции (Основная модель)
│   ├── react_search.py  # Узел 2: Re-Act цикл поиска и валидации (Мини-роутер)
│   ├── matrix.py        # Узел 3: Сборка Trade-off матрицы (Основная модель)
│   └── unraveling.py    # Узел 4: Глубокая детализация выбранного решения (Основная модель)
│
├── graph.py             # Сборка StateGraph, Conditional Edges и MemorySaver
├── main.py              # CLI-интерфейс (Typer + Rich) с паузой на ввод пользователя
├── requirements.txt     # Зависимости с точными версиями
└── README.md            # Инструкция по запуску

```

---

## 📐 Pydantic-схемы (schemas.py)

```python
from typing import List, Optional
from pydantic import BaseModel, Field

# --- Выходные структуры моделей ---

class CSAbstraction(BaseModel):
    title: str = Field(description="Название фундаментальной проблемы")
    cs_concept: str = Field(description="Термин из CS (например, Cache Invalidation, Graph Topology, Event Sourcing)")
    description: str = Field(description="Почему задача сводится к этой абстракции")

class TradeOffOption(BaseModel):
    id: int = Field(description="Уникальный ID варианта (1, 2, 3)")
    pattern_name: str = Field(description="Название паттерна решения")
    category: str = Field(description="Категория: Классика / SOTA (Современное) / Минимализм")
    fundamental_idea: str = Field(description="Суть архитектурного паттерна")
    pros: List[str] = Field(description="Плюсы и сильные стороны")
    cons_and_risks: List[str] = Field(description="Точки отказа, риски и ограничения")
    operational_cost: str = Field(description="Нагрузка на инфраструктуру, RAM и сложность поддержки")

class AnalysisReport(BaseModel):
    abstractions: List[CSAbstraction]
    options: List[TradeOffOption]

# --- Состояние Графа (LangGraph State) ---

class EngineState(BaseModel):
    user_problem: str = Field(description="Исходная проблема от пользователя")
    context_constraints: str = Field(default="", description="Ограничения (например: Mac M1, Python, local, low latency)")
    abstractions: List[CSAbstraction] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    found_facts: List[str] = Field(default_factory=list)
    search_iterations: int = Field(default=0)
    is_facts_sufficient: bool = Field(default=False)
    report: Optional[AnalysisReport] = None
    selected_option_id: Optional[int] = None
    unraveled_details: Optional[str] = None

```

---

## 🔄 Логика Графа (graph.py)

1. **`decomposition_node` (Использует `qwen2.5-coder:7b`)**:
* Принимает `user_problem` и `context_constraints`.
* Возвращает список `CSAbstraction`.


2. **`react_search_node` (Использует `qwen2.5-coder:1.5b`)**:
* На основе абстракций генерирует целевые поисковые запросы (моделирует локальный или веб-поиск).
* Оценивает, достаточно ли фактов для построения матрицы (`is_facts_sufficient`).


3. **`should_continue` (Conditional Edge / Мини-роутер)**:
* Если `is_facts_sufficient == True` ИЛИ `search_iterations >= 3` $\rightarrow$ переход в `matrix_node`.
* Иначе $\rightarrow$ повтор `react_search_node` с `search_iterations += 1`.


4. **`matrix_node` (Использует `qwen2.5-coder:7b`)**:
* Генерирует ровно 3 варианта `TradeOffOption` в рамках `AnalysisReport`.


5. **`human_review_node` (INTERRUPT)**:
* Граф приостанавливается перед `unraveling_node`.
* CLI запрашивает у пользователя `selected_option_id` (какой вариант "раскрутить").


6. **`unraveling_node` (Использует `qwen2.5-coder:7b`)**:
* Выдает исчерпывающий разбор: алгоритм, структуры данных, пример кода/конфига и чек-лист для деплоя.
