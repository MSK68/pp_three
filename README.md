# Классификация эмоций на основе текста

Этот проект реализует многометочную классификацию эмоций с использованием модели ai-forever/ruBert-large. В процессе работы данные расширяются за счет датасета ru-izard-emotions, очищаются, анализируются, обучается нейронная сеть, а также выполняется предсказание эмоций.

---

## Содержание

1. [Требования](#требования)
2. [Датасеты](#датасеты)
3. [Установка](#установка)
4. [Предобработка](#предобработка)
5. [Обучение и валидация](#обучение-и-валидация)
6. [Тестирование](#тестирование)
7. [Создание файла сабмита](#создание-файла-сабмита)
8. [Результаты](#результаты)

---

## Требования

Для запуска проекта установите следующие библиотеки:

```bash
pip install -q transformers datasets librosa torch scikit-learn matplotlib pandas
```

---

## Датасеты

Скачайте данные по [ссылке](https://disk.yandex.ru/d/awG8jCY01BGcAQ) и разместите файлы следующим образом:

```
.
├── train.csv
├── valid.csv
├── test_without_answers.csv
```

### Метки

Датасет содержит следующие метки эмоций:

- `anger`
- `disgust`
- `fear`
- `joy`
- `sadness`
- `surprise`
- `neutral`

---

## Установка

1. Клонируйте репозиторий и перейдите в папку проекта.
2. Установите все необходимые библиотеки, указанные в разделе "Требования".

---

## Предобработка

1. **Анализ данных**:
   - Проверка на пропущенные значения и дубликаты.
   - Анализ уникальных символов в текстах.
   - Построение гистограмм распределения эмоций и их со-встречаемости.
2. **Очистка данных**: 
   - Перевод текста в нижний регистр
   - Удаление лишних символов
   - Нормализация пробелов
   - Удаление дубликатов
   - Лемматизация
   - Удаление стоп-слов

---

## Обучение и валидация

1. Определяется модель на основе `CustomBertModel` из библиотеки Transformers.
2. Токенизация датасета и подготовка DataLoader для PyTorch.
3. Обучение модели с использованием функции ошибки `BCEWithLogitsLoss` и оптимизатора AdamW.

Пример запуска обучения:

```python
for epoch in range(epochs):
    print(f"Epoch: {epoch+1}")
    train_loss = train(model, criterion, optimizer, scheduler, train_dataloader)
    val_loss, val_outputs, val_targets = validation(model, criterion, valid_dataloader)

    # Вычисление F1-score
    val_f1 = f1_score(val_targets, (np.array(val_outputs) > 0.5).astype(int), average='weighted')
    print(f"Train loss: {train_loss}, Valid loss: {val_loss}, Valid F1: {val_f1}")

    # Ранняя остановка
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break
```

---

## Тестирование

1. Загрузка и предобработка тестового датасета.
2. Генерация предсказаний с использованием обученной модели.

Пример команды для тестирования:

```python
outputs, _ = validation(model, criterion, test_dataloader)
```

---

## Создание файла сабмита

1. Обновите файл `test_without_answers.csv` предсказанными значениями.
2. Сохраните результат в файл `submission.csv`.

Пример команды:

```python
df[labels] = outputs.astype(int)
df.to_csv("/content/submission.csv", index=False)
```

---

## Результаты

На валидационных данных при обучении в течение пяти эпох получили значение:
- Valid loss: 0.26827253162457226

На неразмеченных данных:
- Valid loss: 0.2817318362557378

---



