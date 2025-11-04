#!/bin/python3
from pyspark import SparkContext
# Initialize SparkContext
sc = SparkContext("local", "TextFileExample")

rdd = sc.textFile("lab2_text.txt")
rdd.take(5)
# предобработка
import re
words = (rdd
    .map(lambda x: x.strip())
    # убираем whitespace по краям
    .map(lambda x: x.lower())
    # приводим к нижнему регистру
    .flatMap(lambda x: x.split()) # разделение на слова
    # регулярка: только слова 4+ символов:
    .filter(lambda x: re.match(r'^[A-Za-zА-Яа-я-]{4,}$', x))
)

words.take(5)
def counted(rdd, ascending=False):
    """Подсчёт количества элементов и сортировка: [(item, num), ...]"""
    return (rdd
    .map(lambda x: (x, 1))
    # добавляем 1 для суммы
    .reduceByKey(lambda a, b: a + b) # складываем
    #
    #.filter(lambda x: x[1] > 1)
    # фильтруем по более 1 появлению появлений
    .sortBy(lambda x: x[1], ascending=ascending) # сортировка по убыванию
    )
# биграммы и триграммы
indexed = (words
    .zipWithIndex()
    .map(lambda x: (x[1], x[0]))
)
indexed.take(5)
# [(word, index), ...]
# [(index, word), ...]
shifted = (indexed
    .map(lambda x: (x[0] - 1, x[1]))
)
shifted.take(5)
# сдвиг индекса [(index - 1, word), ...]
bigrams = (indexed
    .join(shifted)
    .mapValues(lambda words: " ".join(words))
    .map(lambda x: x[1])
)
bigrams.take(5)
# [(index, ["word1", "word2"]), ...]
# [(i, "word1 word2"), ...]
# убираем индексы
print(counted(bigrams).take(20))
shifted2 = (indexed
    .map(lambda x: (x[0] - 2, x[1]))
)
shifted2.take(5)
# сдвиг индекса [(index - 2, word), ...]
trigrams = (indexed
    .join(shifted)
    .mapValues(lambda words: " ".join(words))
    .join(shifted2)
    .mapValues(lambda words: " ".join(words))
    .map(lambda x: x[1])
)
trigrams.take(5)
print(counted(trigrams).take(20))
