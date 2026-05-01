"""Simple timetable-style GA DNA operations."""

from __future__ import annotations

import copy
import random


class DNA:
    def __init__(self, data, mutation_rate):
        self.realdata = data
        self.data = copy.deepcopy(data)
        self.mutation_rate = mutation_rate
        self.genes = []
        self.fitness = 0.0

        for cls in self.data:
            class_genes = []
            subjects = list(cls[2])
            for _ in range(len(subjects)):
                gene = self.new_gen(cls[1], subjects)
                class_genes.append(gene)
                subjects.remove(gene[0])
            self.genes.append([cls[0], class_genes])

    def select_teacher(self, subject, teachers):
        t = random.choice(teachers)
        return t if subject in t.subjects else None

    def new_gen(self, teachers, subjects):
        subject = random.choice(subjects)
        teacher = self.select_teacher(subject, teachers)
        while teacher is None:
            teacher = self.select_teacher(subject, teachers)
        return [subject, teacher]

    def calc_fitness(self):
        score = 0
        for i in range(len(self.genes)):
            for k in range(len(self.genes)):
                for j in range(len(self.genes[i][1])):
                    if (
                        j < len(self.genes[k][1])
                        and self.genes[i][1][j][1].name != self.genes[k][1][j][1].name
                    ):
                        score += 1
        self.fitness = float(score) / 100.0
        return self.fitness
