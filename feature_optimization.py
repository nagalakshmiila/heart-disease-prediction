"""
feature_optimization.py
------------------------
This is the "Feature Optimization" part of the project title.

Novelty vs. most existing heart-disease papers (which use plain
correlation filtering, chi-square, or a single RFE pass):
we run a HYBRID two-stage optimizer:

  Stage 1 - RFE pre-filter:
      Random-Forest-ranked Recursive Feature Elimination quickly removes
      clearly useless features and gives an importance ranking.

  Stage 2 - Genetic Algorithm wrapper search:
      A GA searches the space of feature subsets (chromosome = bitmask over
      features), using 5-fold cross-validated F1 of a fast Random Forest as
      the fitness function, seeded with the RFE ranking so it converges fast.

The final selected subset is whatever the GA converges to. This is the
subset every model (baseline AND proposed hybrid DL model) is evaluated on,
so the accuracy comparison between them is fair.
"""

import json
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

SELECTED_FEATURES_PATH = "saved_models/selected_features.json"


def rfe_ranking(X, y, n_keep=None):
    n_keep = n_keep or max(4, X.shape[1] // 2)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=n_keep)
    rfe.fit(X, y)
    ranking = rfe.ranking_  # 1 = most important
    return ranking


def _fitness(mask, X, y, cv):
    if mask.sum() == 0:
        return 0.0
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cols = np.where(mask == 1)[0]
    score = cross_val_score(rf, X[:, cols], y, cv=cv, scoring="f1").mean()
    # small penalty per feature encourages a compact, more generalizable subset
    penalty = 0.002 * mask.sum()
    return score - penalty


def genetic_feature_selection(X, y, n_features, pop_size=20, generations=25,
                               mutation_rate=0.1, seed_ranking=None,
                               random_state=42):
    rng = np.random.default_rng(random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # initial population: random bitmasks, biased towards RFE-important features
    population = []
    for _ in range(pop_size):
        if seed_ranking is not None:
            probs = 1.0 / seed_ranking  # lower rank number -> higher prob of being kept
            probs = probs / probs.sum()
            keep_n = rng.integers(n_features // 2, n_features)
            chosen = rng.choice(n_features, size=keep_n, replace=False, p=probs)
            mask = np.zeros(n_features, dtype=int)
            mask[chosen] = 1
        else:
            mask = rng.integers(0, 2, n_features)
        population.append(mask)

    best_mask, best_fit = None, -1

    for gen in range(generations):
        fitnesses = np.array([_fitness(ind, X, y, cv) for ind in population])
        gen_best_idx = fitnesses.argmax()
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = fitnesses[gen_best_idx]
            best_mask = population[gen_best_idx].copy()

        # selection: tournament
        new_population = []
        for _ in range(pop_size):
            i, j = rng.integers(0, pop_size, 2)
            winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
            new_population.append(winner.copy())

        # crossover (single point)
        for i in range(0, pop_size - 1, 2):
            if rng.random() < 0.8:
                point = rng.integers(1, n_features)
                a, b = new_population[i].copy(), new_population[i + 1].copy()
                new_population[i][:point], new_population[i + 1][:point] = b[:point], a[:point]

        # mutation
        for ind in new_population:
            flip = rng.random(n_features) < mutation_rate
            ind[flip] = 1 - ind[flip]
            if ind.sum() == 0:  # never allow empty subset
                ind[rng.integers(0, n_features)] = 1

        population = new_population
        print(f"[GA] generation {gen + 1}/{generations} - best F1 so far: {best_fit:.4f}")

    return best_mask, best_fit


def optimize_features(X, y, feature_names, save=True):
    print("[feature_optimization] Stage 1: RFE pre-ranking...")
    ranking = rfe_ranking(X, y)
    print("[feature_optimization] Stage 2: Genetic Algorithm wrapper search...")
    best_mask, best_fit = genetic_feature_selection(
        X, y, n_features=X.shape[1], seed_ranking=ranking
    )
    selected = [f for f, m in zip(feature_names, best_mask) if m == 1]
    print(f"[feature_optimization] Selected {len(selected)}/{len(feature_names)} "
          f"features (CV F1={best_fit:.4f}): {selected}")

    if save:
        import os
        os.makedirs("saved_models", exist_ok=True)
        with open(SELECTED_FEATURES_PATH, "w") as f:
            json.dump({"selected_features": selected, "cv_f1": best_fit}, f, indent=2)

    return selected
