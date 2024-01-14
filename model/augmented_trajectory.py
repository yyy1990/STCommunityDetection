import random
import copy
import numpy as np
from collections import Counter


def random_remove(seed_p, weighted=True):
    if weighted:
        places = Counter(seed_p)
        placeids = list(places.keys())
        if len(placeids)>1:
            weights= list(places.values())
            weights = np.array(weights)
            weights = 1/weights
            p = random.choices(population=placeids, weights=weights)[0]
            ans_p = [x for x in seed_p if x!=p]
        else:
            i = random.randint(0, len(seed_p)-1)
            ans_p = copy.deepcopy(seed_p)
            del ans_p[i]
    else: # random remove
        places = set(seed_p)
        if len(places) == 1:
            i = random.randint(0, len(seed_p)-1)
            ans_p = copy.deepcopy(seed_p)
            del ans_p[i]
        else:
            p = random.choice(places)
            ans_p = [x for x in seed_p if x!=p]
    return ans_p



def random_add(seed_p, graph, aug_num, weighted=True):
    places = Counter(seed_p)
    placeids = list(places.keys())
    if weighted:
        weights = list(places.values())
    else:
        weights = [1 for i in range(placeids)]

    places_start = random.choices(population=placeids, weights=weights, k=aug_num)
    places_add = []
    for p in places_start:
        if p in graph:
            weight = graph[p]
            next_p = random.choices(population=list(weight.keys()), weights=list(weight.values()))[0]
        else:
            next_p = random.choice(list(graph.keys()))
        places_add.append(next_p)
    ans_p = seed_p+places_add
    return ans_p


def random_add_choice(seed_p, spatial_add_prob, spatial_edge, temporal_edge, aug_num, weighted):
    prob = random.random()
    if prob<spatial_add_prob:
        return random_add(seed_p, spatial_edge, aug_num, weighted)
    else:
        return random_add(seed_p, temporal_edge, aug_num, weighted)


def random_aug(seed_p, spatial_edge, temporal_edge, spatial_add_prob=0.5, add_prob=0.5, min_len=5, max_len=4096, aug_num=20,
               weighted_add=True, weighted_remove = True):
    if min_len>1 and len(seed_p)<=min_len:
        seed_p = random_add_choice(seed_p, spatial_add_prob, spatial_edge, temporal_edge, aug_num, weighted=weighted_add)
    elif max_len>1 and len(seed_p)>=max_len:
        seed_p = random_remove(seed_p, weighted=weighted_remove)
    else:
        prob = random.random()
        if prob<add_prob:
            seed_p = random_add_choice(seed_p, spatial_add_prob, spatial_edge, temporal_edge, aug_num, weighted=weighted_add)
        else:
            seed_p = random_remove(seed_p, weighted=weighted_remove)
    return seed_p


def random_aug_walk(seed_p, spatial_edge, temporal_edge, spatial_add_prob=0.5, add_prob=0.5, min_len=5, max_len=4096, aug_num=20):
    seed_p = random_aug(seed_p, spatial_edge, temporal_edge, spatial_add_prob, add_prob, min_len, max_len, aug_num)
    return seed_p
