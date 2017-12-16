import csv
import time
import scipy.stats as stats
import numpy as np
from collections import OrderedDict
from User import User


def time_usage(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        return_value = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Time to execute: {total_time}')
        return return_value
    return wrapper


def get_csv_data(path):
    data = []
    columns = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    for row in zip(*data):
        columns[row[0]] = list(row[1:])
    return data, columns


# Returns the probabilities of a user selecting a given site based on their preferences
def similarity_probabilities_with_user(user_prefs, site_scores):
    scores = OrderedDict()
    for site in site_scores.keys():
        score = 0
        for i in range(len(user_prefs)):
            # score += max(user_prefs[i] - site_scores[site][i], 0)
            score += user_prefs[i] ** site_scores[site][i]
        # scores[site] = (len(user_prefs) * 9) - score
        scores[site] = score
    normalized_scores = normalize([i for i in scores.values()])
    for i, site in enumerate(site_scores.keys()):
        scores[site] = normalized_scores[i]
    return dict(scores)


# Normalize values into probabilities that sum to 1
def normalize(vals, upper_bound=1):
    return [i * upper_bound / sum(vals) for i in vals]


# Select a specific site given user preferences and site scores
def select_site(user_preferences, site_scores):
    scores = similarity_probabilities_with_user(user_preferences, site_scores)
    return np.random.choice(list(scores.keys()), 1, p=list(scores.values()))[0]


# Select a specific category from user preferences
def select_preferences(categories, user, num=1):
    return np.random.choice(categories, num, p=normalize(user.preferences)).tolist()
    # Alternative option, return max
    # cat_user_map = dict(zip(categories, user.preferences))
    # return tuple([max(cat_user_map, key=cat_user_map.get) for _ in range(num)])


# Returns the probabilities of transitioning from one site to another
# Same site value takes in a number between 0 and 1 that will set matching site to
# All other sites will be normalized around that number
def similarity_probabilities_with_sites(site_name, site_scores, same_site_value):
    scores = OrderedDict()
    for site in site_scores.keys():
        score = 0
        for i in range(len(site_scores[site])):
            score += abs(site_scores[site_name][i] - site_scores[site][i])
        scores[site] = (len(site_scores) * 9) - score
    normalized_scores = normalize([i for i in scores.values()], 1 - same_site_value)
    for i, site in enumerate(site_scores.keys()):
        if site == site_name:
            scores[site] = same_site_value
        else:
            scores[site] = normalized_scores[i]
    return dict(scores)


def generate_truncated_normal_distribution(size, mu, sigma, lower, upper):
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(size).tolist()


def generate_users(categories, num_users, mean, sigma):
    distribution = generate_truncated_normal_distribution(len(categories) * num_users, mean, sigma, 0, 10)
    users = []
    count = 0
    for i in range(num_users):
        u = User()
        for _ in categories:
            u.preferences.append(distribution[count])
            count += 1
        users.append(u)
    return users


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    return opt, max_prob


def dp_table(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)