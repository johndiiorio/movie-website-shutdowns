from utils import (
    get_csv_data,
    generate_users,
    similarity_probabilities_with_user,
    normalize,
    time_usage,
    select_preferences,
    viterbi
)


@time_usage
def simulation():
    # Assumptions
    preferences_csv_file = 'preferences.csv'
    t_matrix_csv_file = 'transition_matrix.csv'
    num_users = 100000
    t_matrix_start_value = .9
    num_observations = 3
    mean_user_preference = 5
    sigma_user_preference = 3
    sites_to_destroy = ['KickassTorrents', 'Torrentz']

    # Parse CSV
    data, columns = get_csv_data(preferences_csv_file)
    hidden_states = columns['Name']
    observables = data[0][1:]
    categories = list(columns.keys())
    categories.remove('Name')

    site_scores = {}
    for row in data[1:]:
        site_scores[row[0]] = [int(x) for x in row[1:]]

    users = generate_users(categories, num_users, mean_user_preference, sigma_user_preference)

    # Probability of each user starting in each site
    # Each array is a user, the values are the sites (in order from the csv)
    state_pi_map = {k: 0 for k in hidden_states}
    pi_list = []
    for user in users:
        user_sim_prob = similarity_probabilities_with_user(user.preferences, site_scores)
        for k, v in user_sim_prob.items():
            state_pi_map[k] += v
        pi_list.append(user_sim_prob)
    state_pi_map = {k: v / num_users for k, v in state_pi_map.items()}
    print('Probability of a user starting at a site:')
    print(state_pi_map)

    # Transition matrix
    transition_matrix = {}
    data_t_matrix, columns_t_matrix = get_csv_data(t_matrix_csv_file)
    data_t_matrix = data_t_matrix[1:]
    # Set initial values from csv
    for row in data_t_matrix:
        transition_matrix[row[0]] = dict(zip(hidden_states, normalize([int(x) for x in row[1:]], 1 - t_matrix_start_value)))
    # Destroy sites and set self values (i.e. Netflix -> Netflix, Primewire -> Primewire)
    for key, value in transition_matrix.items():
        value[key] = t_matrix_start_value
        for site_to_destroy in sites_to_destroy:
            value[site_to_destroy] = 0
    # Normalize again to account for changes
    for key, value in transition_matrix.items():
        transition_matrix[key] = dict(zip(hidden_states, normalize(value.values())))

    # Populate emission matrix for each user
    emission_matrix_list = []
    for user in users:
        emission_matrix = {}
        for h_state in hidden_states:
            emission_matrix[h_state] = {}
            for i, observable in enumerate(observables):
                emission_matrix[h_state][observable] = user.preferences[i] / 10
        emission_matrix_list.append(emission_matrix)

    observations_over_time_list = []
    for user in users:
        observations_over_time_list.append(tuple(select_preferences(categories, user, num_observations)))

    freq_hidden_states = {k: 0 for k in hidden_states}
    for i, user in enumerate(users):
        path, max_prob = viterbi(
            observations_over_time_list[i],
            tuple(hidden_states),
            pi_list[i],
            transition_matrix,
            emission_matrix_list[i]
        )
        freq_hidden_states[path[-1]] += 1
    print('After destroying the sites:')
    print({k: v / num_users for k, v in freq_hidden_states.items()})


if __name__ == '__main__':
    simulation()
