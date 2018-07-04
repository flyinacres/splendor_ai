from game import Game
from ai import SplendorAI
from player import get_phase_parameters
from constants import *
from datetime import datetime
# from player import get_phase_parameters
import sys
from collections import defaultdict
from collections import Counter
import tensorflow as tf

from tensorflow.python.client import device_lib

base_game = Game(id=0, n_players=4, players_compare_self_to_others=True)
print(device_lib.list_local_devices())

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# temperature will decrease over time
def calculate_temperature(round):
    return 1.9/(1 + round/1.2) /(1.5+round/1.1)/(1+round/5)


# constants to define; default
NETWORK_HYPERPARAMETERS = {
    # player input
    'player_funnel_layers': [15,12,10],
    'reserved_funnel_layers': [12,10,],
    'inject_reserved_at_player_funnel_index': 1, # 0 same as input, 1 = as first layer, etc.
    'card_funnel_layers': [12,12,8],
    # game input
    'game_funnel_layers': [15, 12, 10],
    'game_objective_funnel_layers': [10, 8],
    'game_card_funnel_layers': [15, 12, 10],
    # overall, slightly increased from default
    'main_dense_layers': [108, 52, 16], #this is when everything is combined

    # output layers
    # this does not include the win layer
    'output_layers': [
        {
            'name': 'Q1',
            'lag': 1,
            'score': 1,
            'discount': 0.1,
            'gems': 0.01,

        },
        {
            'name': 'Q3',
            'lag':  3,
            'score': 1,
            'discount': 0.1,
            'gems': 0,
        },
        {
            'name': 'Q5',
            'lag': 5,
            'score': 1,
            'discount': 0.05,
            'gems': 0,
        },
    ],
}

def get_phase_parameters(phase):
    """
    training will be divided into 5 phases

    """
    if phase==1:
        return {
            'Q1': 0.5,
            'Q3': 0.3,
            'Q5': 0.15,
            'win': 0.05,
        }
    elif phase==2:
        return {
            'Q1': 0.35,
            'Q3': 0.25,
            'Q5': 0.25,
            'win': 0.15,
        }
    elif phase==3:
        return {
            'Q1': 0.25,
            'Q3': 0.25,
            'Q5': 0.25,
            'win': 0.25,
        }
    elif phase==4:
        return {
            'Q1': 0.15,
            'Q3': 0.2,
            'Q5': 0.35,
            'win': 0.3,
        }
    elif phase==5:
        return {
            'Q1': 0.05,
            'Q3': 0.1,
            'Q5': 0.35,
            'win': 0.50,
        }
    elif phase==6:
        return {
            'Q1': 0.03,
            'Q3': 0.1,
            'Q5': 0.22,
            'win': 0.75
        }

players = base_game.players
game_data = defaultdict(list)
run_name='run_9'

def summarize_counters(counter_list):
    master_counter = Counter()
    n = len(counter_list)
    for counter in counter_list:
        master_counter.update(counter)
    keys = list(master_counter.keys())
    for k in sorted(keys):
        master_counter[k] /= n
    return master_counter

set_durations = {}

n_rounds = 9
n_sets_per_round = 3
n_simulations_per_set = 300

start_time = datetime.now()
for i in range(n_rounds):
    print('ON ROUND', i)
    phase = min(6, i // 2 + 1)
    temperature = calculate_temperature(i)

    for p_i, player in enumerate(players):
        player.temperature = temperature
        player.decision_weighting = get_phase_parameters(phase)

    for j in range(n_sets_per_round):
        set_start_time = datetime.now()
        for k in range(n_simulations_per_set):
            # if (i==0) and (j==0):
            #    # soft restart
            #    break
            new_game = Game(id=i * n_sets_per_round + n_simulations_per_set + j * n_simulations_per_set + k,
                            players=players)
            stalemate = new_game.run()
            # stalemates should be extraordinarily rare and pretty much nonexistent
            if stalemate:
                very_long_game = new_game
                print('teaching ai not to stalemate -_-')
                for player in players:
                    player.transfer_history_to_ai()
                    player.ai.train_models(verbose=0, n_epochs=3)
            sys.stdout.write('.')
            sys.stdout.flush()
            # record historic game data for each set
            game_data[(i, j)].append(new_game.copy_plain_data_for_self_and_players())
        set_stop_time = datetime.now()
        duration = (set_start_time - set_stop_time).seconds
        for player in players:
            player.transfer_history_to_ai()
            # train more epochs this time to make better predictions
            player.ai.train_models(verbose=0, n_epochs=30, batch_size=1500)
        set_durations[(i, j, k)] = duration
        print('/')
        sys.stdout.flush()
        avg_game_length = np.mean([x['game']['turn'] for x in game_data[(i, j)]])
        print('R/S %d/%d AVERAGE GAME LENGTH: ' % (i, j), avg_game_length)
        avg_cards_purchased = np.mean([
            [
                pdata['n_cards']
                for pdata in x['players'].values()
            ]
            for x in game_data[(i, j)]
        ]
        )
        print('R/S %d/%d AVERAGE CARDS PURCHASED: ' % (i, j), avg_cards_purchased)
        # calculate win rates
        player_data = [x['players'] for x in game_data[(i, j)]]
        win_values = [[player_data[idx][pid]['win'] for idx in range(n_simulations_per_set)] for pid in range(4)]
        win_rates = [np.mean(v) for v in win_values]
        print('WIN RATES: ', str(win_rates))
        # reservation averages
        reservation_counters = [
            [
                player_data[idx][pid]['total_times_reserved']
                for idx in range(n_simulations_per_set)
            ]
            for pid in range(4)
        ]
        print('RESERVATION SUMMARY: ', [summarize_counters(counters) for counters in reservation_counters])

        # gem taking averages
        gem_counters = [
            [
                player_data[idx][pid]['total_gems_taken']
                for idx in range(n_simulations_per_set)
            ]
            for pid in range(4)
        ]
        print('GEMS SUMMARY: ', [summarize_counters(counters) for counters in gem_counters])

    for player in players:
        player.reset(reset_extended_history=True)

    for p_i, player in enumerate(players):
        model_name = '{run_name}_player_%s_'.format(run_name=run_name) % str(p_i)
        player.ai.save_models(model_name, index=i)

stop_time = datetime.now()
for time in [start_time, stop_time]:
    print(time.strftime('%x %X'))

# save historic data
import pickle

with open('{run_name}_game_data.dat'.format(run_name=run_name), 'wb') as f:
    pickle.dump(game_data, f)

with open('{run_name}_duration_data.dat'.format(run_name=run_name), 'wb') as f:
    pickle.dump(set_durations, f)

