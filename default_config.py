num_seeds = 10
state_space_sizes = [8]#, 10, 12, 14] # [2**i for i in range(1,6)]
action_space_sizes = [8]#2, 4, 8, 16] # [2**i for i in range(1,6)]
delays = [0] + [2**i for i in range(4)]
sequence_lengths = [1, 2, 3, 4]#i for i in range(1,4)]
reward_densities = [0.25] # np.linspace(0.0, 1.0, num=5)
# make_reward_dense = [True, False]
terminal_state_densities = [0.25] # np.linspace(0.1, 1.0, num=5)
algorithms = ["DQN"]
seeds = [i for i in range(num_seeds)]
# Others, keep the rest fixed for these: learning_starts, target_network_update_freq, double_dqn, fcnet_hiddens, fcnet_activation, use_lstm, lstm_seq_len, sample_batch_size/train_batch_size, learning rate
# More others: adam_epsilon, exploration_final_eps/exploration_fraction, buffer_size
num_layerss = [1, 2, 3, 4]
layer_widths = [8, 32, 128]
fcnet_activations = ["tanh", "relu", "sigmoid"]
learning_startss = [500, 1000, 2000, 4000, 8000]
target_network_update_freqs = [8, 80, 800]
double_dqn = [False, True]
learning_rates = []
