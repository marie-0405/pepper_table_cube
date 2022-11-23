import os

epsilon_begin = 0.2  # TODO decrease to 0.2 or delete
epsilon_end = 0.01
data_file_name = 'test2'
date = '2022-11-21'
file_name_end = ['first_with_human']
gamma = 0.99  # TODO TUNING 0.7 良さそうだったらだんだん小さくする
# gamma = 0.7
lr = 3e-4
# TODO leraing rate を変更することがでっきる
# TODO Change nepisode
# nepisodes = 300
# test
nepisodes = 3
nsteps = 50
nvideos = 20
w1 = 2.0
w2 = 3.0
base_reward = 0

if __name__ == '__main__':
  for fne in file_name_end:
    # os.mkdir('../training_results/{}-{}'.format(date, fne))
    # os.mkdir('../training_results/{}-{}/model'.format(date, fne))
    # os.mkdir('../training_results/{}-{}/optimizer'.format(date, fne))
    os.mkdir('../training_results/{}-{}/'.format(date, 'test_' + fne))  # TODO test
    # os.mkdir('./huuman_data/{}-{}/'.format(date, fne))  # TODO human
