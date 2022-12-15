import os

epsilon_begin = 0.2  # TODO decrease to 0.2 or delete
epsilon_end = 0.01
video_date = '2022-12-01'
video_file_name = 'direct_to_cube'
date = '2022-12-14'
file_name_end = ['baseline_40steps']
gamma = 0.99  # TODO TUNING 0.7 良さそうだったらだんだん小さくする
dropout_rate = 0.8

# gamma = 0.7
lr = 3e-4
weight_decay = 1e-2
# TODO leraing rate を変更することがでっきる
# TODO Change nepisode
# nepisodes = 300
# test
nepisodes = 400
nsteps = 40
nvideos = 20
w1 = 2.0
w2 = 3.0
base_reward = 0

if __name__ == '__main__':
  for fne in file_name_end:
    os.mkdir('../training_results/{}-{}'.format(date, fne))
    os.mkdir('../training_results/{}-{}/model'.format(date, fne))
    os.mkdir('../training_results/{}-{}/optimizer'.format(date, fne))
    # os.mkdir('../training_results/{}-{}/'.format(date, 'test_' + fne))  # TODO test
    # os.mkdir('./human_data/{}-{}/'.format(video_date, video_file_name))  # TODO human
