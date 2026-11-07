#!/bin/bash
# 9 обучений: 3 варианта x 3 сида, по 3 процесса одновременно.
# Запуск:  caffeinate -i bash run_v4_training.sh
# Если параллельные запуски SUMO будут конфликтовать, уберите "&" в трёх строках.
set -e
mkdir -p models
for s in 0 1 2; do
  python train_mixed.py --reward default --seed $s --name ppo_mixed_default_s$s &
  python train_mixed.py --reward backlog --seed $s --name ppo_mixed_backlog_s$s &
  python train_mixed.py --reward default --loads medium --seed $s --name ppo_single_default_s$s &
  wait
done
echo "Все обучения завершены"
