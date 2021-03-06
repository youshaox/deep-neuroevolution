#!/bin/sh
NAME=exp_`date "+%m_%d_%H_%M_%S"`
ALGO=$1
EXP_FILE=$2
tmux new -s $NAME -d
#  todo nohup存日志；改成debug; 改worker数量
tmux send-keys -t $NAME 'cd /Users/youshaoxiao/PycharmProjects/deep-neuroevolution;gopy3;python -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --algo '$ALGO' --exp_file '"$EXP_FILE" ' > master.out 2>&1&' C-m
tmux split-window -t $NAME
tmux send-keys -t $NAME 'cd /Users/youshaoxiao/PycharmProjects/deep-neuroevolution;gopy3;python -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --algo '$ALGO' --num_workers 7' ' > workers.out 2>&1&' C-m
tmux a -t $NAME
