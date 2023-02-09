#! /bin/bash -eux

tmux new -s check-heavy -d

tmux send-keys -t check-heavy 'tmux split-window -h -t 0' C-m
tmux send-keys -t check-heavy 'tmux split-window -v -t 0' C-m
tmux send-keys -t check-heavy 'tmux split-window -v -t 2' C-m
tmux send-keys -t check-heavy 'tmux split-window -v -t 2' C-m

tmux send-keys -t check-heavy 'tmux select-layout tiled' C-m

tmux send-keys -t check-heavy 'tmux respawn-pane -k -t 1 "htop"' C-m
tmux send-keys -t check-heavy 'tmux respawn-pane -k -t 2 "sudo dmesg -wH"' C-m
tmux send-keys -t check-heavy 'tmux respawn-pane -k -t 3 "sudo journalctl -f"' C-m
tmux send-keys -t check-heavy 'tmux respawn-pane -k -t 4 "dstat -cdngyip --top-io --top-bio --top-mem --top-cpu"' C-m
tmux send-keys -t check-heavy 'sudo iftop' C-m

tmux attach -t check-heavy
