#!/bin/bash

# RUN THIS ON LAPTOP/WORKSTATION THAT RUNS LAPTOP_BRINGUP

SESSION="teleoperation_bringup_tmux"

# Check if the tmux session already exists
tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
    # Create a new tmux session and run the commands in separate panes
    tmux new-session -d -s $SESSION

    tmux send-keys -t $SESSION "python3 /home/robot/droid/scripts/bringup/activate_fci.py" C-m
    tmux send-keys -t $SESSION "sshpass -p kovan123 ssh panda2@10.0.0.3" C-m
    tmux send-keys -t $SESSION "sudo /home/panda2/droid/scripts/bringup/nuc_bringup.sh" C-m
    tmux send-keys -t $SESSION "kovan123" C-m

    tmux split-window -h -t $SESSION
    sleep 2

    tmux send-keys -t $SESSION "sudo /home/robot/droid/scripts/bringup/laptop_bringup.sh" C-m
    tmux send-keys -t $SESSION "teleoperasyon" C-m   
    
fi
# Attach to the tmux session
tmux set -g mouse on 

tmux bind -n C-r kill-session

tmux attach-session -t $SESSION
