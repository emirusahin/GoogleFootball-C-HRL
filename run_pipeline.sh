#!/usr/bin/env bash
# run_pipeline.sh — fully detached TPU training pipeline

# 1) Configuration: edit these to suit your project
BUCKET="gs://my-football-bucket"
EXP_NAME="exp_full_run"
SAVE_PATH="$BUCKET/checkpoints/$EXP_NAME"
RESULTS_PATH="$BUCKET/results/$EXP_NAME"
TMUX_SESSION="football_train"
VENV_DIR="$HOME/tpu-env"

# 2) Ensure bucket exists (no-op if already there)
echo ">> Ensuring GCS bucket exists: $BUCKET"
gsutil ls $BUCKET &>/dev/null || gsutil mb $BUCKET

# 3) Activate your Python venv
echo ">> Activating venv at $VENV_DIR"
source "$VENV_DIR/bin/activate"

# 4) Kill any existing session, then start a fresh detached tmux
echo ">> Starting tmux session: $TMUX_SESSION"
tmux kill-session -t $TMUX_SESSION 2>/dev/null
tmux new-session -d -s $TMUX_SESSION

# 5) Build the training command
TRAIN_CMD="python3 train_tpu.py \
  --num_actors=16 \
  --unroll_length=512 \
  --train_epochs=4 \
  --train_minibatches=8 \
  --clip_range=0.08 \
  --gamma=0.993 \
  --gae_lambda=0.95 \
  --entropy_coef=0.003 \
  --value_function_coef=0.5 \
  --grad_norm_clip=0.64 \
  --learning_rate=0.000343 \
  --max_steps=5000000 \
  --early_stop_episodes=200 \
  --early_stop_reward=1.9 \
  --num_cores=8 \
  --save_path=\"$SAVE_PATH\" \
  --results_path=\"$RESULTS_PATH\" \
  > train.log 2>&1"

# 6) Send it into tmux
tmux send-keys -t $TMUX_SESSION "$TRAIN_CMD" C-m

# 7) Print next steps
echo ">> Training launched in tmux session '$TMUX_SESSION'."
echo "   Logs are being written to train.log"
echo "   Checkpoints → $SAVE_PATH"
echo "   Metrics       → $RESULTS_PATH"
echo
echo "To monitor live logs, run:"
echo "  tmux attach -t $TMUX_SESSION"
echo
echo "To detach again: Ctrl-B D"
