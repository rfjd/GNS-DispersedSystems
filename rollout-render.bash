DATA_PATH="data/multidisperse-sedimentation/"
MODEL_PATH="RUNS/multidisperse-sedimentation/models/"
number_steps=2000000
testfilename="test"
ROLLOUT_PATH="RUNS/multidisperse-sedimentation/rollout/"
echo "DATA_PATH: ${DATA_PATH}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "ROLLOUT_PATH: ${ROLLOUT_PATH}"

mkdir -p ${ROLLOUT_PATH}

# Rollout Prediction
python3 -m gns.main --mode="rollout" --data_path="${DATA_PATH}" --model_path="${MODEL_PATH}" --output_path="${ROLLOUT_PATH}" --model_file="model-${number_steps}.pt" --train_state_file="train_state-${number_steps}.pt"

# # Renderer
# for case in $(seq 0 5); do
# 	python3 -m gns.render_rollout_particles2D --rollout_dir="${ROLLOUT_PATH}" --rollout_name="rollout_ex${case}" --step_stride=3
# 	mv ${ROLLOUT_PATH}/rollout_ex$case.gif ${ROLLOUT_PATH}/rollout_ex$case-${number_steps}.gif
# done
