DIR="RUNS"
DATASET_NAME="RotatingSpheres-monodisperse"

DATA_PATH="data/${DATASET_NAME}/"
MODEL_PATH="${DIR}/${DATASET_NAME}/models/"
ROLLOUT_PATH="${DIR}/${DATASET_NAME}/rollout/"

number_steps=1000000

mkdir -p ${MODEL_PATH}
mkdir -p ${ROLLOUT_PATH}

# Train
python3 -m gns-PyTorch.train --data_path="${DATA_PATH}" --model_path="${MODEL_PATH}" -ntraining_steps=$number_steps

# Rollout Prediction
python3 -m gns-Pytorch.train --mode="rollout" --data_path="${DATA_PATH}" --model_path="${MODEL_PATH}" --output_path="${ROLLOUT_PATH}" --model_file="model-${number_steps}.pt" --train_state_file="train_state-${number_steps}.pt"

case=0
# Renderer
python3 -m gns-Pytorch.render_rollout --rollout_dir="${ROLLOUT_PATH}" --rollout_name="rollout_ex${case}" --step_stride=3

mv ${ROLLOUT_PATH}/rollout_ex$case.gif ${ROLLOUT_PATH}/rollout_ex$case-${number_steps}.gif
