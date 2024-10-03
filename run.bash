DIR="RUNS"
C=6
NUM_MESSAGE_PASSING_STEPS=10
NUM_ENCODED_NODE_FEATURES=128
NUM_ENCODED_EDGE_FEATURES=128
MLP_LAYER_SIZE=128
number_steps=2000000

DATASET_NAME="SpheresBox-Multidisperse-DENSE-16"

DATA_PATH="data/$DATASET_NAME/"
FLAGS="--C=$C --NUM_MESSAGE_PASSING_STEPS=$NUM_MESSAGE_PASSING_STEPS --NUM_ENCODED_NODE_FEATURES=$NUM_ENCODED_NODE_FEATURES --NUM_ENCODED_EDGE_FEATURES=$NUM_ENCODED_EDGE_FEATURES --MLP_LAYER_SIZE=$MLP_LAYER_SIZE"

# OUT_DIR=$(echo $FLAGS | awk '{gsub(/ /, ""); gsub(/--/,"-"); print}')
# OUT_DIR="$DIR/$DATASET_NAME-$OUT_DIR/"
# MODEL_PATH="$OUT_DIR/models/"
# ROLLOUT_PATH="$OUT_DIR/rollout/"
MODEL_PATH="$DIR/$DATASET_NAME/models/"
ROLLOUT_PATH="$DIR/$DATASET_NAME/rollout/"

mkdir -p ${MODEL_PATH}
mkdir -p ${ROLLOUT_PATH}

# Train
python3 -m gns.main --data_path="${DATA_PATH}" --model_path="${MODEL_PATH}" --ntraining_steps=$number_steps $FLAGS

# Rollout Prediction
python3 -m gns.main --mode="rollout" --data_path="${DATA_PATH}" --model_path="${MODEL_PATH}" --output_path="${ROLLOUT_PATH}" --model_file="model-${number_steps}.pt" --train_state_file="train_state-${number_steps}.pt" $FLAGS

# Renderer
# cases="0 1 2 3 4 5"
for case in $(seq 0 5); do
	python3 -m gns.render_rollout_particles2D --rollout_dir="${ROLLOUT_PATH}" --rollout_name="rollout_ex${case}" --step_stride=3
	mv ${ROLLOUT_PATH}/rollout_ex$case.gif ${ROLLOUT_PATH}/rollout_ex$case-${number_steps}.gif
done

cp run.bash "$DIR/$DATASET_NAME/"
cp gns/main.py "$DIR/$DATASET_NAME/"
