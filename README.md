# Optimize_AVSS

This repository contains implementations for audio-visual speech synthesis and voice conversion using deep learning models.

---

## 🛠 Installation

To install all required dependencies:

```bash
pip install -r requirements.txt
Note: This project uses the VoxCeleb2 and LRS3-TED datasets.

# Training
To train the voice conversion model, run:
python train_audio.py --data_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL

To train the audio-visual synthesis model, run:
python train_audiovisual.py --video_path PATH_TO_TRAINING_DATA --experiment_name EXPERIMENT_NAME --save_freq SAVE_FREQ --test_path PATH_TO_TEST_AUDIO --batch_size BATCH_SIZE --save_dir PATH_TO_SAVE_MODEL --use_256 --load_model LOAD_MODEL_PATH

To convert an input audio file using a trained model:
python test_audio.py --model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT

To synthesize audio-visual outputs using a trained model:
python test_audiovisual.py --load_model PATH_TO_MODEL --wav_path PATH_TO_INPUT --output_file PATH_TO_OUTPUT --use_256

We provide an additional utility script for automatic hyperparameter tuning using Taguchi Design of Experiments and Bayesian Optimization.
python optimize_avss.py
