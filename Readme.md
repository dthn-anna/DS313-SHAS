# DS313-SHAS

The following is a guide to reproducing the study **"SHAS - Approaching Optimal Segmentation for End-to-End Speech Translation"**, using two training approaches — local training and training in cloud-based environments such as Colab and Kaggle.

## 📁 Data Preparation

### ✅ Step 1. Set up environment

```bash
export SHAS_ROOT=D:/SHAS/SHAS_ROOT
export MUSTC_ROOT=D:/SHAS/MUSTC_ROOT
export MTEDX_ROOT="/d/SHAS/MTEDX_ROOT"
export SEGM_DATASETS_ROOT=D:/SHAS/SEGM_DATASETS_ROOT
export ST_MODELS_PATH=D:/SHAS/ST_MODELS_PATH
export RESULTS_ROOT=D:/SHAS/RESULTS_ROOT
export FAIRSEQ_ROOT=D:/SHAS/FAIRSEQ_ROOT
export MWERSEGMENTER_ROOT=D:/SHAS/MWERSEGMENTER_ROOT
```

### ✅ Step 2. Clone SHAS repo

```bash
git clone https://github.com/mt-upc/SHAS.git ${SHAS_ROOT}
```

### ✅ Step 3. Create Conda environment (Python 2.7)

```bash
conda create -n snakes27 python=2.7
conda activate snakes27
```

### ✅ Step 4. Download mwerSegmenter tool

```bash
mkdir -p $MWERSEGMENTER_ROOT
curl -O  https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
tar -zxvf mwerSegmenter.tar.gz -C ${MWERSEGMENTER_ROOT}
rm -r mwerSegmenter.tar.gz
```

### ✅ Step 5. Create Conda environment (Python 3.9)

```bash
conda create -n shas python=3.9.6
conda activate shas
```

### ✅ Step 6. Install dependencies

```bash
conda install -c pytorch pytorch=1.10.0 torchaudio=0.10.0 cudatoolkit=10.2.89
conda install -c conda-forge pandas=1.3.3 tqdm=4.62.3 numpy=1.22.1 transformers=4.11.3 pip=21.2.4

pip install sacrebleu==1.5.0 sacremoses==0.0.46 webrtcvad==2.0.10 pydub==0.25.1 \
            wandb==0.12.9 SoundFile==0.10.3.post1 PyYAML==6.0 scikit_learn==1.0.2 \
            tweepy==4.5.0 sentencepiece==0.1.96

pip install numpy==1.22.1 torch==1.10.0 torchaudio==0.10.0 \
            pip==21.2.4 "protobuf<=3.20.3"
```

### ✅ Step 7. Clone Fairseq module

```bash
git clone -b shas https://github.com/mt-upc/fairseq.git ${FAIRSEQ_ROOT}
conda activate shas
conda install pip
pip install "pip<24.1"
conda install python=3.9.6
pip install --editable D:/SHAS/FAIRSEQ_ROOT
```

### ✅ Step 8. Download mTEDx dataset

```bash
mkdir -p "${MTEDX_ROOT}/log_dir"
for lang_pair in it-en it; do
  curl -L "https://www.openslr.org/resources/100/mtedx_${lang_pair}.tgz" \
    -o "${MTEDX_ROOT}/log_dir/${lang_pair}.tgz"
  tar -xzf "${MTEDX_ROOT}/log_dir/${lang_pair}.tgz" -C "${MTEDX_ROOT}"
done
```

### ✅ Step 9. Convert audio format

```bash
cd D:/SHAS
conda install -c conda-forge parallel ffmpeg
ls ${MTEDX_ROOT}/*/data/{train,valid,test}/wav/*.flac | \
  parallel -j 12 ffmpeg -i {} -ac 1 -ar 16000 -hide_banner -loglevel error {.}.wav
```

### ✅ Step 10. Data Segmentation

```bash
for lang_pair in {it-en,it-it}; do
  mkdir -p ${SEGM_DATASETS_ROOT}/mTEDx/${lang_pair}
  for split in {train,valid,test}; do
    python ${SHAS_ROOT}/src/data_prep/prepare_dataset_for_segmentation.py \
      -y ${MTEDX_ROOT}/${lang_pair}/data/${split}/txt/${split}.yaml \
      -w ${MTEDX_ROOT}/${lang_pair}/data/${split}/wav \
      -o ${SEGM_DATASETS_ROOT}/mTEDx/${lang_pair}
  done
done
```

### ✅ Step 11. Download S2T model

```bash
mult_model_path=${ST_MODELS_PATH}/joint-s2t-multilingual
mkdir -p $mult_model_path
cd "$mult_model_path"

for file in {checkpoint17.pt,config.yaml,tgt_dict.txt,dict.txt,spm.model}; do
  curl -O https://dl.fbaipublicfiles.com/joint_speech_text_4_s2t/iwslt/iwslt_data/${file}  
done

sed -i "s+/path/spm.model+${mult_model_path}/spm.model+" ${mult_model_path}/config.yaml
python ${SHAS_ROOT}/src/data_prep/fix_joint_s2t_cfg.py -c ${mult_model_path}/checkpoint17.pt
```

---

## 📊 Training SFC Segmentation Model

### ✅ Step 1. Export environment variables (WSL)

```bash
export SHAS_ROOT="/mnt/d/SHAS/SHAS_ROOT" export MUSTC_ROOT="/mnt/d/SHAS/MUSTC_ROOT" export MTEDX_ROOT="/mnt/d/SHAS/MTEDX_ROOT" export SEGM_DATASETS_ROOT="/mnt/d/SHAS/SEGM_DATASETS_ROOT" export ST_MODELS_PATH="/mnt/d/SHAS/ST_MODELS_PATH" export RESULTS_ROOT="/mnt/d/SHAS/RESULTS_ROOT" export FAIRSEQ_ROOT="/mnt/d/SHAS/FAIRSEQ_ROOT" export MWERSEGMENTER_ROOT="/mnt/d/SHAS/MWERSEGMENTER_ROOT" export path_to_wavs="/mnt/d/SHAS/MTEDX_ROOT/it-en/data/test/wav" export path_to_checkpoint="/mnt/d/SHAS/RESULTS_ROOT/supervised_hybrid/mult_sfc_model/ckpts/step-1682.pt" export path_to_custom_segmentation_yaml="/mnt/d/SHAS/output/custom_segments.yaml" export max_segment_length=14 export path_to_original_segmentation_yaml="/mnt/d/SHAS/MTEDX_ROOT/it-en/data/test/txt/test.yaml" export path_to_original_segment_transcriptions="/mnt/d/SHAS/MTEDX_ROOT/it-en/data/test/txt/test.it" export path_to_original_segment_translations="/mnt/d/SHAS/MTEDX_ROOT/it-en/data/test/txt/test.en" export src_lang=it export tgt_lang=en export path_to_st_model_ckpt="/mnt/d/SHAS/ST_MODELS_PATH/joint-s2t-multilingual/checkpoint17.pt" export PYTHONPATH="/mnt/d/SHAS/FAIRSEQ_ROOT"
```

### ✅ Step 2. Fix torchaudio bugs

- In file `data.py`, after imports:

```python
torchaudio.set_audio_backend("soundfile")
```

- Replace `torchaudio.backend.sox_io_backend.load(...)` with:

```python
torchaudio.load(...)
```

### ✅ Step 3. Training

```bash
experiment_name=mult_sfc_model
python ${SHAS_ROOT}/src/supervised_hybrid/train.py \
    --datasets ${SEGM_DATASETS_ROOT}/mTEDx/it-it \
    --results_path ${RESULTS_ROOT}/supervised_hybrid \
    --model_name facebook/wav2vec2-xls-r-300m \
    --experiment_name $experiment_name \
    --train_sets train \
    --eval_sets valid \
    --batch_size 16 \
    --learning_rate 2.5e-4 \
    --update_freq 20 \
    --max_epochs 8 \
    --classifier_n_transformer_layers 2 \
    --wav2vec_keep_layers 15
```

### ✅ Step 4. Conda env in WSL

```bash
conda create -n shas_wsl python=3.9.6
```

### ✅ Step 5. Install libraries (WSL)

```bash
conda install -c pytorch pytorch=1.10.0 torchaudio=0.10.0 cudatoolkit=10.2.89 conda install -c conda-forge pandas=1.3.3 tqdm=4.62.3 numpy=1.22.1 transformers=4.11.3 pip=21.2.4

pip install sacrebleu==1.5.0 sacremoses==0.0.46 webrtcvad==2.0.10 pydub==0.25.1 wandb==0.12.9 SoundFile==0.10.3.post1 PyYAML==6.0 scikit_learn==1.0.2 tweepy==4.5.0 sentencepiece==0.1.96

pip install "protobuf<=3.20.3"
```

### ✅ Step 6. Install Fairseq (WSL)

```bash
conda install pip
pip install "pip<24.1"
cd FAIRSEQ_ROOT
pip install --editable .
```

### ✅ Step 7. Segment test set

```bash
bash ${SHAS_ROOT}/src/eval_scripts/eval_custom_segmentation.sh \
  $path_to_wavs \
  $path_to_custom_segmentation_yaml \
  $path_to_original_segmentation_yaml \
  $path_to_original_segment_transcriptions \
  $path_to_original_segment_translations \
  $src_lang \
  $tgt_lang \
  $path_to_st_model_ckpt
```

After completing this step, the file `custom_segment.yaml` will be generated in your output directory. Together with the test audio files from mTEDx, this YAML file will serve as input to the translation pipeline on Kaggle using the provided notebook `shas-translation-eval.ipynb`.

Once speech translation is complete, you'll use four output files — `test.it.xlm`, `test.en.xlm`, `translations.txt`, and `translation_formatted.txt`  — to perform local evaluation via the WSL terminal.

---

## 🎯 Evaluate Translation Results

### ✅ Step 1. Set env vars in WSL

```bash
# similar to Step 1 above
```

### ✅ Step 2. Align and Score

```bash
eval "$(conda shell.bash hook)"
conda activate snakes27

bash ${MWERSEGMENTER_ROOT}/segmentBasedOnMWER.sh \
    ${working_dir}/${split_name}.${src_lang}.xml \
    ${working_dir}/${split_name}.${tgt_lang}.xml \
    ${working_dir}/translations_formatted.txt \
    $st_model_basename \
    $tgt_lang \
    ${working_dir}/translations_aligned.xml \
    normalize \
    1

eval "$(conda shell.bash hook)"
conda activate shas_wsl
python ${SHAS_ROOT}/src/eval_scripts/score_translation.py $working_dir
```

---

## ☁️ Cloud Training

In addition to local CPU training, SHAS also supports cloud-based training (e.g., Colab, Kaggle). See notebook:

`SHAS-Train.ipynb.`

Then perform translation + evaluation just like the local setup using the generated `custom_segments.yaml`.

