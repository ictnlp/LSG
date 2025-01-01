# LSG

**Authors**: [Shoutao Guo](https://scholar.google.com/citations?user=XwHtPyAAAAAJ&hl=zh-CN), [Shaolei Zhang](https://scholar.google.com/citations?user=gWwAWo4AAAAJ&hl=zh-CN), [Zhengrui Ma](https://scholar.google.com/citations?user=dUgq6tEAAAAJ&hl=zh-CN), [Yang Feng](https://yangfengyf.github.io/)*

Code for AAAI 2025 paper "Large Language Models Are Read/Write Policy-Makers for Simultaneous Generation"

💡Highlight:
1. LSG is a **L**LM-driven **S**imultaneous **G**eneration framework, which allows the off-the-shelf LLMs to decide the generation timing and produce output concurrently.
2. Experiments on simultaneous text-to-text translation and speech-to-text translation demonstrates LSG achieves SOTA performance on standard benchmarks.
3. LSG shows robust performance on streaming ASR task.

## 🚀Quick Start

### 1. Requirements and Installation

* Python version = 3.11.9

* PyTorch version = 2.2.1

* Transformers version = 4.32.0

* Install our library:

```
git clone https://github.com/ictnlp/LSG
cd LSG
pip install -e .
```

### 2. Download Models

#### Text-to-text Translation
We keep settings with [Agent-SiMT](https://arxiv.org/abs/2406.06910). We use [Llama2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) as the base model and fine-tune it by sampling 50k samples from WMT15 German-English (download [here](https://www.statmt.org/wmt15)) and MusT-C English-German dataset (download [here](https://mt.fbk.eu/must-c/)). The detailed fine-tuning scripts can be found [here](https://github.com/ictnlp/SiLLM).

#### Speech-to-text Translation and Streaming ASR

We directly use off-the-shelf [Qwen-Audio](https://github.com/QwenLM/Qwen-Audio) model for speech input.

### 3. Prepare Inference Data

We prepare the test data following [SimulEval](https://github.com/facebookresearch/SimulEval) format.

[source_audio.txt](https://github.com/ictnlp/LSG/blob/main/qwen_audio_st/translation_file/source_audio.txt): Each line records the path of a source speech.
[target.txt](https://github.com/ictnlp/LSG/blob/main/qwen_audio_st/translation_file/target.txt): Each line records the reference text, e.g., target translation or source transcription (used to calculate the BLEU or WER metrics).

### 4. Inference with SimulEval

Run the following scripts to performance evaluation. We provide the inference scripts for simultaneous speech-to-text translation and streaming ASR.

#### Simultaneous Speech-to-Text Translation
We prepare the inference scripts in the [eval_contrastive_policy.sh](https://github.com/ictnlp/LSG/blob/main/qwen_audio_st/eval_contrastive_policy.sh).

```
export CUDA_VISIBLE_DEVICES=0,1

DELTA=delta
ALPHA=alpha
LOW_BOUND=low_bound
TOP_BOUND=top_bound
SEG_SIZE=640
MODEL=qwen_audio_dir
SOURCE=translation_file/source_audio.txt
TARGET=translation_file/target.txt

simuleval --agent contrastive_policy.py \
    --source-segment-size $SEG_SIZE \
    --source_size $SEG_SIZE \
    --source $SOURCE \
    --target $TARGET \
    --threshold $ALPHA \
    --low_bound $LOW_BOUND \
    --top_bound $TOP_BOUND \
    --decision_ratio $DELTA \
    --lang_pair fr_en \
    --model_dir $MODEL \
    --output result_log_${SEG_SIZE}_${LOW_BOUND}_${TOP_BOUND}_${DELTA}_${ALPHA}

```


#### Streaming ASR
We prepare the inference scripts in the [eval_contrastive_policy_asr.sh](https://github.com/ictnlp/LSG/blob/main/qwen_audio_asr/eval_contrastive_policy_asr.sh).

```
export CUDA_VISIBLE_DEVICES=0,1

DELTA=delta
ALPHA=alpha
LOW_BOUND=low_bound
TOP_BOUND=top_bound
SEG_SIZE=640
MODEL=qwen_audio_dir
SOURCE=translation_file/source_audio.txt
TARGET=translation_file/target.txt


simuleval --agent contrastive_policy_asr.py \
    --source-segment-size $SEG_SIZE \
    --source_size $SEG_SIZE \
    --source $SOURCE \
    --target $TARGET \
    --threshold $ALPHA \
    --low_bound $LOW_BOUND \
    --top_bound $TOP_BOUND \
    --decision_ratio $DELTA \
    --lang_pair fr_fr \
    --quality-metrics WER \
    --model_dir $MODEL \
    --output result_log_${SEG_SIZE}_${LOW_BOUND}_${TOP_BOUND}_${DELTA}_${ALPHA}
```

### 🖋Citation

If you have any questions, please feel free to submit an issue or contact guoshoutao22z@ict.ac.cn.

If our work is useful for you, please cite as:
```
@article{lsg_ictnlp,
      title={Large Language Models Are Read/Write Policy-Makers for Simultaneous Generation}, 
      author={Shoutao Guo and Shaolei Zhang and Zhengrui Ma and Yang Feng},
      year={2025},
      journal={Proceedings of the AAAI Conference on Artificial Intelligence}
}
```
