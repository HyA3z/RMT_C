This repository contains code for reproducing the Recurrent Memory Transformer experiments for the paper:

## In Search of Needles in a 11M Haystack: Recurrent Memory Finds What LLMs Miss

## RMT resources

[ [paper](https://arxiv.org/abs/2207.06881) ] [ [code](https://github.com/booydar/recurrent-memory-transformer/) ] **Recurrent Memory Transformer** implementation and various training examples.

[ [paper](https://arxiv.org/abs/2402.10790) ] [ [code](https://github.com/booydar/recurrent-memory-transformer/tree/babilong-release) ] **BABILong** - a long-context benchmark that supports 20 tasks and various sources of background text. 

[ [code](https://github.com/booydar/babilong) ] **Evaluate your long-context LLM on BABILong!**

RMT is a memory-augmented segment-level recurrent Transformer. We implement our memory mechanism as a wrapper for any Hugging Face model by adding special memory tokens to the input sequence. The model is trained to control both memory operations and sequence representations processing.

<img src="img/RMT_scheme.png" alt="drawing" width="400"/>

## Installation
```bash
pip install -e .
```
This command will install `lm_experiments_tools` with only required packages for Trainer and tools.

`lm_experiments_tools` Trainer supports gradient accumulation, logging to tensorboard, saving the best models
based on metrics, custom metrics and data transformations support.

### Install requirements for all experiments
Full requirements for all experiments are specified in requirements.txt. Install requirements after cloning the repo:
```bash
pip install -r requirements.txt
```

### Loading the dataset
Dataset can be found in the `data` directory. To unpack it, run 
```bash
unzip data/tasks_1-20_v1-2.zip
```

Another option is to [download](https://huggingface.co/datasets/facebook/babi_qa) it from Hugging Face.

Dataset usage example can be found in `notebooks/babilong_usage_example.ipynb`

### Running experiments
For reproducing experiments please use the `scripts_exp/babilong directory`. Bash scripts contain hyperparameters and commands to run the experiments. 

Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NP=4 ./finetune_babilong_qa1_rmt_vary_n_seg.sh
```

Don't forget to pass the path to your dataset folder using the `--babi_path` variable. 

### Citation
If you find our work useful, please consider citing the RMT papers:

```
@misc{kuratov2024search,
      title={In Search of Needles in a 11M Haystack: Recurrent Memory Finds What LLMs Miss}, 
      author={Yuri Kuratov and Aydar Bulatov and Petr Anokhin and Dmitry Sorokin and Artyom Sorokin and Mikhail Burtsev},
      year={2024},
      eprint={2402.10790},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```
@inproceedings{
        bulatov2022recurrent,
        title={Recurrent Memory Transformer},
        author={Aydar Bulatov and Yuri Kuratov and Mikhail Burtsev},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=Uynr3iPhksa}
}
```