## Require

- V100
- CUDA 11.3, cudnn 8.3
- Python 3.8
- torch 11.1

## Install

```pip install ./```

## Getting Started

1. Go to a specific directory, such as ```examples/pytorch/translation```
   1. All experiments are based on hugging-face's examples.
   2. Mimose's experiments include [multiple-choice](https://github.com/mimose-project/mimose-transformers/tree/mimose/examples/pytorch/multiple-choice), [question-answering](https://github.com/mimose-project/mimose-transformers/tree/mimose/examples/pytorch/question-answering), [text-classification](https://github.com/mimose-project/mimose-transformers/tree/mimose/examples/pytorch/text-classification), [translation](https://github.com/mimose-project/mimose-transformers/tree/mimose/examples/pytorch/translation)
2. run script, ```sh run_t5_base_un_4000_dc.sh x y z```. Then, get the log file under `./log`.
   1. Where ```x``` represents the size of the video memory that can be used by the process, ```y``` represents the memory size for the memory fragmentation, and ```z``` represents the number of warmup iterations. The unit of ```x``` and ```y``` is GB.
   2. For other experiments, the corresponding scripts are as follows:
      1. multiple-choice: ```sh run_swag_roberta_dc.sh x```
      2. question-answering: ```sh run_qa_womax_dc.sh x``` and ```sh run_qa_xlnet_dc.sh x```
      3. text-classification: ```sh run_dc.sh x```
3. Go to directory ```plot_script```, change the file path to the latest.
4. run ```python3 plot_figure_10_a_d.py ``` to get figure 10.
5. run ```python3 plot_figure_11.py``` to get figure 11.

