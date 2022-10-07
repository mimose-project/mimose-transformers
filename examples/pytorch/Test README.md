## Setup

- Task：multiple-choice、question-answering、text-classification、token-classification、object-detection
- GPU：V100、A100

> 通过对Trainer的部分函数进行修改，实现对输入的seq length大小和相应的torch.cuda.max_memory_allocated()（在backward结束时）数据收集
>
> 显存的收集方法后期会进行调整，以避免mb15上部分tensor过大导致的峰值偏移问题。
>
> 同时还对save_checkpoint相关函数进行修改，应该不会在训练中保存model的checkpoint

## Base Script

> 需要注意增加 `--num_train_epochs`, 因为大部分默认情况下并不是1
>
> 在大部分任务的README中，会对seq length进行padding或者限制max的情况，这里是已经全部去除
>
> 如果要限制memory使用，可以增加```--memory_threshold xx```，单位GB，可以自动设置"set_per_process_memory_fraction"的参数。

### multiple-choice

```bash
python3 run_swag.py \
  --model_name_or_path bert-base-uncased \
  --output_dir /tmp/test-swag-no-trainer \
  --do_train \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --memory_threshold 8
```

### question-answering

```bash
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --output_dir /tmp/test_squad/ \
  --doc_stride 128 \
  --num_train_epochs 1 \
  --memory_threshold 8
```

### text-classification

```bash
python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --output_dir /tmp/$TASK_NAME/ \
  --num_train_epochs 1 \
  --memory_threshold 8
```

### token-classification

```bash
python3 run_ner.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name conll2003 \
  --output_dir /tmp/test-ner \
  --do_train \
  --num_train_epochs 1 \
  --per_device_train_batch_size 64 \
  --memory_threshold 8
```

## 测试范围

> 通过`tee`命令收集输出内容，不同input size下的显存占用会在每个iter中通过`torch.cuda.max_memory_allocated()`进行收集，然后使用`torch.cuda.reset_peak_memory_stats()`重置统计数据。在训练结束后dump出来，后续会有专门的脚本解析log。
>
> gc：对backbone中的每个`block`对象使用checkpoint。
>
> regular：不使用任何显存优化方法
>
> mb15测试：需要考虑是否使用light seq，如果可以两个版本都测一下。

**重要提醒**：

1. 可以先拿一些任务在不同的GPU（型号不变）上进行测试，测试GPU体质对速度的影响；如果影响巨大，则需要固定某些GPU上进行测试。
2. 需要考虑温度对GPU速度的影响，统一在测试开始前sleep一段时间（降温），或者跑几个benchmark（升温）

| task                 | dataset    | model             | GPU        | epoch | batch size | optim          | work         | memory budget(GB)                               | 结果                              |
| -------------------- | ---------- | ----------------- | ---------- | ----- | ---------- | -------------- | ------------ | ----------------------------------------------- | --------------------------------- |
| multi-choice         | swag       | bert-base-uncased | V100-32 GB | 1     | 16         | default        | dtr          | 3-8(间隔1)                                      | 含有`memory_count`关键字的log文件 |
| question-answering   | SQuAD      | bert-base-uncased | V100-32 GB | 1     | 12         | default        | dtr          | 3-8(间隔1)                                      | 含有`memory_count`关键字的log文件 |
| text-classification  | GLUE       | bert-base-cased   | V100-32 GB | 1     | 32         | default        | dtr          | 3-8(间隔1)                                      | 含有`memory_count`关键字的log文件 |
| token-classification | CoNLL NER  | bert-base-uncased | V100-32 GB | 1     | 64         | default        | dtr          | 3-8(间隔1)                                      | 含有`memory_count`关键字的log文件 |
|                      |            |                   |            |       |            |                |              |                                                 |                                   |
| object detection     | mscoco2017 | mb15              | A100-80 GB | 1     | 1          | FusedFP16AdamW | gc、dtr      | 50-80(间隔10)                                   | 含有`memory_count`关键字的log文件 |
| object detection     | mscoco2017 | swin-L            | V100-32 GB | 1     | 16         | FusedFP16AdamW | regular、dtr | 尚未详细测试，从gc的显存占用增长到regular的一半 | 含有`memory_count`关键字的log文件 |
| object detection     | mscoco2017 | swinV2-G          | A100-80 GB | 1     | 1          | FusedFP16AdamW | gc、dtr      | 尚未详细测试，从gc的显存占用增长到regular的一半 | 含有`memory_count`关键字的log文件 |

### 关键文件修改

NLP的主要文件在各个目录下的`trainer.py`中，通过重载原来的Trainer类来实现hook的功能。

其中比较重要的成员函数：

1. 初始化：`__init__(self, ...)`
2. forward & backward：`training_step(self, ...)`
3. update：在`train（self, ...)`中，关键词为`optimizer.step()`之类的

如果需要在原来的`Trainer`中进行修改，则直接访问`src/transformers/trainer.py`



CV的模型主要看使用的训练框架，这里暂不赘述

## 部分结果

> method 说明：
>
> - None：没有使用任何显存优化技术
> - DC-X：使用Dynamic Checkpoint，且 memory 上限为X GB
> - DCS-X：使用Dynamic Checkpoint，且 memory 上限为X GB。但是只是用最大输入下的 checkpoint scheme
> - PC-X：previously Checkpoint。使用 Dynamic Checkpoint 最大输入下的 checkpoint scheme，但是跳过了 warmup 阶段，直接修改对应的forward函数。memory 上限为X。此为分析 Dynamic Checkpoint 的 overhead
> - GC：gradient_checkpointing。会对 bert 中所有的 encoder 使用 checkpoint，此为Dynamic Checkpoint理论上的显存使用最小值。
>
> token-classification 好像当时嫌弃数据集太小，就忘了测。NLP论文中常见的数据集为GLUE、SQuAD、SWAG。
>
> 这些任务在默认配置下，甜品显存点大概为6-8 GB左右，即大部分输入很少使用checkpoint。

| task                 | epochs | model             | batch size | opti method | time(%H:%M:%S) |
| -------------------- | ------ | ----------------- | ---------- | ----------- | -------------- |
| multiple-choice      | 3      | bert-base-uncased | 16         | None        | 1:02:59        |
| multiple-choice      | 3      | bert-base-uncased | 16         | DC-6        | 1:05:47        |
| multiple-choice      | 3      | bert-base-uncased | 16         | DCS-6       | 1:18:03        |
| multiple-choice      | 3      | bert-base-uncased | 16         | PC-6        | 1:17:59        |
|                      |        |                   |            |             |                |
| question-answering   | 2      | bert-base-uncased | 12         | None        | 1:20:01        |
| question-answering   | 2      | bert-base-uncased | 12         | DC-8        | 1:21:15        |
| question-answering   | 2      | bert-base-uncased | 12         | DCS-8       | 1:31:49        |
| question-answering   | 2      | bert-base-uncased | 12         | GC          | 1:45:03        |
|                      |        |                   |            |             |                |
| text-classification  | 1      | bert-base-cased   | 32         | None        | 38:48          |
| text-classification  | 1      | bert-base-cased   | 32         | GC          | 50:30          |
|                      |        |                   |            |             |                |
| token-classification | 3      | bert-base-uncased | 64         | None        |                |
|                      |        |                   |            |             |                |
|                      |        |                   |            |             |                |

