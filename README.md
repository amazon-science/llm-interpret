## Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale

This repository contains code to reproduce the experiments in the paper "[Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale](https://arxiv.org/abs/2212.09095)",
published in the main proceedings of ACL 2023.

## Setup

Set up and activate an initial conda environment using the provided `environment.yml` file.
```
conda env create -f environment.yml
conda activate opt
```

Install [PyTorch](https://pytorch.org/) based on your system configuration. We used the following with
AWS EC2 p4 instances:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Getting Started

Our code is based off 🤗Hugging Face's [transformers](https://github.com/huggingface/transformers)
and Eleuther AI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
libraries.

Run the following sequence of commands (in that order) to clone and set up both libraries in your file
system. We point to particular hashes associated with our runs, but it may be possible that our code is
forward-compatible with newer versions.

```
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 11fa0bf4394998634e6c6e0c9fc2fc8211415042
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 9832ac7c736519fcfeedb88c8368cf0ab08b2b58
```

### Changes to 🤗Transformers

We modified the implementation of the Open Pre-Trained Transformer (OPT) in 🤗Transformers to allow
for importance score computations. Specifically:
1. we use hooks to store the gradient of the loss w.r.t. the output of attention heads (see `context_layer_val` and `context_layer_val_grad`)
2. we define masks to "knock-off" particular feed forward networks (see `fc_mask` and `layer_fc_mask`)

The modified implementation is located at [transformers/models/opt/modeling_opt.py](transformers/models/opt/modeling_opt.py)
in this repo.

Copy this script to the corresponding location for OPT in the local clone of `transformers`.

### Changes to `lm-evaluation-harness`

We added support for OPT in `lm-evaluation-harness` following the existing example for GPT-2,
see [lm_eval/models/opt.py](lm_eval/models/opt.py). This utilizes the core modifications to OPT in the
local clone of `transformers` described above. We used a custom device map to shard the model
parameters for our compute capacity, which can be modified according to one's own compute resourcing.

We also adapted other existing scripts from `lm-evaluation-harness` in the `lm_eval` directory:
1. [lm_eval/base.py](lm_eval/base.py) has the core logic of computing attention head importance scores,
see the `calculate_importance()` method.
2. [lm_eval/evaluator.py](lm_eval/evaluator.py) contains the code-flow to allow for original evaluation
as well as attention head importance score computation. The computed head importance scores are dumped
in pickle files.
3. [lm_eval/utils.py](lm_eval/utils.py) contains methods for dataset and data loader creation used
for attention head importance score computation, see the `create_dataloader()` and
`get_dataloader_from_dataset()` methods.
4. Each task defined in [lm_eval/tasks/](lm_eval/tasks/) is updated to create the associated data
loader via `utils.py` as described above and define a getter method for the data loader, see the
`get_dataloader()` method.

The driver script `main.py` is also adapted to allow these changes to be leveraged. Note that this
script dumps the evaluation results into JSON-formatted text files, which are necessary to create some plots in
our paper.

Copy these scripts to their corresponding locations in the local clone of `lm-evaluation-harness`.


### Induction Heads: Prefix Matching and Copying
[lm_eval/prefix_matching_copying.py](lm_eval/prefix_matching_copying.py) contains our
implementation for computing prefix matching and copying scores for attention heads,
also described in detail with pseudocode in our paper's Appendix. The original
algorithm by Anthropic is described in the Additional Details section of the Transformer Circuits Thread post [here](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#data-collection).
Please refer to our paper's Appendix for a description of the modifications we made
to their algorithm.

Copy this script to the `lm_eval` directory in the local clone of `lm-evaluation-harness`.

### Plotting
We provide the scripts used to create the plots in our paper in the
[scripts/](scripts/) directory. These scripts assume that the importance scores
are already computed and dumped in pickle files and the task-specific evaluation
results are dumped in JSON-formatted text files using the code described above.

Note that you may have to edit these scripts a bit according to the naming convention
you adopt for the importance score pickle and evaluation result text files you create.


## Sample Commands
In this section, we provide sample commands leveraging the code described above
for a few use-cases. We recommend diving into the code and understanding the
supported args to be able to leverage all supported functionality.

### Model and Tokenizer Caching
Load the pre-trained model and tokenizer into explicitly defined cache directories
as a one-time operation:
```
cd lm-evaluation-harness
python
>>> from transformers import AutoModel, AutoTokenizer
>>> model = AutoModel.from_pretrained('facebook/opt-66b', cache_dir='opt66b_checkpoints/')
>>> tokenizer = AutoTokenizer.from_pretrained('facebook/opt-66b', cache_dir='opt66b_tokenizer/')
```

### Attention Head Importance Scores
The following command computes and saves attention head importance scores for the
Physical IQA (PIQA) task in the 1-shot setting:
```
python main.py --model opt --model_args pretrained=facebook/opt-66b,model_cache_dir=opt66b_checkpoints,tokenizer_cache_dir=opt66b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt66b/1shot_piqa.pkl --num_fewshot 1
```

### Masking A Feed Forward Network

To mask a particular feed forward network (FFN) and evaluate the model on a
particular task, the following sample command can be used. OPT has 64 layers and
in this case, we are masking the FFN in layer 10 (indexing starting from 0) when
evaluating the model on the PIQA task in the 5-shot setting.

```
python main.py --model opt --model_args pretrained=facebook/opt-66b,model_cache_dir=opt66b_checkpoints,tokenizer_cache_dir=opt66b_tokenizer,mask_fc=10 --tasks piqa --output_path results/66b/5shot_fc_pruning/piqa/5shot_fc_10.txt --batch_size 2 --num_fewshot 5
```

### Iterative Pruning of Attention Heads

To mask unimportant attention heads and evaluate the model on a particular task,
the following sample command can be used. In this case, we are masking 20% (range: 0-90%)
of the task and shot-specific unimportant attention heads and evaluating the model
on the PIQA task in the 1-shot setting.

```
python main.py --model opt --model_args pretrained=facebook/opt-66b,model_cache_dir=opt66b_checkpoints,tokenizer_cache_dir=opt66b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt66b/1shot_piqa.pkl,head_percent_mask=20 --tasks piqa --output_path results/66b/piqa/1shot_piqa_percent.txt --batch_size 2 --num_fewshot 1
```

### FFN Importance Scores

The following command leverages `fc_importance.py`, which computes importance
scores for each FFN as the difference between the baseline accuracy and the
accuracy after masking the FFN for each task, and dumps them to pickle files.
The accuracy upon independently masking each FFN is assumed to have already been
computed as described above with an earlier sample command.

```
python scripts/plotting/fc_importance.py --results_path results/66b/5shot_fc_pruning/ --base_results_path results/66b/ --shot 5-shot --save_plot_path paper_plots/fc_importance/5-shot.png --dump_fc_importance --dump_fc_importance_path logs/fc_knocking_importance/
```

### Iterative Pruning of FFNs

To mask unimportant FFNs and evaluate the model on a particular task, the following
sample command can be used. In this case, we are masking 20% (range: 0-90%) of the
task and shot-specific unimportant FFNs and evaluating the model on the PIQA task
in the 5-shot setting.

```
python main.py --model opt --model_args pretrained=facebook/opt-66b,model_cache_dir=opt66b_checkpoints,tokenizer_cache_dir=opt66b_tokenizer,mask_iterative_fc=1,fc_importance_path=logs/fc_knocking_importance/5shot_piqa.pkl,fc_percent_mask=20 --tasks piqa --output_path results/66b/piqa/5shot_20_fc_percent.txt --batch_size 1 --num_fewshot 5
```

### Combined Pruning of Heads and FFNs

To evaluate the model on a particular task after combined pruning of attention heads
and FFNs, the following sample command can be used. In this case, we are masking
20% of the unimportant attention heads and 30% of the unimportant FFNs and evaluating
the model on the PIQA task in the 1-shot setting.

```
python main.py --model opt --model_args pretrained=facebook/opt-66b,model_cache_dir=opt66b_checkpoints,tokenizer_cache_dir=opt66b_tokenizer,mask_iterative_fc=1,fc_importance_path=logs/fc_knocking_importance/1shot_piqa.pkl,fc_percent_mask=30,mask_heads=1,head_importance_path=logs/head_importance/opt66b/1shot_piqa.pkl,head_percent_mask=20 --tasks piqa --output_path results/66b/piqa/1shot_30_fc_20_head_percent.txt --batch_size 2 --num_fewshot 1
```

### Prefix Matching and Copying

To compute, plot and save prefix matching and copying scores, the following pair of
sample commands can be used.

Prefix Matching:
```
python -m lm_eval.prefix_matching_copying --prefix_matching --pretrained facebook/opt-66b --model_cache_dir opt66b_checkpoints/ --tokenizer_cache_dir opt66b_tokenizer/ --save_plot_path_mean paper_plots/induction_heads/pfx_matching_mean.png --save_plot_path_var paper_plots/induction_heads/pfx_matching_var.png --save_outputs paper_plots/induction_heads/pfx_matching.pkl
```

Copying:
```
python -m lm_eval.prefix_matching_copying --copying_score --pretrained facebook/opt-66b --model_cache_dir opt66b_checkpoints/ --tokenizer_cache_dir opt66b_tokenizer/ --save_plot_path_mean paper_plots/induction_heads/copying_mean.png --save_plot_path_var paper_plots/induction_heads/copying_var.png --save_outputs paper_plots/induction_heads/copying.pkl
```


## Citation

If you find our work useful, please consider citing using the following:
```
@misc{bansal2022rethinking,
      title={Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale}, 
      author={Hritik Bansal and Karthik Gopalakrishnan and Saket Dingliwal and Sravan Bodapati and Katrin Kirchhoff and Dan Roth},
      year={2022},
      eprint={2212.09095},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

See [THIRD-PARTY](THIRD-PARTY.md) for a summary of changes made to third-party libraries,
described in the **Getting Started** section in detail, along with the associated licenses.

