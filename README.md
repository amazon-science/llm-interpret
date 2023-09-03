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
script dumps the evaluation results into JSON files, which are necessary to create some plots in
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
results are dumped in JSON files using the code described above.

Note that you may have to edit these scripts a bit according to the naming convention
you adopt for the importance score pickle and evaluation result JSON files you create.


## Sample Commands



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

