import os
import torch
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from scripts.plotting.style import *
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--prefix_matching", action = 'store_true')
parser.add_argument("--copying_score", action = 'store_true')
parser.add_argument("--random_model", action = 'store_true')
parser.add_argument("--frequent_exclude_ratio", type=float, default = 0.04)
parser.add_argument("--pretrained", type = str, default = 'facebook/opt-66b')
parser.add_argument("--model_cache_dir", type = str, default = None)
parser.add_argument("--tokenizer_cache_dir", type = str, default = None)
parser.add_argument("--num_seeds", type = int, default = 100)
parser.add_argument("--save_plot_path_mean", type=str, default=None)
parser.add_argument("--save_plot_path_var", type=str, default=None)
parser.add_argument("--save_outputs", type=str, default=None)
parser.add_argument("--use_save_outputs", action = 'store_true')

args = parser.parse_args()

if not args.use_save_outputs:
    device_map = 'auto'
    if not args.random_model:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained, cache_dir = args.model_cache_dir, device_map = device_map)
    else:
        config = AutoConfig.from_pretrained(args.pretrained)
        model = AutoModelForCausalLM.from_config(config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, use_fast = False, cache_dir = args.tokenizer_cache_dir)
    ## create a ranking of bpe tokens using tokenizer.bpe_ranks that stores bpe token merges based on frequency in pretraining text
    ## BPE tokens are saved as per merging order in the dict bpe_ranks
    ## more details about merging at https://huggingface.co/docs/transformers/tokenizer_summary
    ranked_dict = dict()
    ranked_vocab_size = len(list(tokenizer.bpe_ranks.keys()))
    check_all_ranks = [0]*ranked_vocab_size
    for merge_tuple,rank in tokenizer.bpe_ranks.items():
        bpe_token = ''.join(merge_tuple)
        ranked_dict[rank] = tokenizer.encoder[bpe_token]
        check_all_ranks[rank] = 1
    assert sum(check_all_ranks) == ranked_vocab_size
    ## exclude fraction of frequent bpe tokens from random sequences
    frequent_excluded_ranks = int(args.frequent_exclude_ratio * ranked_vocab_size)
    ## exclude both most and least frequent tokens
    rank_start, rank_end = frequent_excluded_ranks, ranked_vocab_size - frequent_excluded_ranks
    assert rank_start < rank_end and rank_end > 0
    rank_choice_list = np.arange(rank_start, rank_end)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    final = []

    with torch.no_grad():
        for seed in tqdm(range(args.num_seeds)):
            torch.manual_seed(seed)
            ## ensures final length of the generated sequence is in the range (25,~900)
            length = seed * 2 + 25
            ## sequence is not repeated for copying score
            if args.copying_score:
                length = 4 * length
            ## choose a random sequence excluding most frequent and least frequent bpe tokens
            ## generate tokens without replacement to ensure all chosen tokens are unique
            ## uniqueness ensures prefix matching score to only capture explicit repeats ie repeat of the whole sequence
            generate_ranks = np.random.choice(rank_choice_list, size=length, replace=False)
            ## append a bos_token in the beginning to ensure normal model behaviour
            generate_ids = torch.tensor([tokenizer.bos_token_id] + [ranked_dict[rank] for rank in generate_ranks])
            generate_ids = torch.unsqueeze(generate_ids, 0)
            if not args.random_model:
                generate_ids = generate_ids.to(0)
            if args.prefix_matching:
                ## repeat the sequence excluding the bos token
                new_generated = torch.cat([generate_ids, generate_ids[:,1:].repeat(3, 1).view(-1).unsqueeze(0)], dim = -1)
                if not args.random_model:
                    new_generated = new_generated.to(0)
                assert new_generated.shape[1] == 4*length + 1
                out = model(input_ids = new_generated)
                decoder = model.get_decoder()
                attn_matrix = torch.zeros((num_layers, num_heads))
                for layer in range(num_layers):
                    attn_probs = decoder.layers[layer].self_attn.attn_probs
                    for head in range(num_heads):
                        attn_prob = attn_probs[head]
                        c = 0
                        for j in range(length+1, 4*length+1):
                            for num in range(j//length):
                                attn_matrix[layer][head] += attn_prob[j][(num*length)+(j%length)+1].item()
                            c += 1
                        attn_matrix[layer][head] = attn_matrix[layer][head] / c
                final.append(attn_matrix.unsqueeze(0))
            
            elif args.copying_score:
                new_generated  = generate_ids
                decoder = model.get_decoder()
                input_shape = new_generated.size()
                input_ids = new_generated.view(-1, input_shape[-1])
                past_key_values_length = 0
                
                inputs_embeds = decoder.embed_tokens(input_ids)
                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
                pos_embeds = decoder.embed_positions(attention_mask, past_key_values_length)
                attention_mask = decoder._prepare_decoder_attention_mask(
                    attention_mask, input_shape, inputs_embeds, past_key_values_length
                )
                hidden_states = inputs_embeds + pos_embeds

                copying_matrix = torch.zeros((num_layers, num_heads))
                for layer in tqdm(range(num_layers)):
                    layer_ = decoder.layers[layer]
                    hs = layer_.self_attn_layer_norm(hidden_states)
                    layer_self_attn = layer_.self_attn
                    attn_probs = layer_self_attn(hidden_states = hs, attention_mask = attention_mask, output_attentions = True)[1].squeeze(0)
                    value_states = layer_self_attn._shape(layer_self_attn.v_proj(hs), -1, 1).squeeze(0) #n_heads, length, dim_head
                    h, l, d = value_states.shape
                    # convert h, l, d_h -> h, l, d_e so that it can be fed to out_proj directly
                    value_states = [torch.cat([torch.zeros((1,l,i*d), dtype = value_states.dtype, device = value_states.device), value_states[i,:,:].unsqueeze(0), \
                            torch.zeros((1,l,(h-i-1)*d), dtype = value_states.dtype, device = value_states.device)], dim = -1) for i in range(len(value_states))]
                    value_states = torch.cat(value_states, dim = 0) # h, l, d_e
                    output = layer_self_attn.out_proj(value_states)

                    logits = model.lm_head(output).contiguous() # h, l, vocab_size
                    logits = F.softmax(logits, dim = -1)

                    for head in range(num_heads):
                        attn_prob = attn_probs[head]
                        _, ind = torch.sort(attn_prob, dim = 1)
                        max_ind = ind[:, -1]
                        c = 0
                        ## iterate the complete random sequence
                        for j in range(1, length + 1):
                            c += 1
                            assert (max_ind[j] <= j)
                            ## tokens that can be attended to in the current time step ie 0 to j
                            attendable_input = input_ids[0][:(j+1)]
                            ## logits of attendable tokens
                            attendable_logits = logits[head][j][attendable_input]
                            ## mean of the logits
                            mean_of_logits = attendable_logits.mean()
                            ## raise logits
                            raised_logits = attendable_logits - mean_of_logits
                            ## relu over raised logits
                            relu_raised_logits = torch.nn.functional.relu(raised_logits)
                            relu_raised_logit_max_ind = relu_raised_logits[max_ind[j]].item()
                            relu_raised_logit_all = relu_raised_logits.sum().item()
                            ## ratio of raised logit
                            copying_score = 0
                            ## edgecase: if all logits are of equal value then relu_raised_logit_all can be 0
                            if relu_raised_logit_all != 0:
                                copying_score = relu_raised_logit_max_ind / relu_raised_logit_all
                            copying_matrix[layer][head] += copying_score
                        copying_matrix[layer][head] = copying_matrix[layer][head] / c
                final.append(copying_matrix.unsqueeze(0))
            else:
                raise RuntimeError("Neither prefix matching nor copying score selected")
    final = torch.cat(final, dim = 0)
    mean = final.mean(dim = 0)
    variance = final.var(dim = 0)
    os.makedirs(os.path.dirname(args.save_outputs), exist_ok = True)

    with open(args.save_outputs, 'wb') as f:
        pickle.dump({'mean': mean, 'variance': variance}, f)

if args.use_save_outputs:
    with open(args.save_outputs, 'rb') as f:
        res = pickle.load(f)
        mean, variance = res['mean'], res['variance']
        num_layers, num_heads = mean.shape


max_, min_ = mean.max(), mean.min()
print(max_, min_)
print(mean.shape)
## changed the range for best visualization of copying score
ax = sns.heatmap(mean.numpy(), xticklabels = [(i+1) if i%2==0 else None for i in range(num_heads)], yticklabels = [(i+1) if i%2==0 else None for i in range(num_layers)], vmin = min_, vmax = max_)
plt.ylabel('Layers')
plt.xlabel('Heads')
plt.title('Prefix Matching Score' if args.prefix_matching else 'Copying Score')
ax.invert_yaxis()

os.makedirs(os.path.dirname(args.save_plot_path_mean), exist_ok = True)
plt.savefig(args.save_plot_path_mean)
plt.savefig(args.save_plot_path_mean[:-4]+'.pdf')

plt.close()

max_, min_ = variance.max(), variance.min()
print(max_, min_)
ax = sns.heatmap(variance.numpy(), xticklabels = [(i+1) if i%2==0 else None for i in range(num_heads)], yticklabels = [(i+1) if i%2==0 else None for i in range(num_layers)], vmin = min_, vmax = max_)
plt.ylabel('Layers')
plt.xlabel('Heads')
plt.title('Prefix Matching Score' if args.prefix_matching else 'Copying Score')
ax.invert_yaxis()

os.makedirs(os.path.dirname(args.save_plot_path_var), exist_ok = True)
plt.savefig(args.save_plot_path_var)
plt.savefig(args.save_plot_path_var[:-4]+'.pdf')
