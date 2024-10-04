import json
import logging
import os
import math
import random
import shutil
from pathlib import Path
from itertools import chain

# from dotenv import load_dotenv
import torch
import numpy as np
import accelerate
from torch.utils.data import DataLoader
import datasets
from datasets import Dataset, load_dataset, load_from_disk

from lm_experiments_tools import Trainer, TrainerArgs

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler

from peft import get_peft_model, LoraConfig, TaskType
# load_dotenv()
from babilong_utils import TaskDataset, SentenceSampler, NoiseInjectionDataset

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
# torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
# torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_dataset', type=str, help="Task name", default="qa1_single-supporting-fact")
parser.add_argument('--noise_dataset', type=str, help="Task name", default='wikitext')
parser.add_argument('--noise_dataset_split', type=str, help="Task name", default=None)
parser.add_argument('--babi_path', type=str, help="path to babi folder", default="data/tasks_1-20_v1-2/en-10k")


parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--block_size', type=int, default=128, help='max size of language modeling block')
parser.add_argument('--history_size', type=int, default=0, help='max number of past tokens for each block')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')

# Babilong parameters
parser.add_argument('--sample_size', type=int, default=None, help='max number of tokens in sample')
parser.add_argument('--max_n_facts', type=int, default=None, help='drop samples with higher number of facts')
parser.add_argument('--task_start_pct', type=float, default=None, help='left border of facts in sample, between 0 and 1')
parser.add_argument('--task_end_pct', type=float, default=None, help='right border of facts in sample, between task_start_pct and 1')


# RMT args 
parser.add_argument('--segment_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='randomly sample input size for each batch')
parser.add_argument('--mixed_length_ratio', type=float, default=0.0, help='used for mixed length curriculum. '
                    'r > 0.0 means that we will start to sample batches with lengths <= max_n_segments')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_alignment', type=str, help='way of aligning segments, one of right, left, center', default=None)
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')

# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')

# LoRA args
parser.add_argument('--use_lora', action='store_true', default=False, help='')
parser.add_argument('--lora_attn_dim', type=int, default=8, help='')
parser.add_argument('--lora_attn_alpha', type=int, default=32, help='')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='')
parser.add_argument('--layers_pattern', type=str, default=None, help='')

# Parallel Adapter args
parser.add_argument('--use_adapter', action='store_true', default=False, help='')
parser.add_argument('--adapter_bottleneck_dim', type=int, default=512, help='')
parser.add_argument('--adapter_dropout', type=float, default=0.1, help='')
parser.add_argument('--adapter_scale', type=float, default=4.0, help='')

# Dataset args
parser.add_argument('--pile_subset_names', type=str, default=None, help='use only these subsets of The PILE, separated by ;')
parser.add_argument('--min_tokens_in_document', type=int, default=None, help='do not use documents shorter than this value')
parser.add_argument('--max_tokens_in_document', type=int, default=None, help='do not use documents longer than this value')


if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    from accelerate.logging import get_logger
    logger = get_logger('')

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    # # create model path and save configuration
    # # todo: use prepare run
    # if accelerator.is_main_process and args.model_path is not None:
    #     model_path = Path(args.model_path)
    #     if not model_path.exists():
    #         Path(model_path).mkdir(parents=True)
    #     args_dict = collect_run_configuration(args)
    #     # todo: if model path exists and there is config file, write new config file aside
    #     json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)
    #     open(model_path / 'git.diff', 'w').write(get_git_diff())

    prepare_run(args, logger, logger_fmt)

    if not args.from_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    max_tokens = 1e6
    tokenizer.model_max_length = max_tokens

    # Prepare datasets
    logger.info(f'preparing dataset for {args.task_dataset}')
    try:
        # exit()
        # noise_dataset = datasets.load_dataset(args.noise_dataset, args.noise_dataset_split)
        noise_dataset = datasets.load_from_disk('/root/recurrent-memory-transformer-babilong/data/pg19')
        noise_dataset_train = noise_dataset['train']
        noise_dataset_test = noise_dataset['test']
    except ConnectionError:
        noise_dataset_train = datasets.Dataset.from_file('/root/.cache/huggingface/datasets/pg19/default/0.1.0/64837d6fce7251337df051ca74e9a5435d1c9cb7f3033ba257826e44d338f83c/pg19-train.arrow')
        noise_dataset_test = datasets.Dataset.from_file('/root/.cache/huggingface/datasets/pg19/default/0.1.0/64837d6fce7251337df051ca74e9a5435d1c9cb7f3033ba257826e44d338f83c/pg19-test.arrow')
    
    # task dataset 
    train_path = os.path.join(args.babi_path, f"{args.task_dataset}_train.txt")
    test_path = os.path.join(args.babi_path, f"{args.task_dataset}_test.txt")

    task_dataset_train = TaskDataset(train_path, max_n_facts=args.max_n_facts)
    task_dataset_test = TaskDataset(test_path, max_n_facts=args.max_n_facts)

    # background text
    # Option 1 ---> Not add Q at the begining 
    # qa_margin = 20          # leave space for questions and answers


    # Option 2 ---> Add Q at the first seg  |  Assume each question has 5 tokens
    # qa_margin = 20 + 5 

    # Option 3 ---> Add Q at the all seg
    qa_margin = 20 + (args.max_n_segments - 1) * 5

    if args.vary_n_segments:  # choose sample sizes according to each number of segments up to args.max_n_segments
        # train_sample_size = [int(args.sample_size / i) for i in range(1, args.max_n_segments + 1)]
        train_sample_size = [int(args.segment_size * i) for i in range(1, args.max_n_segments)] + [args.sample_size]
        train_sample_size = [s - qa_margin for s in train_sample_size]
        logger.info(f'Will be choosing sample size randomly from {train_sample_size} for training')
    else:
        sample_size = args.sample_size - qa_margin
        train_sample_size = args.sample_size - qa_margin
    test_sample_size = args.sample_size - qa_margin
    max_sentence_len = None
    if (args.task_start_pct is not None) and (args.task_end_pct is not None):
        # do not sample sentences longer than task position range * 0.5
        max_sentence_len = int((args.task_end_pct - args.task_start_pct) * 0.5 * args.sample_size)
        
    noise_sampler_train = SentenceSampler(noise_dataset_train, tokenizer=tokenizer, max_sentence_len=max_sentence_len, shuffle=True, random_seed=None)
    noise_sampler_test = SentenceSampler(noise_dataset_test, tokenizer=tokenizer, max_sentence_len=max_sentence_len, shuffle=True, random_seed=42)

    train_dataset = NoiseInjectionDataset(task_dataset=task_dataset_train,
                                            noise_sampler=noise_sampler_train,
                                            tokenizer=tokenizer,
                                            sample_size=train_sample_size,
                                            mixed_length_ratio=args.mixed_length_ratio,
                                            task_start_pct=args.task_start_pct,
                                            task_end_pct=args.task_end_pct
                                            )

    test_dataset = NoiseInjectionDataset(task_dataset=task_dataset_test,
                                            noise_sampler=noise_sampler_test,
                                            tokenizer=tokenizer,
                                            sample_size=test_sample_size,
                                            mixed_length_ratio=args.mixed_length_ratio,
                                            task_start_pct=args.task_start_pct,
                                            task_end_pct=args.task_end_pct
                                            )
    
    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_token = tokenizer.encode('GEN')[0]
    eos_token = tokenizer.eos_token_id

    # # Get question tokenizer
    # que_token = tokenizer.encode('QUEST')[0]

    def collate_fn(batch):
        targets = [torch.tensor(b['target_tokens']) for b in batch]
        # Option 1 --> Add question at the begining of sentences
        # input_ids = [torch.tensor(b['question_tokens'] + b['input_tokens'] + b['question_tokens'] + [gen_token] + b['target_tokens'] + [eos_token]) for b in batch]
        # gen_inputs = [torch.tensor(b['question_tokens'] + b['input_tokens'] + b['question_tokens'] + [gen_token]) for b in batch]

        # Option 2 --> Not add question at the begining of sentences  
        # input_ids = [torch.tensor(b['input_tokens'] + b['question_tokens'] + [gen_token] + b['target_tokens'] + [eos_token]) for b in batch]
        # gen_inputs = [torch.tensor(b['input_tokens'] + b['question_tokens'] + [gen_token]) for b in batch]

        # Option 3 --> Add question to each segment
        input_ids, gen_inputs = [], []
        for b in batch:
            question_len = len(b['question_tokens'])
            add_len = args.segment_size - question_len
            template_list = []

            # print(len(b['input_tokens']))
            for i in range(args.max_n_segments-1):
                # print(i*args.segment_size-question_len)
                # print(i)
                template_list += b['question_tokens']
                template_list += b['input_tokens'][i*add_len:(i+1)*add_len]
            
            template_list += b['input_tokens'][(args.max_n_segments-1)*add_len:]
            input_ids.append(torch.tensor(template_list + b['question_tokens'] + [gen_token] + b['target_tokens'] + [eos_token]))
            gen_inputs.append(torch.tensor(template_list + b['question_tokens'] + [gen_token]))

        
        # torch.set_printoptions(threshold=torch.inf)
        # print(len(input_ids[0]))
        # exit()

        attention_mask = [torch.ones_like(b, dtype=int) for b in input_ids]
        labels_mask = [torch.zeros_like(b, dtype=bool) for b in input_ids]
        for m, t in zip(labels_mask, targets):
            m[-len(t) - 2:] = True

        input_ids = pad_sequence(input_ids, padding_value=id_pad_value, batch_first=True)
        gen_inputs = pad_sequence(gen_inputs, padding_value=id_pad_value, batch_first=True)
        attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True)
        labels_mask = pad_sequence(labels_mask, padding_value=0, batch_first=True)

        collated = {}
        collated['input_ids'] = collated['labels'] = input_ids
        collated['input_ids_generate'] = gen_inputs
        collated['labels_mask'] = labels_mask
        collated['attention_mask'] = attention_mask.bool()
        collated['attention_mask_generate'] = (gen_inputs != id_pad_value).bool()
        collated['target_text'] = [b['answer'] for b in batch]
        return collated

    # train_dataset, valid_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers, 'collate_fn': collate_fn}

    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    train_sampler = DistributedSampler(train_dataset, rank=accelerator.process_index,
                                       num_replicas=accelerator.num_processes, shuffle=True, drop_last=True,
                                       seed=args.seed)
    test_sampler = DistributedSampler(test_dataset, rank=accelerator.process_index,
                                      num_replicas=accelerator.num_processes, drop_last=False, shuffle=False)
    train_dataloader = DataLoader(batch_size=per_worker_batch_size, dataset=train_dataset, sampler=train_sampler,
                                  **kwargs)
    test_dataloader = DataLoader(batch_size=per_worker_batch_size, dataset=test_dataset, sampler=test_sampler, **kwargs)

    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model
    model_cls = get_cls_by_name(args.model_cls)
    logger.info(f'Using model class: {model_cls}')

    if args.use_adapter:
        model_cfg = AutoConfig.from_pretrained(args.from_pretrained)

        model_cfg.use_parallel_adapter = args.use_adapter
        model_cfg.parallel_adapter_mode = 'ffn'
        model_cfg.adapter_bottleneck_dim = args.adapter_bottleneck_dim
        model_cfg.adapter_dropout = args.adapter_dropout
        model_cfg.adapter_scale = args.adapter_scale

        model = model_cls(config=model_cfg)

        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        base_model = model_cls.from_pretrained(args.from_pretrained, use_safetensors=False)

        model.load_state_dict(base_model.state_dict(), strict=False)
        del base_model
        logger.info(f'Added adapters')

    else:
        if not args.from_pretrained:
            model_cfg = AutoConfig.from_pretrained(args.model_cfg)
            model = model_cls(config=model_cfg)
        else:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
            model = model_cls.from_pretrained(args.from_pretrained, use_safetensors=False)

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_attn_dim, 
            lora_alpha=args.lora_attn_alpha, 
            lora_dropout=args.lora_dropout,
            layers_pattern=args.layers_pattern
            )
        model = get_peft_model(model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()
    

    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "pytorch_model.bin")
        # model = torch.load(backbone_cpt, map_location='cpu')
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt, strict=False)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')

        # -------------------------------------------------------------------------------------
        # Define compress module 
        from torch import nn
        from transformers import GPT2Config, AutoModel, GPT2Model
        from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
        
        # Option 1 ----> Use different model
        config = GPT2Config(
            n_layer=4,
        )

        compress_module = GPT2Model.from_pretrained('gpt2', config=config)  # From pretrained 
        # compress_module = GPT2Model(config)  # From scratch

        # # Option 2 ----> Use gpt2 first 4 layer as compress module
        # from transformers import GPT2PreTrainedModel
        # from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
        # class CompressModule(GPT2PreTrainedModel):
        #     def __init__(self, original_model):
        #         super().__init__(original_model.config)

        #         self.wte = original_model.transformer.wte
        #         self.wpe = original_model.transformer.wpe
        #         self.drop = original_model.transformer.drop
                
        #         # 共享前4个层
        #         self.h = nn.ModuleList(original_model.transformer.h[:4])  
        #         self.ln_f = original_model.transformer.ln_f
                

        #     def forward(self, input_ids=None, inputs_embeds=None, output_hidden_states=None, past_key_values=None, attention_mask=None, layer_past=None, position_ids=None, **kwargs):
        #         if input_ids is not None:
        #             input_shape = input_ids.size()
        #             input_ids = input_ids.view(-1, input_shape[-1])
        #             batch_size = input_ids.shape[0]
        #         elif inputs_embeds is not None:
        #             input_shape = inputs_embeds.size()[:-1]
        #             batch_size = inputs_embeds.shape[0]

        #         if past_key_values is None:
        #             past_length = 0
        #             past_key_values = tuple([None] * len(self.h))

        #         device = input_ids.device if input_ids is not None else inputs_embeds.device
                
        #         if position_ids is None:
        #             position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        #             position_ids = position_ids.unsqueeze(0)

        #         if inputs_embeds is None:
        #             inputs_embeds = self.wte(input_ids)

        #         position_embeds = self.wpe(position_ids)
        #         hidden_states = inputs_embeds + position_embeds
        #         attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None

        #         if attention_mask is not None:
        #             attention_mask = attention_mask[:, None, None, :]
        #             attention_mask = attention_mask.to(dtype=self.dtype)
        #             attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        #         hidden_states = self.drop(hidden_states)
                
        #         all_hidden_states = () if output_hidden_states else None
        #         for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        #             if output_hidden_states:
        #                 all_hidden_states = all_hidden_states + (hidden_states,)

        #             outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask)

        #         # print(len(outputs))
        #         hidden_states = outputs[0]
                
        #         hidden_states = self.ln_f(hidden_states)

        #         if output_hidden_states:
        #             all_hidden_states = all_hidden_states + (hidden_states,)

        #         return BaseModelOutputWithPastAndCrossAttentions(
        #             last_hidden_state=hidden_states,
        #             hidden_states=all_hidden_states,
        #         )

        # compress_module = CompressModule(model)

        # # Define hidden size and inner dim
        # hidden_size = config.hidden_size    
        # inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        # transit_module = GPT2MLP(inner_dim, config)
       # -------------------------------------------------------------------------

        # Frozen compress_module wte and GPT-2
        compress_module.wte.requires_grad_(False) # Let compress module use same wte with main model
        for name, param in model.named_parameters():
            param.requires_grad_(False)
        

        # for name, param in compress_module.named_parameters():
        #     if 'wte' not in name and 'wpe' not in name and 'ln_f.weight' not in name and 'ln_f.bias' not in name:
        #         param.requires_grad_(True)

        print('-------------------Base_Model-------------------------------------')
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")

        print('-------------------Compress_Module-------------------------------------')
        for name, param in compress_module.named_parameters():
            if param.requires_grad == True:
                print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")

        # -----------------------------------------------------------------------------------------------------------------------
        cell = memory_cell_cls(model, args.num_mem_tokens, compress_module)
        if args.segment_alignment not in {None, 'left'}:
            logger.info(f"Using custom segment alignment: {args.segment_alignment}")
        
        max_n_segments = args.max_n_segments
        if max_n_segments in {-1, None}:
            max_n_segments = np.ceil(args.sample_size / args.segment_size)
        model = recurrent_wrapper_cls(cell, 
                                      segment_size=args.segment_size,
                                      max_n_segments=max_n_segments, 
                                      segment_alignment=args.segment_alignment,
                                      k2=args.k2,
        )
                                    

        ## load cpt of rmt
        if args.model_cpt:
            model_cpt = os.path.join(args.model_cpt, "pytorch_model.bin")
            cpt = torch.load(model_cpt, map_location='cpu')
            model.load_state_dict(cpt, strict=False)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')
        
        # # Copy index
        # copy_idx = [0, 3, 7, 11]
        # for i in range(len(copy_idx)):
        #     model.memory_cell.compress_module.h[i].load_state_dict(model.memory_cell.model.transformer.h[copy_idx[i]].state_dict())

    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            if 'memory' not in n and 'lora' not in n and 'adapter' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info(f'Frozen moodel weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    # # fix the not-contiguous error
    # def make_contiguous(module):
    #     with torch.no_grad():
    #         for param in module.parameters():
    #             param.set_(param.contiguous())
    # make_contiguous(model)

    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
    # if args.model_cpt or args.backbone_cpt:
    #     optimizer.load_state_dict(cpt['optimizer_state_dict'])

    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        data['loss'] = output['loss']
        if 'generation_outputs' in output:
            generation_outputs = tokenizer.batch_decode(output['generation_outputs'][:, 1:], add_special_tokens=False)

            for i, o in enumerate(generation_outputs):
                if '<|endoftext|>' in o:
                    generation_outputs[i] = o.split('<|endoftext|>')[0].strip()

            num_correct = np.sum([text == pred for text, pred in zip (batch['target_text'], generation_outputs)])
            num_total = len(generation_outputs)
            data['num_correct'] = [num_correct]            
            data['num_total'] = [num_total]
        elif 'logits' in output:
            data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
            predicted_labels = [p[m[-len(p):]] for p, m in zip(data['predictions'], batch['labels_mask'])]
            predicted_labels = tokenizer.batch_decode(predicted_labels, add_special_tokens=False)
            for i, l in enumerate(predicted_labels):
                if '<|endoftext|>' in l:
                    eos_ind = predicted_labels[i].index('<|endoftext|>')
                    predicted_labels[i] = predicted_labels[i][:eos_ind]

            data['num_correct'] = [np.sum([text == pred for text, pred in zip (batch['target_text'], predicted_labels)])]
            data['num_total'] = [len(predicted_labels)]
            
        return data

    # HF datasets can compute metrics on each gpu process and then aggregate them on process with rank 0
    # synchronization is done by using temporay files on a shared filesystem
    # rank and number of workers is set by num_process and process_id params
    # BUT our Trainer aggregates all prediction from all gpus!
    #   this will lead to computing metrics for predictions repeated xN_GPUS times
    # need to try:
    # - keep_in_memory=True, may lead to OOM for large validation sets, after sync predictions and targets for the full
    #       validation set would be stored on each GPU -> xN_GPUs RAM
    #   - implemented currently
    # - compute metrics on batch lvl
    # - add support of HF metrics and turn off aggregation in case if metric has .add_batch method
    # scrolls_metric = datasets.load_metric(scrolls_metric_path, args.task_name, keep_in_memory=True)

    model, optimizer = accelerator.prepare(model, optimizer)
    # model, optimizer, _ = accelerator.prepare(model, optimizer, train_dataloader)

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        if 'num_correct' in data:
            metrics['exact_match'] = np.sum(data['num_correct']) / np.sum(data['num_total'])
        try:
            perplexity = math.exp(data["loss"].mean())
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        return metrics

    ### booydar
    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    trainer = Trainer(args, accelerator, model, optimizer, train_dataloader, test_dataloader,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      ###booydar
                      batch_metrics_fn=batch_metrics_fn,
                      generate_kwargs={"pad_token_id": id_pad_value, "max_new_tokens":10})

    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        accelerator.wait_for_everyone()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best')
            logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        # if valid_dataloader is not None:
        #     logger.info('Runnning validation on valid data:')
        #     trainer.validate(valid_dataloader, write_tb=False, split='valid')
        if test_dataloader is not None:
            logger.info('Runnning validation on test data:')
            trainer.validate(test_dataloader, write_tb=True, split='test')
        trainer.save_metrics(save_path=args.model_path)
    else:
        # if args.save_best:
        #     best_model_path = str(Path(args.model_path) / 'model_best')
        #     logger.info(f'Loading best saved model from {best_model_path}')
        #     trainer.load(best_model_path)
        # # run validation, do not write to tensorboard
        # logger.info('Running validation on train set:')
        # trainer.validate(train_dataloader, split='train', write_tb=False)
        # if valid_dataloader is not None:
        #     logger.info('Running validation on valid data:')
        #     trainer.validate(valid_dataloader, write_tb=True, split='valid')
        if test_dataloader is not None:
            logger.info('Runnning validation on test data:')
            trainer.validate(test_dataloader, write_tb=True, split='test')