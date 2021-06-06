import os
import argparse
import torch
import pdb
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import (
    MT5Config,
    MT5Tokenizer,
    MT5Model,
    MT5ForConditionalGeneration
)
from pltrainer import PLTrainer
from dataset import TextGenerationDataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
seed_everything(42)

def main(args):
    # model
    config = MT5Config.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    tokenizer = MT5Tokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    if not args.predict:
        model = MT5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
        )
    else:
        model = MT5ForConditionalGeneration(config=config)


    # dataset
    if not args.predict:
        train_dataset = TextGenerationDataset(args,tokenizer)
        train_dataset= train_dataset.get_dataset()["train"]
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'decoder_input_ids','decoder_attention_mask','labels'])
       
        val_dataset = TextGenerationDataset(args,tokenizer,mode="validation")
        val_dataset,sample_dataset = val_dataset.get_dataset()
        val_dataset = val_dataset["validation"]
        sample_dataset = sample_dataset["sample"]
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'decoder_input_ids','decoder_attention_mask','labels'])
        sample_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'decoder_input_ids','decoder_attention_mask','labels'])
    else:
        eval_dataset = TextGenerationDataset(args,tokenizer,mode="eval")
        eval_dataset = eval_dataset.get_dataset()["eval"]
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'id'])

    datasets = {
        'train': train_dataset if not args.predict else None,
        'val': val_dataset  if not args.predict else None,
        'sample': sample_dataset if not args.predict else None,
        'eval': eval_dataset  if args.predict else None,
    }

    # log tensor blog !!!!!  可以印loss 整個分析圖
    tb_logger = pl_loggers.TensorBoardLogger(args.log_dir)

    checkpoint_callback = ModelCheckpoint(
        filename= '{step:05d}-{v_loss:.2f}-{rouge_1:.2f}-{rouge_2:.2f}-{rouge_l:.2f}', # step: 5個整數 
        save_top_k=3, # 前三名
        verbose=False,
        monitor='v_loss', # 看自己的loss dict name
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')


    if not args.predict:
        pltrainer = PLTrainer(args, model,datasets,tokenizer)
    else:
        pltrainer = PLTrainer.load_from_checkpoint(args.predict_model,args=args, model=model,datasets=datasets,tokenizer = tokenizer) 
    
    trainer = pl.Trainer(
        fast_dev_run=False,
        logger=tb_logger, # 自己的logger
        gpus=1, 
        max_epochs=args.max_epochs, 
        auto_scale_batch_size='binsearch', # 自己設定batch size
        progress_bar_refresh_rate=1, # 進度條
        # accelerator='ddp',
        accumulate_grad_batches=12, # distrubuted training 
        # amp_backend='apex',
        # amp_level='O3',
        precision=16, # fp_16 or fp_32
        # gradient_clip_val=0,
        val_check_interval=0.25,
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
        # plugins=DDPPlugin(find_unused_parameters=True),
        callbacks=[checkpoint_callback, lr_monitor],
    )
    if args.predict:
        trainer.test(pltrainer)
    else:
        trainer.fit(pltrainer)
        # trainer.tune(pltrainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/train.jsonl", type=str)
    parser.add_argument("--val_data_dir", default="./data/public.jsonl", type=str)
    parser.add_argument("--eval_data_dir", default="./data/public.jsonl", type=str)
    parser.add_argument("--cache_dir", default="./code/cache", type=str)
    parser.add_argument('--log_dir', default='./code/logs/',type=str,required=False)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--max_summar_length", default=64, type=int)
    parser.add_argument("--num_layers", default=6, type=int)
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--head", default=4, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    # parser.add_argument("--stop_threshold", default=0.5, type=float)
    parser.add_argument("--lr", default=2e-4, type=float) # 3e-6
    # parser.add_argument("--b1", default=0.9, type=float)
    # parser.add_argument("--b2", default=0.999, type=float) # bert-base-chinese
    parser.add_argument('--model_name_or_path', type=str, default = "./code/cache/mt5_small/", help='specific model name of the given model type')
    parser.add_argument('--predict_model', type=str, default = "./code/bestmodel/bestmodel.ckpt", help='specific model name of the given model type')
    parser.add_argument('--doc_stride', type=int, default = 32, help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument('--pad_to_max_length', 
        type=bool, 
        default = True, 
        help="Whether to pad all samples to `max_seq_length`. "
        "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
        "be faster on GPU but will be slower on TPU)."
    )
    parser.add_argument("--max_steps", default=100000, type=int)
    parser.add_argument("--max_epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--output_file", default="./output.jsonl",type=str)
    parser.add_argument("--rl_ratio", default=0,type=float)
    parser.add_argument("--rl_ratio_power", default=1,type=float)
    parser.add_argument("--rl_start_epoch", default=20,type=int)
    parser.add_argument("--output_mode", default="greedy",type=str,choices=["greedy","beam","topk","topp","synthesis"])
    parser.add_argument("--temperature", default=0.9,type=int)
    parser.add_argument("--nums_beam", default=8,type=int)
    parser.add_argument("--topk", default=20,type=int)
    parser.add_argument("--topp", default=0.6,type=float)


    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    main(args)