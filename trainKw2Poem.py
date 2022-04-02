from ast import arg
from dataset_kw import PoemDataset
from transformers import DataCollatorForLanguageModeling
import torch
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, GPT2Config, GPT2LMHeadModel
import argparse
import os

parser = argparse.ArgumentParser(description="Train Keywords to Poem Model")
parser.add_argument('--train_dir', dest='train_dir', help='Train folder directory', type=str, required=True, default='./data/1ext_balanced_rkw_4sen_87609_test_kw2poem_dataset.csv')
parser.add_argument('--log', dest='log', help='Log directory', type=str, default='./logs')
parser.add_argument('--pretrained', dest='pretrained', help='Pretrained model path', type=str, default=None)
parser.add_argument('--saved', dest='saved', help='Saved model path', type=str, default='./models')
parser.add_argument('-e', '--epoch', dest='epoch', help='Epoch for training', type=int, default=1000)
parser.add_argument('-s', '--step', dest='step', help='Training step for training', type=int, default=5000)
parser.add_argument('-bs', dest='batch_size', help='Batch size for training', type=int, default=4)
args = parser.parse_args()

if __name__ == '__main__':
    pretrained_path = args.pretrained
    saved_path = args.saved
    train_path = args.train_dir

    test_path = './data/ext_balanced_rkw_4sen_87609_train_kw2poem_dataset.csv'

    log_dir= args.log

    EPOCH = args.epoch
    STEP = args.step
    BATCH_SIZE= args.batch_size

    if not os.path.isfile(train_path):
        raise Exception('Train file is not found')
    if not os.path.isdir(saved_path):
        os.mkdir(saved_path)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    tokenizer.add_tokens('\n')

    train_dataset = PoemDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size= 128)
    
    test_dataset = PoemDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    task = {"text-generation": {"do_sample": True, "max_length": 128}}
    configuration = GPT2Config(vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()), n_positions=130, n_ctx=130,
                            task_specific_params=task,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id, 
                            pad_token_id=tokenizer.pad_token_id,
                            sep_token_id=tokenizer.sep_token_id
                            )
    poem = GPT2LMHeadModel(configuration)
    
    if pretrained_path == None:
        model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        rand_weight = torch.rand(model_gpt2.lm_head.weight.shape)
        
        model_gpt2.lm_head.weight = torch.nn.parameter.Parameter(rand_weight)
        task_gpt2 = {"text-generation": {"do_sample": True, "max_length": 128}}
        config_gpt2 = configuration = GPT2Config(vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()), n_positions=130, n_ctx=130,
                                task_specific_params=task_gpt2,
                                eos_token_id=tokenizer.eos_token_id,
                                bos_token_id=tokenizer.bos_token_id, 
                                pad_token_id=tokenizer.pad_token_id,
                                sep_token_id=tokenizer.sep_token_id
                                )
        model_gpt2 = GPT2LMHeadModel(config_gpt2)
        model_gpt2.save_pretrained(saved_path)

    load_model_gpt2 = GPT2LMHeadModel.from_pretrained(saved_path)
    poem.load_state_dict(load_model_gpt2.state_dict())
    print('Loaded GPT-2 model')

    training_args = TrainingArguments(
        output_dir=saved_path,
        overwrite_output_dir=True,
        num_train_epochs=EPOCH,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_steps=STEP,
        save_total_limit = 2,
        warmup_steps=STEP,
        logging_dir=log_dir,
        logging_steps=STEP,
        evaluation_strategy="steps"
        )

    device = torch.device('cuda')
    trainer = Trainer(
        model=poem,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    trainer.train()
    trainer.save_model()
