from transformers import AutoTokenizer, pipeline, PretrainedConfig
from query2labels.infer import parser_args, Query2Label
import json
import os

with open('./config.json') as f:
    args = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
# tokenizer = AutoTokenizer.from_pretrained("./GPT2/phoBert", use_fast=False)

# -------Load model saved-----------------#

img2kw_conf_path = os.path.join(args.Img2Kw_model, 'config.json')

parser = parser_args()
parser.add_argument('--config', help='config file', default= img2kw_conf_path)
parser.add_argument('-f')
args = parser.parse_args()

vis_extractor = Query2Label(args)
# infer.main(args)


configuration = {'num_beams': 5, 'max_length': 256, "architectures": ["GPT2LMHeadModel"]}
config = PretrainedConfig()
config.from_dict(configuration)
poem = pipeline('text-generation', model=args.Kw2Poem_model,
                tokenizer=tokenizer,
                config=config)


def main(img):
    clses = vis_extractor.predict(img)
    
    keywords = clses
    print(keywords)
    keywords = ' '.join(keywords)
    poem = generate_poem(keywords)
    return poem


def generate_poem(keywords):
    # Test
    input = '<s>' + keywords + ' [SEP]'
    a = poem(input)
    out = a[0]['generated_text']
    out = out.replace('<s>', '')
    out = out.replace('</s>', '')
    out = out.split('<unk>')
    return '\n'.join(out)