# Dual-Transformer

Dual-Transformer is the framework with the input is an scenery image and the output is Vietnamese six-eight poem. The generated poem is related to the input image by containing the objects from the input image.

The code has been put on [HuggingFace's Space](https://huggingface.co/spaces/crylake/img2poem)

## Installation

1. Clone the repository:

```sh
    git clone https://github.com/chauminhnguyen/Dual-Transformer.git
```

2. Install the requirements

 - Install cuda.

 - Install the requirements.

```sh
    pip install -r requirements.txt
```

3. Modify the path

- Modify the config.json for Query2labels and GPT-2 models path.

- Modify the Query2labels's config.json *(default: models/Query2labels/config.json)* for the pretrained's path.

4. Start the model

```sh
    streamlit run app.py
```

![The general of the Img2Poem website](/Images/general.png "The Img2Poem website")

![Infer an image](/Images/infer.png "The Img2Poem website") -->

## Train Model

### Train Image-to-Keywords Model

I used [Query2Label](https://github.com/SlongLiu/query2labels) for Image-to-Keywords Model. The command below is used to train on my Image-to-Keywords dataset.

```sh
    python main_mlc.py 
                --dataset_dir './data' --backbone resnet101 --dataname coco14 
                --batch-size 1 --print-freq 100 --output "./output" --world-size 1 --rank 0 
                --dist-url tcp://127.0.0.1:3717 --gamma_pos 0 --gamma_neg 2 --dtgfl --epochs 40 
                --lr 1e-4 --optim AdamW --pretrained --num_class 76 --img_size 448 
                --weight-decay 1e-2 --cutout --n_holes 1 --cut_fact 0.5 --hidden_dim 2048 
                --dim_feedforward 4096 --enc_layers 1 --dec_layers 2 --nheads 4 --early-stop 
                --amp --workers 2
```

### Tran Keywords-to-Poem Model

I used [GPT-2](https://huggingface.co/gpt2) for Keywords-to-Poem model.

```sh
    python trainKw2Poem.py
                --train_dir './data/1ext_balanced_rkw_4sen_87609_test_kw2poem_dataset.csv'
                --epoch 100 --step 10000 --batch_size 8
```

## Pretrained Models

|Model name|Link|
| ------ | ------ |
|GPT-2|[link](https://drive.google.com/drive/folders/1F0I2XDJcMhKqRVsgmuzZCKTF_so2NyPg?usp=sharing)|
|Query2label|[link](https://drive.google.com/drive/folders/1GIdrUCoZ_xcONq23UYM15BN2l6zjucSn?usp=sharing)|

## Acknowledgement

We thank the authors of [Query2Label](https://github.com/SlongLiu/query2labels), [GPT-2](https://huggingface.co/gpt2) for facilitating such an opportunity for us to create this framework.