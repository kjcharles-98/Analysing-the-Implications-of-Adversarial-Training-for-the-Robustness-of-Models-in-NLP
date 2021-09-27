import torch
import textattack
import transformers
from transformers import (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)

dirmodel = '/data/cheng/checkpoints/FreeLB-albert-xxlarge-v2-RTE-alr1-amag6e-1-anm0.2-as3-sl512-lr1e-5-bs4-gas1-hdp0.1-adp0-ts20935-ws1256-wd1e-2-seed42/checkpoint-best'


config_class, model_class, tokenizer_class = AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer

config = config_class.from_pretrained(
        dirmodel,
        num_labels=2,
        finetuning_task='rte',
        cache_dir=None,
        attention_probs_dropout_prob=0,
        hidden_dropout_prob=0.1
    )
    
tokenizer = tokenizer_class.from_pretrained(
        dirmodel,
        do_lower_case=True,
        cache_dir=None,
    )

model = model_class.from_pretrained(
        dirmodel,
        from_tf=False,
        config=config,
        cache_dir=None,
    )

model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)