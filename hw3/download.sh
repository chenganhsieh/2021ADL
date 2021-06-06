# pretrain model
wget https://huggingface.co/google/mt5-small/resolve/main/config.json -O ./code/cache/mt5_small/config.json
wget https://huggingface.co/google/mt5-small/resolve/main/spiece.model -O ./code/cache/mt5_small/spiece.model
wget https://huggingface.co/google/mt5-small/resolve/main/tokenizer_config.json -O ./code/cache/mt5_small/tokenizer_config.json

# my checkpoint
wget https://www.dropbox.com/s/4cip6b667wz7wsl/bestmodel.ckpt?dl=1 -O ./code/bestmodel/bestmodel.ckpt
wget https://www.dropbox.com/s/50nr2ekf0cni3dw/events.out.tfevents.1622885144.cuda8.23202.0?dl=1 -O ./code/bestmodel/events.out.tfevents.1622885144.cuda8.23202.0
wget https://www.dropbox.com/s/y0b5nn4a38y9q0v/public.jsonl?dl=1 -O ./public.jsonl