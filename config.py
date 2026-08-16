from pathlib import Path

def get_config():
    return  {
        "batch_size": 16,
        "num_epochs": 8,
        "lr": 10**-4,
        "seq_len": 160,
        "d_model": 256,
        "N": 4,
        "h": 8,
        "d_ff": 1024,
        "max_examples": None,
        "lang_src": "en",
        "lang_tgt": "es",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_file_name = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_file_name)