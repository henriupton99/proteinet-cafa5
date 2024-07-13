import yaml
import argparse

from .preproc.targets import generate_targets

def main(config: str, mode: str):
    
    with open(config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    paths = config['paths']
    vars_mode = config[mode]
    
    if mode == 'preproc':
        generate_targets(
            ids_path = paths['train_ids'],
            labels_path = paths['train_labels'],
            weights_path = paths['ia_weights'],
            go_obo_path = paths['go_obo_files'],
            evidence_codes_path = paths['evidence_codes'],
            targets_path = paths['targets'],
            aspects_list = vars_mode['aspects'],
            go_terms_per_aspects = vars_mode['aspects_labels']
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--mode', type=str, required=True, help='operation mode (preproc/train/infer)')
    args = parser.parse_args()
    main(**vars(args))