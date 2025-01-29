import argparse
import yaml

def train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="checkpoint/train", help="project/name")
    parser.add_argument("--exp_name", default="GcDance", help="save to project/name")
    parser.add_argument("--feature_type", type=str, default="baseline")
    parser.add_argument("--datasplit", type=str, default="cross_genre", choices=["cross_genre", "cross_dancer"])
    parser.add_argument("--data_path", type=str, default="dataset/", help="raw data path")
    parser.add_argument(
        "--render_dir", type=str, default="renders", help="Sample render path"
    )
    
    parser.add_argument(
        "--motion_dir", type=str, default="dataset/train/motion_fea319", help="dataset motion"
    )
    parser.add_argument(
        "--music_fm_dir", type=str, default="dataset/train/wav2clip_fea", help="dataset motion"
    )
    parser.add_argument(
        "--music_basic_dir", type=str, default="dataset/train/music_npy", help="dataset motion"
    )
    parser.add_argument(
        "--wav_dir", type=str, default="dataset/finedance/music_wav", help="dataset motion"
    )
    parser.add_argument(
        "--full_seq_len", type=int, default=120, help="full_seq_len"
    ) 
    parser.add_argument(
        "--windows", type=int, default=10, help="windows"
    ) 
    parser.add_argument(
        "--mix", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--wandb_pj_name", type=str, default="dance_gen", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")        # default=64
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,            # default=100,  
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument(
        "--do_normalize",
        action="store_true",
        help="normalize",
    )
    parser.add_argument(
        "--nfeats", type=int, default=319, help="nfeats"
    ) 
    parser.add_argument(
        "--test_metric", action="store_true", help="test the metric while training"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use wandby while training"
    )
    parser.add_argument(
        "--genre_json", type=str, default='dataset/finedance/dance_lable_data.json', help="json file"
    ) 
    opt = parser.parse_args()
    return opt

def test_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/", help="raw data path")
    
    parser.add_argument("--feature_type", type=str, default="baseline")
    parser.add_argument(
        "--full_seq_len", type=int, default=120, help="full_seq_len"
    ) 
    parser.add_argument("--datasplit", type=str, default="cross_genre", choices=["cross_genre", "cross_dancer"])
    parser.add_argument(
        "--windows", type=int, default=10, help="windows"
    ) 
    parser.add_argument("--out_length", type=float, default=30, help="max. length of output, in seconds")
    parser.add_argument(
        "--render_dir", type=str, default="render", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="checkpoint"
    )
    
    parser.add_argument(
        "--nfeats", type=int, default=319, help="nfeats(319/139)"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="test/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--do_normalize",
        action="store_true",
        help="normalize",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/vol/research/CMVCG/xl/dataset/test_10",
        help="Where to save/load the features",
    )
    parser.add_argument(
        "--genre_json", type=str, default='dataset/finedance/dance_lable_data.json', help="json file"
    ) 
    parser.add_argument(
        "--type", type=int, default=1, 
        help="1 -> hand, 2 -> body, 3 -> plc & align")
    parser.add_argument(
        "--test_10", 
        action="store_true", 
        help="Random test 10 times")
    parser.add_argument(
        "--test_gen", 
        action="store_true", 
        help="Random test 10 times")
    opt = parser.parse_args()
    return opt


def save_arguments_to_yaml(args, file_path):
    arg_dict = vars(args)  
    yaml_str = yaml.dump(arg_dict, default_flow_style=False)

    with open(file_path, 'w') as file:
        file.write(yaml_str)