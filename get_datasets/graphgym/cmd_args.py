import argparse
import sys


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file path',
        #required=True,
        type=str,
        default='./run/reddit-body.tsv_gru_0.5.yaml'
        #default='./run/AS-733_gru_0.5.yaml'
    )
    parser.add_argument(
        '--mark_done',
        dest='mark_done',
        action='store_true',
        help='mark yaml as yaml_done after a job has finished',
        default=None
    )

    parser.add_argument(
        '--override_remark',
        dest='override_remark',
        type=str,
        required=False,
        default=None,
        help='easily override the remark in the yaml file'
    )

    parser.add_argument(
        '--override_data_dir',
        dest='override_data_dir',
        type=str,
        required=False,
        default='../public_data',
        help='easily override the dataset.dir in the yaml file'
    )

    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER
    )

    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    return parser.parse_args()
