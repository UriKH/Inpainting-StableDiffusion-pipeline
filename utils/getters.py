import argparse


def input_output_paths_args(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to the input")
    parser.add_argument("output_path", help="Path to the output")

    args = parser.parse_args()
    return args.input_path, args.output_path
