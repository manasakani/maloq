import argparse
import numpy as np
import re


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute statistics on epoch runtimes.')
    parser.add_argument('--file', type=str, required=True, help='Path to the training report.')
    args = parser.parse_args()

    def parse(line):
        match = re.match(r"Epoch \d+ - Time: (\d+)\.(\d+) seconds", line)
        if match:
            # print(match)
            # print(match.group(0))
            # print(match.group(0, 1))
            # print(match.group(0, 2))
            int_seconds = match.group(0, 1)[1]
            float_seconds = match.group(0, 2)[1]
            return float(f"{int_seconds}.{float_seconds}")

    with open(args.file, 'r') as f:
        lines = f.readlines()
        lines = [parse(l) for l in lines]
        lines = [l for l in lines if l is not None]
    
    median = np.median(lines)
    std = np.std(lines)
    print(f"Median Epoch runtime: {median:.3f} seconds (std: {std:.3f})")
