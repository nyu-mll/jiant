"""Translate raw prediction files for benchmark tasks into format expected by
benchmark leaderboards.
"""
import os
import argparse

from jiant.scripts.benchmarks.benchmarks import GlueBenchmark, SuperglueBenchmark


SUPPORTED_BENCHMARKS = {"GLUE": GlueBenchmark, "SUPERGLUE": SuperglueBenchmark}


def main():
    parser = argparse.ArgumentParser(
        description="Generate formatted test prediction files for benchmark submission"
    )
    parser.add_argument(
        "--input_base_path",
        required=True,
        help="base path where per-task folders contain raw prediction files",
    )
    parser.add_argument("--output_path", required=True, help="output path for formatted files")
    parser.add_argument(
        "--benchmark", required=True, choices=SUPPORTED_BENCHMARKS, help="name of benchmark"
    )
    parser.add_argument(
        "--tasks", required=False, nargs="+", help="subset of benchmark tasks to format"
    )
    args = parser.parse_args()

    benchmark = SUPPORTED_BENCHMARKS[args.benchmark]

    if args.tasks:
        assert set(args.tasks) <= benchmark.TASKS
        task_names = args.tasks
    else:
        task_names = benchmark.TASKS

    for task_name in task_names:
        input_filepath = os.path.join(args.input_base_path, task_name, "test_preds.p")
        output_filepath = os.path.join(
            os.path.abspath(args.output_path), benchmark.BENCHMARK_SUBMISSION_FILENAMES[task_name]
        )
        benchmark.write_predictions(task_name, input_filepath, output_filepath)


if __name__ == "__main__":
    main()
