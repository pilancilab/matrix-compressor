from dataclasses import dataclass
from datetime import datetime as dt
import argparse
import pathlib
from loguru import logger
import pandas as pd
import re

@dataclass
class ScriptArgs:
    """Class for keeping track of input arguments of LPLR scripts"""

    program: pathlib.Path
    model_directory: pathlib.Path
    output_directory: pathlib.Path
    b1: int
    b2: int
    b_nq: int
    cr: float
    map_location: str
    sketch: str = "Gaussian"


def parse_input_line(input_line):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory")
    parser.add_argument("--output-directory")
    parser.add_argument("--b1", type=int)
    parser.add_argument("--b2", type=int)
    parser.add_argument("--b_nq", type=int)
    parser.add_argument("--cr", type=float)
    parser.add_argument("--map-location")
    parser.add_argument("--sketch", default="Gaussian")

    known_args = vars(parser.parse_known_args(input_line)[0])
    logger.debug(known_args)
    return ScriptArgs(**known_args, program=pathlib.Path(input_line[0]))


def parse_log_file(filename: str):
    with open(filename) as f:
        lines = f.read().splitlines()
    pattern = re.compile(
        r".*Name: (?P<name>.*) Shape: (?P<shape>.*) Row Sketch Error: (?P<rs_err>.*) Col Sketch Error: (?P<cs_err>.*) Dtype: (?P<dtype>.*) Naive Quant Err: (?P<nq_err>.*) Compression Ratio: \d"
    )
    parsed_logs = [pattern.match(l) for l in lines]
    df = pd.DataFrame.from_records(
        (p.groupdict() for p in parsed_logs if p is not None)
    )
    df = df.astype({"rs_err": "float64", "cs_err": "float64", "nq_err": "float64"})
    return df


def summarize_log_dataframe(df: pd.DataFrame):
    series = df.aggregate({"rs_err": "mean", "cs_err": "mean", "nq_err": "mean"})
    series["count"] = len(df)
    return series


def format_log_summary_as_str(ser: pd.Series):
    return f"RS Error: {ser['rs_err']:.3f} CS Error: {ser['cs_err']:.3f} NQ Errpr: {ser['nq_err']:.3f} Count: {ser['count']}"

def merge_log_summary_and_args(
    script_args: ScriptArgs, summary: pd.Series, start_time: dt, logfile=""
):
    base_name = script_args.program.stem
    logfile = pathlib.Path(logfile)
    if base_name == "layer_wise_lplr_svd_quantization":
        technique = "LPLR SVD"
    elif base_name == "layer_wise_lplr_quantization":
        technique = "LPLR"
    elif base_name == "layer_wise_error_quantization":
        technique = "LPLR Residual"
    else:
        technique = "Unknown"

    change_time = dt.fromisoformat("2023-05-06T19:32:45.146178")
    compression_ratio = f"{script_args.cr}"
    if start_time <= change_time:
        compression_ratio += "'"
    return {
        "Method": technique,
        "B1": script_args.b1,
        "B2": script_args.b2,
        "B0": script_args.b_nq,
        "CR": compression_ratio,
        "Sketch": script_args.sketch,
        "Row Sketch Error": summary["rs_err"],
        "Col Sketch Error": summary["cs_err"],
        "Naive Quantization Error": summary["nq_err"],
        "Row Count": summary["count"],
        "logfile": logfile.stem if logfile.exists() else "Unknown Path",
        "start_time": start_time
    }