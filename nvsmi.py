#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
A (user-)friendly wrapper to nvidia-smi

Author: Panagiotis Mavrogiorgos
Adapted from: https://github.com/anderskm/gputil

"""

import argparse
import itertools as it
import json
import logging
import operator
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Iterable, List, Optional

__version__ = "0.6.0"


NVIDIA_SMI_GET_GPUS = "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu,timestamp --format=csv,noheader,nounits"
NVIDIA_SMI_GET_PROCS = "nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,gpu_name,used_memory,timestamp --format=csv,noheader,nounits"
NVIDIA_TIME_FMT = "%Y/%m/%d %H:%M:%S.%f"


class GPUState(object):
    def __init__(
        self,
        uuid: str,
        gpu_util: float,
        mem_used: float,
        mem_free: float,
        temperature: float,
        timestamp: datetime,
    ):
        self.uuid = uuid
        self.gpu_util = gpu_util
        self.mem_used = mem_used
        self.mem_free = mem_free
        self.temperature = temperature
        self.timestamp = timestamp

    def __repr__(self) -> str:
        msg = "{timestamp} | UUID: {uuid} | gpu_util: {gpu_util:5.1f}% | mem_util: {mem_util:5.1f}% | mem_free: {mem_free:7.1f}MB"
        msg = msg.format(**self.__dict__)
        return msg

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


class GPU(object):
    def __init__(
        self,
        id: int,
        uuid: str,
        mem_total: float,
        driver: str,
        gpu_name: str,
        serial: str,
        display_mode: str,
        display_active: str,
        timestamp: datetime,
    ):
        self.id = id
        self.uuid = uuid
        self.mem_total = mem_total
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.timestamp = timestamp
        self.states: List[GPUState] = []

    def _append_state(self, gpu_state: GPUState) -> None:
        if gpu_state.uuid == self.uuid:
            self.states.append(gpu_state)
        else:
            logging.warning(
                "GPU UUID and GPUState UUID do not match! State not appended."
            )

    def update_states(self) -> None:
        output = subprocess.check_output(
            shlex.split(f"{NVIDIA_SMI_GET_GPUS} -i {self.id}")
        )
        line = output.decode("utf-8")
        gpu_state = _get_gpu_state(line)
        self.states.append(gpu_state)

    def clear_states(self) -> None:
        self.states = []

    def get_latest_state(self) -> GPUState:
        return self.states[-1]

    def get_states(self) -> List[GPUState]:
        return self.states

    def __repr__(self) -> str:
        msg = "{timestamp} | UUID: {uuid} | id: {id} | mem_total: {mem_total:7.1f}MB"
        msg = msg.format(**self.__dict__)
        return msg

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


class GPUProcess(object):
    def __init__(
        self,
        pid: int,
        process_name: str,
        gpu_uuid: str,
        used_memory: float,
        timestamp: datetime,
    ):
        self.pid = pid
        self.process_name = process_name
        self.gpu_uuid = gpu_uuid
        self.used_memory = used_memory
        self.timestamp = timestamp

    def __repr__(self) -> str:
        msg = "{timestamp} | pid: {pid} | gpu_uuid: {gpu_uuid} | used_memory: {used_memory:7.1f}MB"
        msg = msg.format(**self.__dict__)
        return msg

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


def to_float_or_inf(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        number = float("nan")
    return number


def _get_gpu(line: str) -> GPU:
    values = line.split(", ")
    id = int(values[0])
    uuid = values[1]
    gpu_util = to_float_or_inf(values[2])
    mem_total = to_float_or_inf(values[3])
    mem_used = to_float_or_inf(values[4])
    mem_free = to_float_or_inf(values[5])
    driver = values[6]
    gpu_name = values[7]
    serial = values[8]
    display_active = values[9]
    display_mode = values[10]
    temperature = to_float_or_inf(values[11])
    timestamp = datetime.strptime(values[12], NVIDIA_TIME_FMT)
    gpu = GPU(
        id=id,
        uuid=uuid,
        mem_total=mem_total,
        driver=driver,
        gpu_name=gpu_name,
        serial=serial,
        display_mode=display_mode,
        display_active=display_active,
        timestamp=timestamp,
    )
    gpu._append_state(
        GPUState(
            uuid=uuid,
            gpu_util=gpu_util,
            mem_used=mem_used,
            mem_free=mem_free,
            temperature=temperature,
            timestamp=timestamp,
        )
    )
    return gpu


def _get_gpu_state(line: str) -> GPUState:
    values = line.split(", ")
    uuid = values[1]
    gpu_util = to_float_or_inf(values[2])
    mem_used = to_float_or_inf(values[4])
    mem_free = to_float_or_inf(values[5])
    temp_gpu = to_float_or_inf(values[11])
    timestamp = datetime.strptime(values[12], NVIDIA_TIME_FMT)
    gpu_state = GPUState(
        uuid=uuid,
        gpu_util=gpu_util,
        mem_used=mem_used,
        mem_free=mem_free,
        temperature=temp_gpu,
        timestamp=timestamp,
    )
    return gpu_state


def get_gpus() -> List[GPU]:
    output = subprocess.check_output(shlex.split(NVIDIA_SMI_GET_GPUS))
    lines = output.decode("utf-8").split(os.linesep)
    gpus = [_get_gpu(line) for line in lines if line.strip()]
    return gpus


def _get_gpu_proc(line: str) -> GPUProcess:
    values = line.split(", ")
    pid = int(values[0])
    process_name = values[1]
    gpu_uuid = values[2]
    used_memory = to_float_or_inf(values[4])
    timestamp = datetime.strptime(values[5], NVIDIA_TIME_FMT)
    proc = GPUProcess(
        pid=pid,
        process_name=process_name,
        gpu_uuid=gpu_uuid,
        used_memory=used_memory,
        timestamp=timestamp,
    )
    return proc


def get_gpu_processes() -> List[GPUProcess]:
    output = subprocess.check_output(shlex.split(NVIDIA_SMI_GET_PROCS))
    lines = output.decode("utf-8").split(os.linesep)
    processes = [_get_gpu_proc(line) for line in lines if line.strip()]
    return processes


################################################################ Pause
def is_gpu_available(
    gpu: GPU,
    gpu_util_max: float,
    mem_util_max: float,
    mem_free_min: float,
    include_ids: List[int],
    include_uuids: List[str],
) -> bool:
    gpu.update_states()
    latest_state = gpu.get_latest_state()
    mem_util = (latest_state.mem_used / latest_state.mem_free) * 100
    return (
        (latest_state.gpu_util <= gpu_util_max)
        and (mem_util <= mem_util_max)
        and (latest_state.mem_free >= mem_free_min)
        and (gpu.id in include_ids)
        and (gpu.uuid in include_uuids)
    )


def get_available_gpus(
    gpu_util_max: float = 1.0,
    mem_util_max: float = 1.0,
    mem_free_min: float = 0,
    include_ids: List[int] = [],
    include_uuids: List[str] = [],
):
    """Return up to `limit` available cpus"""
    # Normalize inputs (include_ids and include_uuis need to be iterables)
    gpus = get_gpus()
    include_ids = include_ids or [gpu.id for gpu in gpus]
    include_uuids = include_uuids or [gpu.uuid for gpu in gpus]
    # filter available gpus
    selectors = (
        is_gpu_available(
            gpu, gpu_util_max, mem_util_max, mem_free_min, include_ids, include_uuids
        )
        for gpu in gpus
    )
    available_gpus = it.compress(gpus, selectors)
    return available_gpus


def get_parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(
        prog="nvsmi", description="A (user-)friendy interface for nvidia-smi"
    )
    main_parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )

    subparsers = main_parser.add_subparsers(
        help="Choose mode of operation", dest="mode", title="subcommands"
    )  # noqa
    ls_parser = subparsers.add_parser("ls", help="List available GPUs")
    ps_parser = subparsers.add_parser("ps", help="Examine the process of a gpu.")

    # list
    ls_parser.add_argument(
        "--ids", type=int, nargs="+", metavar="", help="List of GPU IDs to include"
    )

    ls_parser.add_argument(
        "--uuids", type=str, nargs="+", metavar="", help="List of GPU uuids to include"
    )

    ls_parser.add_argument(
        "--limit",
        type=int,
        default=999,
        metavar="",
        help="Limit the number of the GPUs",
    )
    ls_parser.add_argument(
        "--mem-free-min",
        type=float,
        default=0,
        metavar="",
        help="The minimum amount of free memory (in MB)",
    )
    ls_parser.add_argument(
        "--mem-util-max",
        type=int,
        default=100,
        metavar="",
        help="The maximum amount of memory [0, 100]",
    )
    ls_parser.add_argument(
        "--gpu-util-max",
        type=int,
        default=100,
        metavar="",
        help="The maximum amount of load [0, 100]",
    )
    ls_parser.add_argument(
        "--sort",
        default="id",
        choices=["id", "gpu_util", "mem_util"],
        metavar="",
        help="Sort the GPUs using the specified attribute",
    )
    ls_parser.add_argument(
        "--json", action="store_true", help="Show results in JSON format"
    )

    # processes
    ps_parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        metavar="",
        help="Show only the processes of the GPU matching the provided ids",
    )
    ps_parser.add_argument(
        "--uuids",
        nargs="+",
        type=str,
        metavar="",
        help="Show only the processes of the GPU matching the provided UUIDs",
    )
    ps_parser.add_argument(
        "--json", action="store_true", help="Show results in JSON format"
    )

    return main_parser


def _take(n: int, iterable: Iterable):
    "Return first n items of the iterable as a list"
    return it.islice(iterable, n)


def is_nvidia_smi_on_path() -> Optional[str]:
    return shutil.which("nvidia-smi")


def _nvsmi_ls(args):
    gpus = list(
        get_available_gpus(
            gpu_util_max=args.gpu_util_max,
            mem_util_max=args.mem_util_max,
            mem_free_min=args.mem_free_min,
            include_ids=args.ids,
            include_uuids=args.uuids,
        )
    )
    gpus.sort(key=operator.attrgetter(args.sort))
    for gpu in _take(args.limit, gpus):
        output = gpu.to_json() if args.json else gpu
        print(output)


def _nvsmi_ps(args):
    gpus = get_gpus()
    uuid_to_id = {gpu.uuid: gpu.id for gpu in gpus}
    processes = get_gpu_processes()
    if args.ids or args.uuids:
        for proc in processes:

            if (
                uuid_to_id.get(proc.gpu_uuid, None) in args.ids
                or proc.gpu_uuid in args.uuids
            ):
                output = proc.to_json() if args.json else proc
                print(output)
    else:
        for proc in processes:
            output = proc.to_json() if args.json else proc
            print(output)


def validate_ids_and_uuids(args):
    gpus = get_gpus()
    gpu_ids = {gpu.id for gpu in gpus}
    gpu_uuids = {gpu.uuid for gpu in gpus}
    invalid_ids = args.ids.difference(gpu_ids)
    invalid_uuids = args.uuids.difference(gpu_uuids)
    if invalid_ids:
        sys.exit(f"The following GPU ids are not available: {invalid_ids}")
    if invalid_uuids:
        sys.exit(f"The following GPU uuids are not available: {invalid_uuids}")


def _main():
    parser = get_parser()
    args = parser.parse_args()
    # Cast ids and uuids to sets
    if args.mode is None:
        parser.print_help()
        sys.exit()
    else:
        args.ids = set(args.ids) if args.ids else set()
        args.uuids = set(args.uuids) if args.uuids else set()
        validate_ids_and_uuids(args)
        if args.mode == "ls":
            _nvsmi_ls(args)
        elif args.mode == "ps":
            _nvsmi_ps(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    # cli mode
    if not is_nvidia_smi_on_path():
        sys.exit("Error: Couldn't find 'nvidia-smi' in $PATH: %s" % os.environ["PATH"])
    _main()
