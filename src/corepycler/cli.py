import os
import re
import time
import click
import logger
import logging
import math
import random
import psutil
import cpuinfo
import subprocess
import textwrap
import statistics
import platform
import importlib.metadata
from typing import Optional
from enum import Enum, StrEnum, auto
from collections import defaultdict
from pydantic import RootModel, Field
from pydantic.dataclasses import dataclass
from ruamel.yaml import YAML, CommentedMap
from pathlib import Path
from pyfiglet import Figlet
from datetime import datetime, timedelta
from click_option_group import optgroup, MutuallyExclusiveOptionGroup
from tabulate import tabulate
from yaspin import yaspin

from .__about__ import __version__
from . import APP_NAME

log = logger.get_logger(__name__)
_stress_proc: Optional[subprocess.Popen] = None

ROOT_PATH = Path(__file__).parent.parent.parent
TOOLS_PATH = ROOT_PATH / "tools"
PRIME95_DIR = "prime95"
LOGS_PATH = ROOT_PATH / "logs"


class UpperStrEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.upper()


def render_help_header() -> None:
    author = importlib.metadata.metadata(__package__)["Author-email"].split()[0]
    click.echo(Figlet(font='slant', width=90).renderText(APP_NAME))
    click.echo(f"By {author}, version: {__version__}\n")


class HelpGroup(click.Group):
    def format_help(self, ctx, formatter) -> None:
        render_help_header()
        super().format_help(ctx, formatter)


class HelpCommand(click.Command):
    def format_help(self, ctx, formatter) -> None:
        render_help_header()
        super().format_help(ctx, formatter)


class CoreTestOrder(UpperStrEnum):
    SEQUENTIAL = auto()
    RANDOM = auto()


class StressTestProgram(UpperStrEnum):
    PRIME95 = auto()


class Prime95Mode(UpperStrEnum):
    SSE = auto()
    # AVX = auto()
    # AVX2 = auto()
    # AVX512 = auto()


class Prime95FFTSize(Enum):
    SMALLEST = (4096, 20480)
    SMALL = (40960, 245760)
    LARGE = (458752, 8388608)
    HUGE = (9175040, 33554432)
    ALL = (4096, 33554432)
    MODERATE = (1376256, 4194304)
    HEAVY = (4096, 1376256)
    HEAVYSHORT = (4096, 163840)

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_value(cls, value: tuple[int, int]) -> 'Prime95FFTSize':
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No matching Prime95FFTSize for value: {value}")

    @staticmethod
    def convert_to_k(size: tuple) -> tuple[int, int]:
        return math.floor(size[0] / 1024), math.ceil(size[1] / 1024)


@dataclass
class Prime95Config:
    mode: Prime95Mode
    fft_size: str | tuple[int, int]


@dataclass
class Config:
    version: int = 1
    runtime_per_core_m: int = 6
    suspend_periodically: bool = True
    core_test_order: CoreTestOrder = CoreTestOrder.SEQUENTIAL
    skip_core_on_error: bool = True
    stop_on_error: bool = False
    delay_between_cores: int = 15
    max_iterations: int = 1000
    cores_to_ignore: list[int] = Field(default_factory=lambda: [])  # noqa
    # look_for_whea_errors: bool = True # TODO: how on linux?
    stress_test_program: StressTestProgram = StressTestProgram.PRIME95
    prime95: Prime95Config = Field(default_factory=lambda: Prime95Config(mode=Prime95Mode.SSE,
                                                                         fft_size=Prime95FFTSize.SMALL.name))

    def __post_init__(self) -> None:
        if isinstance(self.prime95.fft_size, str):
            self.prime95.fft_size = Prime95FFTSize[self.prime95.fft_size].value

    @classmethod
    def from_file(cls, path: str | Path) -> 'Config':
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Config file {path.as_posix()} not exists.')

        config = RootModel[cls].model_validate(YAML().load(path.read_text())).root
        try:
            if config.version != cls.version:
                raise ValueError(f'Invalid config file version. Expected: {cls.version}, actual: {config.version}.')
        except KeyError:
            raise ValueError(f'Missing version field in config.')
        return config

    def to_file(self, path: str | Path) -> None:
        try:
            self.prime95.fft_size = Prime95FFTSize.from_value(self.prime95.fft_size).name
        except ValueError:
            pass

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = CommentedMap(RootModel[self.__class__](self).model_dump(mode='json'))
        d.yaml_set_comment_before_after_key('runtime_per_core_m', before="Runtime per core in minutes.")
        d.yaml_set_comment_before_after_key('suspend_periodically',
                                            before="Periodically suspend the stress test program "
                                                   "to simulate load changes to idle and back.")
        d.yaml_set_comment_before_after_key('skip_core_on_error', before="Skip a core that has thrown an error "
                                                                         "in the following iterations.")
        d.yaml_set_comment_before_after_key('cores_to_ignore', before="Logical core ids of physical cores to ignore.")
        with path.open('w+') as file:
            yaml = YAML()
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.dump(d, file)

    @staticmethod
    def get_default_filename() -> str:
        return 'config.yaml'

    def print(self) -> None:
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        log.info("Configuration:")
        table_data = [["Stress test program", self.stress_test_program]]
        if self.stress_test_program == StressTestProgram.PRIME95:
            table_data.append(["Selected test mode", f"{self.prime95.mode}"])
            fft_size = Prime95FFTSize.convert_to_k(self.prime95.fft_size)
            table_data.append(["Selected FFT size", f"{fft_size[0]}K - {fft_size[1]}K"])
        table_data.append(["Detected processor", cpuinfo.get_cpu_info()['brand_raw']])
        table_data.append(["Logical/Physical cores", f"{logical_cores}/{physical_cores} cores"])
        table_data.append(["Runtime per core", f"{self.runtime_per_core_m} minutes"])
        table_data.append(["Suspend periodically", f"{self.suspend_periodically}"])
        table_data.append(["Skip core on error", f"{self.skip_core_on_error}"])
        table_data.append(["Stop on error", f"{self.stop_on_error}"])
        table_data.append(["Test order of cores", f"{self.core_test_order}"])
        table_data.append(["Number of iterations", f"{self.max_iterations}"])
        table_data.append(["Estimated testing time",
                           f"{timedelta(seconds=int((self.runtime_per_core_m * 60
                                                     + self.delay_between_cores) * physical_cores * self.max_iterations)
                                                - self.delay_between_cores)}"])
        print(tabulate(table_data, tablefmt="simple"))


class Profile(Enum):
    CO_STABILITY = Config(runtime_per_core_m=30,
                          core_test_order=CoreTestOrder.SEQUENTIAL,
                          max_iterations=2,
                          skip_core_on_error=True,
                          stop_on_error=False,
                          delay_between_cores=15,
                          stress_test_program=StressTestProgram.PRIME95,
                          suspend_periodically=True,
                          prime95=Prime95Config(Prime95Mode.SSE, (720 * 1024, 1344 * 1024)))
    CO_STABILITY_FAST = Config(runtime_per_core_m=3,
                               core_test_order=CoreTestOrder.SEQUENTIAL,
                               max_iterations=2,
                               skip_core_on_error=True,
                               stop_on_error=False,
                               delay_between_cores=15,
                               stress_test_program=StressTestProgram.PRIME95,
                               suspend_periodically=True,
                               prime95=Prime95Config(Prime95Mode.SSE, (720 * 1024, 1344 * 1024)))

def core_test_loop(config: Config) -> None:
    cores = get_physical_core_ids()
    if config.core_test_order == CoreTestOrder.RANDOM:
        cores = dict(random.sample(list(cores.items()), len(cores)))

    cores = {k: v for k, v in cores.items() if k not in config.cores_to_ignore}
    bad_cores = set()
    core_stats = defaultdict(lambda: [0, 0])

    iteration = 0
    error_cnt = 0
    tested_cores = set()

    start_time = time.time()
    try:
        while iteration < config.max_iterations:
            tested_cores.clear()
            for i, (core_id, logical_id) in enumerate(cores.items()):
                if core_id in bad_cores:
                    log.info(f"Skipping core {core_id} due to previous error.")
                    continue

                time_left = (config.runtime_per_core_m * 60 + config.delay_between_cores) * len(cores) * (
                            config.max_iterations - iteration) - config.delay_between_cores
                time_left -= (config.runtime_per_core_m * 60 + config.delay_between_cores) * i
                success = run_prime95(core_id,
                                      logical_id,
                                      config.runtime_per_core_m,
                                      config.suspend_periodically,
                                      config.prime95.fft_size,
                                      time_left)
                tested_cores.add(core_id)

                log.info(f"Progress {len(tested_cores)}/{len(cores)} "
                         f"| Iteration {iteration + 1}/{config.max_iterations} "
                         f"| Run time {timedelta(seconds=int(time.time() - start_time))}")

                if success:
                    core_stats[core_id][0] += 1
                else:
                    core_stats[core_id][1] += 1
                    error_cnt += 1
                    if config.stop_on_error:
                        log.error(f"Stopping test due to an error on core {core_id}.")
                        raise RuntimeError
                    else:
                        if config.skip_core_on_error:
                            log.warning(f"Skip core {core_id} due to an error in next iterations.")
                            bad_cores.add(core_id)
                        else:
                            log.error(f"Error on core {core_id}.")

                is_last_core = core_id == list(cores.keys())[-1]
                is_last_iteration = iteration == config.max_iterations - 1
                if config.delay_between_cores and not (is_last_core and is_last_iteration):
                    log.info(f"Waiting {config.delay_between_cores} s before next core.")
                    time.sleep(config.delay_between_cores)

            iteration += 1
    except (KeyboardInterrupt, RuntimeError):
        if _stress_proc and _stress_proc.poll() is None:
            _stress_proc.terminate()
            _stress_proc.wait()

        log.info(f"Progress {len(tested_cores)}/{len(cores)} | Iteration {iteration}/{config.max_iterations} "
                 f"| Run time {timedelta(seconds=int(time.time() - start_time))}")

    if error_cnt == 0:
        log.info("No core has thrown an error.")
    else:
        log.info(f"Errors detected: {error_cnt}")
    if iteration == config.max_iterations:
        log.info(f"All {iteration}/{config.max_iterations} test iterations completed successfully!")
        log.info(f"Run time: {timedelta(seconds=int(time.time() - start_time))}")
        log.info(f"Tested cores: {len(tested_cores)}/{len(cores)}")
        print_core_stats(core_stats)


def print_core_stats(core_stats: dict[int, list[int]]) -> None:
    table_data = [[f"Core {core_id}", stats[0], stats[1]]for core_id, stats in core_stats.items()]
    log.info("Core statistics:")
    print(tabulate(table_data, headers=["", "Success", "Error"], tablefmt="simple"))


def burn_test(duration_m: int) -> None:
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    log.info("Configuration:")
    table_data = [["Stress test program", "PRIME95"],
                  ["Detected processor", cpuinfo.get_cpu_info()['brand_raw']],
                  ["Logical/Physical cores", f"{logical_cores}/{physical_cores} cores"],
                  ["Test duration", f"{duration_m} minutes"]
    ]
    print(tabulate(table_data, tablefmt="simple"))

    start_time = time.time()
    try:
        success = run_prime95_burn(duration_m)
        if not success:
            log.error(f"Stopping test due to an error.")
            raise RuntimeError
    except (KeyboardInterrupt, RuntimeError):
        if _stress_proc and _stress_proc.poll() is None:
            _stress_proc.terminate()
            _stress_proc.wait()

    log.info(f"All completed successfully!")
    log.info(f"Run time: {timedelta(seconds=int(time.time() - start_time))}")
    log.info(f"Tested cores: {physical_cores}/{physical_cores}")


def run_prime95(core_id: int,
                logical_id: int,
                duration_m: int,
                suspend_periodically: bool,
                fft_size: tuple,
                time_left: int) -> bool:
    fft_size = Prime95FFTSize.convert_to_k(fft_size)
    prepare_prime95_config(fft_size[0], fft_size[1], 1)

    log.info(f"Running Prime95 on core {core_id} with FFT size {fft_size[0]}K - {fft_size[1]}K "
             f"for {duration_m} minutes.")

    global _stress_proc
    _stress_proc = subprocess.Popen([get_prime95_path(), '-t'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True)
    os.set_blocking(_stress_proc.stdout.fileno(), False)

    ps_proc = psutil.Process(_stress_proc.pid)
    ps_proc.cpu_affinity([logical_id])

    start_time = time.time()
    tick_interval = 10
    suspend_duration = 1
    next_tick = start_time + tick_interval
    log_temps_next_tick = next_tick - tick_interval / 2
    test_pass_cnt = 0

    with yaspin().dots12 as sp:
        while time.time() - start_time < duration_m * 60:
            try:
                line = _stress_proc.stdout and _stress_proc.stdout.readline()
                if line:
                    log_msg = re.sub(r'^.*?] ', '', line).strip()
                    if log_msg:
                        with sp.hidden():
                            log.info(f"[Prime95]: {log_msg}")
                        test_pass = detect_prime95_test_pass(log_msg)
                        if test_pass is not None:
                            if test_pass:
                                test_pass_cnt += 1
                            else:
                                with sp.hidden():
                                    log.error(f"Prime95 test error on core {core_id}")
                                return False
            except BlockingIOError:
                pass

            if _stress_proc.poll() is not None:
                with sp.hidden():
                    log.error(f"Prime95 terminated unexpectedly on core {core_id}")
                return False

            if time.time() >= next_tick:
                if suspend_periodically:
                    with sp.hidden():
                        log.debug(f"Suspending the stress test process for {suspend_duration} seconds.")
                    ps_proc.suspend()
                    time.sleep(suspend_duration)
                    ps_proc.resume()
                    with sp.hidden():
                        log.debug("Stress test process resumed.")
                next_tick += tick_interval
                log_temps_next_tick = next_tick - tick_interval / 2

            if time.time() >= log_temps_next_tick:
                log_temps_next_tick *= 2  # Do not print every iteration after tick till next tick.
                with sp.hidden():
                    print_cpu_temperature()

            sp.text = f"Time left: {timedelta(seconds=int(time_left - (time.time() - start_time)))}"
            time.sleep(0.1)

    _stress_proc.terminate()
    _stress_proc.wait()

    if test_pass_cnt == 0:
        log.error(f"No test pass detected on core {core_id}")
        return False

    log.info(f"Test completed in {timedelta(seconds=int(time.time() - start_time))} on core {core_id}")
    return True


def run_prime95_burn(duration_m: int) -> bool:
    prepare_prime95_config(720, 720, 1, burning=True)

    log.info(f"Running Prime95 on all cores for {duration_m} minutes.")

    global _stress_proc
    _stress_proc = subprocess.Popen([get_prime95_path(), '-t'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True)
    os.set_blocking(_stress_proc.stdout.fileno(), False)

    start_time = time.time()
    test_pass_cnt = 0
    log_temps_interval = 30
    log_temps_next_tick = start_time + log_temps_interval

    with yaspin().dots12 as sp:
        while time.time() - start_time < duration_m * 60:
            try:
                line = _stress_proc.stdout and _stress_proc.stdout.readline()
                if line:
                    log_msg = filter_prime95_logs(re.sub(r'^.*?] ', '', line).strip())
                    if log_msg:
                        with sp.hidden():
                            log.info(f"[Prime95]: {log_msg}")
                        test_pass = detect_prime95_test_pass(log_msg)
                        if test_pass is not None:
                            if test_pass:
                                test_pass_cnt += 1
                            else:
                                with sp.hidden():
                                    log.error(f"Prime95 test error.")
                                return False
            except BlockingIOError:
                pass

            if _stress_proc.poll() is not None:
                with sp.hidden():
                    log.error(f"Prime95 terminated unexpectedly.")
                return False

            if time.time() >= log_temps_next_tick:
                log_temps_next_tick += log_temps_interval
                with sp.hidden():
                    print_cpu_temperature()

            sp.text = f"Time left: {timedelta(seconds=int(duration_m * 60 - (time.time() - start_time)))}"
            time.sleep(0.1)

    _stress_proc.terminate()
    _stress_proc.wait()

    if test_pass_cnt == 0:
        log.error(f"No test pass detected.")
        return False

    log.info(f"Test completed in {timedelta(seconds=int(time.time() - start_time))}")
    return True


def print_cpu_temperature() -> None:
    sensors_temperatures = psutil.sensors_temperatures()
    core_temps = get_core_temps(sensors_temperatures) or get_cpu_thermal(sensors_temperatures)
    if core_temps:
        table_data = []
        row_cnt = -1
        for i, t in enumerate(core_temps):
            if i % 8 == 0:
                table_data.append([])
                row_cnt += 1
            table_data[row_cnt].append(f"C{i}: {t:.1f}")
        log.info("Core temperatures [*C]:")
        print(tabulate(table_data, tablefmt="simple_grid"))
        log.info(f"Average CPU temperature: {statistics.mean(core_temps):.1f} *C")
        return

    k10temp = get_k10temp(sensors_temperatures)
    if k10temp:
        label, temp = next(iter(k10temp.items()))
        log.info(f"CPU temperature {label}: {temp:.1f} *C")
        return


def get_core_temps(sensors_temperatures: dict) -> Optional[list[float]]:
    if 'coretemp' in sensors_temperatures:
        temps = [t for t in sensors_temperatures['coretemp'] if 'core' in t.label.lower()]
        if temps:
            def core_index(entry):
                match = re.search(r'core\s*(\d+)', entry.label.lower())
                return int(match.group(1)) if match else float('inf')
            temps = sorted(temps, key=core_index)
            return [t.current for t in temps]
    return None


def get_k10temp(sensors_temperatures: dict) -> Optional[dict[str, float]]:
    if 'k10temp' in sensors_temperatures:
        for t in sensors_temperatures['k10temp']:
            if t.label.lower() in ('tctl', 'tdie'):
                return {t.label: t.current}
    return None


def get_cpu_thermal(sensors_temperatures: dict) -> Optional[list[float]]:
    if 'cpu_thermal' in sensors_temperatures:
        return [t.current for t in sensors_temperatures['cpu_thermal']]
    return None


def filter_prime95_logs(log_msg: str) -> Optional[str]:
    ignore_keywords = (
        "worker starting",
        "worker stopped",
        "please read stress.txt",
    )
    if any(kw in log_msg.lower() for kw in ignore_keywords):
        return None
    return log_msg


def detect_prime95_test_pass(log_msg: str) -> Optional[bool]:
    log_msg = log_msg.lower()
    if 'self-test' in log_msg:
        return bool(re.search(r'self[- ]test\s+\d+k?\s+passed!?', log_msg))
    elif 'torture test failed' in log_msg or 'fatal error' in log_msg:
        return False
    else:
        return None


def prepare_prime95_config(min_fft_k: int, max_fft_k: int, single_size_duration_m: int, burning: bool = False) -> None:
    log_path = LOGS_PATH / f"Prime95_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SSE_{min_fft_k}K-{max_fft_k}K.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    config_path = get_prime95_path().parent / "prime.txt"
    if burning:
        config_path.write_text(textwrap.dedent(f"""\
            RollingAverageIsFromV27=1
            CpuSupportsSSE=1
            CpuSupportsSSE2=1
            CpuSupportsAVX=0
            CpuSupportsAVX2=0
            CpuSupportsFMA3=0
            CpuSupportsAVX512F=0
            results.txt={log_path.as_posix()}
            TortureMem=0
            TortureTime={single_size_duration_m}
            MinTortureFFT={min_fft_k}
            MaxTortureFFT={max_fft_k}
            TortureWeak={get_torture_weak_value(False, False, False)}
            V2UOptionsConverted=1
            V300ptionsConverted=1
            ExitOnX=1
            ResultsFileTimestampInterval=60
            StressTester=1
            UsePrimenet=0
        """))
    else:
        config_path.write_text(textwrap.dedent(f"""\
            RollingAverageIsFromV27=1
            CpuSupportsSSE=1
            CpuSupportsSSE2=1
            CpuSupportsAVX=0
            CpuSupportsAVX2=0
            CpuSupportsFMA3=0
            CpuSupportsAVX512F=0
            NumWorkers=1
            NumCores=1
            CoresPerTest=1
            results.txt={log_path.as_posix()}
            TortureHyperthreading=0
            TortureMem=0
            TortureTime={single_size_duration_m}
            MinTortureFFT={min_fft_k}
            MaxTortureFFT={max_fft_k}
            TortureWeak={get_torture_weak_value(False, False, False)}
            V2UOptionsConverted=1
            V300ptionsConverted=1
            ExitOnX=1
            ResultsFileTimestampInterval=60
            EnableSetAffinity=0
            EnableSetPriority=0
            StressTester=1
            UsePrimenet=0
        """))


def get_torture_weak_value(cpu_supports_fma3: bool, cpu_supports_avx: bool, cpu_supports_avx512: bool) -> int:
    """
    Calculate the TortureWeak value based on CPU feature support.

    If a feature is *disabled*, its corresponding constant is added.
    If enabled, it's ignored.

    Constants from Prime95's cpuid.h:
    - CPU_AVX512F = 1048576
    - CPU_FMA3    = 32768
    - CPU_AVX     = 16384
    """

    # Convert True/False (support) to 0/1 (disabled flag)
    fma3_flag   = int(not cpu_supports_fma3)
    avx_flag    = int(not cpu_supports_avx)
    avx512_flag = int(not cpu_supports_avx512)

    torture_weak_value = (
        avx512_flag * 1048576 +
        fma3_flag   * 32768 +
        avx_flag    * 16384
    )

    return torture_weak_value


def get_physical_core_ids() -> dict[int, int]:
    # TODO: Add Windows implementation
    physical_cores = set()

    with open('/proc/cpuinfo', 'r') as f:
        physical_id = logical_id = None

        for line in f:
            if line.startswith('processor'):
                logical_id = int(line.strip().split(':')[1])
            elif line.startswith('physical id'):
                physical_id = int(line.strip().split(':')[1])
            elif line.startswith('core id'):
                core_id = int(line.strip().split(':')[1])
                if None not in (physical_id, core_id, logical_id):
                    physical_cores.add((physical_id, core_id, logical_id))

    # Deduplicate by (physical_id, core_id) and return one logical ID per physical core
    seen = set()
    unique_logical_ids = []

    for physical_id, core_id, logical_id in sorted(physical_cores):
        key = (physical_id, core_id)
        if key not in seen:
            unique_logical_ids.append(logical_id)
            seen.add(key)

    return {i: core_id for i, core_id in enumerate(sorted(unique_logical_ids))}


def get_prime95_path() -> Path:
    if platform.system() == 'Linux':
        prime_executable = f'mprime'
    elif platform.system() == 'Windows':
        prime_executable = f'prime95'
    else:
        raise NotImplementedError(f"Unsupported platform: {platform.system()}")
    return TOOLS_PATH / _get_platform_dir() / f'{PRIME95_DIR}/{prime_executable}{_get_executable_extension()}'


def _get_platform_dir() -> str:
    arch = 'x86_64' if platform.architecture()[0].rstrip('bit') == '64' else 'x86_32'
    if platform.system() == 'Linux':
        return f'linux-{arch}'
    elif platform.system() == 'Windows':
        return f'win32-{arch}'
    elif platform.system() == 'Darwin':
        return f'macos-{arch}'
    else:
        raise NotImplementedError(f"Unsupported platform: {platform.system()}")


def _get_executable_extension() -> str:
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        return ''
    elif platform.system() == 'Windows':
        return '.exe'
    else:
        raise NotImplementedError(f"Unsupported platform: {platform.system()}")


@click.group(cls=HelpGroup, context_settings=dict(help_option_names=['-h', '--help']), invoke_without_command=True)
@click.version_option(__version__, '-V', '--version', message='%(version)s')
@optgroup.group('Configuration', cls=MutuallyExclusiveOptionGroup)
@optgroup.option('-c', '--config', 'config_path', type=click.Path(exists=False, dir_okay=False),
                 help='Config file path.')
@optgroup.option('-p', '--profile', type=click.Choice(list(Profile.__members__)), help='Testing profile.')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose mode.')
def cli(config_path: str, profile: str, verbose: bool):
    """Run Core Pycler."""

    # TODO: Add support for other platforms
    if platform.system() != 'Linux':
        raise NotImplementedError(f"Unsupported platform: {platform.system()}")

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log.info(f"{APP_NAME} version {__version__}")

    if not click.get_current_context().invoked_subcommand:
        if profile:
            config = Profile[profile].value
        else:
            if config_path:
                config = Config.from_file(config_path)
            else:
                log.info("Running with default configuration.")
                config = Config()

        config.print()
        core_test_loop(config)


@cli.command(cls=HelpCommand)
@click.argument('duration', type=click.INT)
def burn(duration: int):
    """Burn CPU using all cores for [DURATION] minutes."""

    burn_test(duration)


@cli.command(cls=HelpCommand)
@click.option('-p', '--path', type=click.Path(exists=False, dir_okay=False), default=None,
              help='Path in which a file will be written.')
def dump_config(path: str):
    """Dump config to a file."""

    if path:
        path = Path(path)
    else:
        path = ROOT_PATH / Config.get_default_filename()
    Config().to_file(path)
    log.info(f'Configuration file written in the: {path}')
