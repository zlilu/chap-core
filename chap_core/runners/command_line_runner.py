import logging
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path
from time import monotonic, sleep

from chap_core.exceptions import CommandLineException, ModelConfigurationException
from chap_core.hpo.trial_timeout import (
    HpoTrialCleanupError,
    HpoTrialTimeoutError,
    current_trial_timeout_seconds,
    subprocess_trial_deadline,
)
from chap_core.runners.runner import Runner, TrainPredictRunner

logger = logging.getLogger(__name__)

_TERMINATION_SECONDS = 2.0


class CommandLineRunner(Runner):
    def __init__(self, working_dir: str | Path, dry_run=False):
        super().__init__(dry_run=dry_run)
        self._working_dir = working_dir

    def run_command(self, command):
        return self._execute(command, self._working_dir)

    def store_file(self, file_path: str | None = None) -> None:
        pass


def _communicate_finished(process: subprocess.Popen[bytes], timeout: float) -> bool:
    """
    Verify direct Popen child has finished and its captured pipes reached EOF.
    """
    try:
        process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def _process_group_exists(pgid: int) -> bool:
    """
    Verify that every process in the process group is gone.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """
    Terminate a times HPO command and verify process-group cleanup.
    """
    pgid = process.pid

    try:
        # First ask the whole process group to terminate normally.
        with suppress(ProcessLookupError):  # the group may have exited before signaling
            os.killpg(pgid, signal.SIGTERM)
        group_exists = _process_group_exists(pgid)
        if _communicate_finished(process, timeout=_TERMINATION_SECONDS) and not group_exists:
            return
        # If the group has disappeared but communication is still incomplete,
        # something may have escaped the process group.
        if not group_exists:
            raise HpoTrialCleanupError(f"Process group {pgid} disappeared, but the command did not finish")
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        # Check that the direct child finished and pipes closed
        if not _communicate_finished(process, timeout=_TERMINATION_SECONDS):
            raise HpoTrialCleanupError(f"Process group {pgid} did not finish after SIGKILL")
        # Allow a bounded period for the group to disappear
        deadline = monotonic() + _TERMINATION_SECONDS
        while _process_group_exists(pgid):
            if monotonic() >= deadline:
                raise HpoTrialCleanupError(f"Process group {pgid} still exists after SIGKILL")
            sleep(0.05)
    # Preserves deliberately raised cleanup errors with their messages.
    except HpoTrialCleanupError:
        raise
    except Exception as exc:
        # outer hpo meta_learn excepts Exception as failed trial, convert it to HpoTrialCleanupError should abort HPO
        raise HpoTrialCleanupError(f"Failed to clean up HPO trial process group {pgid}") from exc


def run_command(command: str, working_directory=Path("."), env: dict | None = None):
    """Runs a unix command using subprocess.

    If called inside a timed HPO objective, it respects an active HPO trial deadline.

    Parameters
    ----------
    command : str
        The command to run
    working_directory : Path
        The directory to run the command in
    env : dict, optional
        Environment variables to use. If None, uses the current environment.
    """
    logging.debug(f"Running command: {command}")

    process: subprocess.Popen[bytes] | None = None
    timeout_active = False
    communicated = False

    try:
        with subprocess_trial_deadline() as deadline:
            timeout_active = deadline is not None

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_directory,
                shell=True,
                env=env,
                # For timed HPO trials, create an independent process group containing
                # the shell and its descendants (uv, Python/R model process, etc.).
                start_new_session=timeout_active,
            )

            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    timeout_seconds = current_trial_timeout_seconds()
                    raise HpoTrialTimeoutError(
                        f"HPO trial exceeded timeout of {timeout_seconds:g} seconds while starting command: {command}"
                    )
            try:
                stdout, stderr = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                timeout_seconds = current_trial_timeout_seconds()
                raise HpoTrialTimeoutError(
                    f"HPO trial exceeded timeout of {timeout_seconds:g} seconds while running command: {command}"
                ) from exc
            communicated = True
    except BaseException:
        # Clean up on Ctrl-C or another unexpected exception while a timed trial has started but communicate did not finish normally.
        if process is not None and timeout_active and not communicated:
            _terminate_process_group(process)
        raise

    # Model output is not guaranteed to be valid UTF-8 (locale-dependent R
    # warnings, for instance); a failed model must still produce a readable
    # error message rather than a UnicodeDecodeError.
    output = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
    return_code = process.returncode

    if return_code != 0:
        message = (
            f"Command '{command}' failed with return code {return_code}, "
            f"Full output from command below: \n ----- \n{output} \n--------"
        )
        logger.error(message)
        raise CommandLineException(message)

    return output


class CommandLineTrainPredictRunner(TrainPredictRunner):
    def __init__(
        self,
        runner: Runner,
        train_command: str,
        predict_command: str,
        model_configuration_filename: str | None = None,
        report_command: str | None = None,
    ):
        self._runner = runner
        self._train_command = train_command
        self._predict_command = predict_command
        self._model_configuration_filename = model_configuration_filename
        self._report_command = report_command

    def _format_command(self, command, keys):
        try:
            return command.format(**keys)
        except KeyError as e:
            raise ModelConfigurationException(
                f"Was not able to format command {command}. Does the command contain wrong keys or keys that there is not data for in the dataset?"
            ) from e

    def _handle_polygons(self, command, keys, polygons_file_name=None):
        # adds polygons to keys if polygons exist. Does some checking with compatibility with command
        if polygons_file_name is not None:
            if "{polygons}" not in command:
                logger.warning(
                    f"Dataset has polygons, but command {command} does not ask for polygons. Will not insert polygons into command."
                )
            else:
                keys["polygons"] = polygons_file_name
        return keys

    def _handle_config(self, command, keys):
        if "{model_config}" not in command:
            return keys
        keys["model_config"] = self._model_configuration_filename
        return keys

    def train(self, train_file_name, model_file_name, polygons_file_name=None):
        keys = {"train_data": train_file_name, "model": model_file_name}
        keys = self._handle_polygons(self._train_command, keys, polygons_file_name)
        keys = self._handle_config(self._train_command, keys)
        command = self._format_command(self._train_command, keys)
        logger.debug(f"Running command {command}")
        return self._runner.run_command(command)

    def predict(self, model_file_name, historic_data, future_data, output_file, polygons_file_name=None):
        keys = {
            "historic_data": historic_data,
            "future_data": future_data,
            "model": model_file_name,
            "out_file": output_file,
        }
        keys = self._handle_polygons(self._predict_command, keys, polygons_file_name)
        keys = self._handle_config(self._predict_command, keys)
        command = self._format_command(self._predict_command, keys)
        return self._runner.run_command(command)

    def report(self, model_file_name, historic_data, output_file, polygons_file_name=None):
        if self._report_command is None:
            raise NotImplementedError("This runner does not support report generation")
        keys = {
            "model": model_file_name,
            "historic_data": historic_data,
            "out_file": output_file,
        }
        keys = self._handle_polygons(self._report_command, keys, polygons_file_name)
        keys = self._handle_config(self._report_command, keys)
        command = self._format_command(self._report_command, keys)
        logger.debug(f"Running command {command}")
        return self._runner.run_command(command)
