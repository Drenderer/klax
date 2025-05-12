from __future__ import annotations
from abc import ABC
import datetime
import importlib

from jaxtyping import PyTree
import pickle
from pathlib import Path

from ._trainstate import TrainingState


class Callback(ABC):
    """An abstract callback. Inherit from this class to create a custom
    callback."""

    def __call__(self, state: TrainingState) -> bool | None:
        """Called after each step during training."""
        pass

    def on_training_end(self, state: TrainingState) -> None:
        """Called when training ends."""
        pass

    def on_training_start(self, state: TrainingState) -> None:
        """Called when training starts."""
        pass


class HistoryCallback(Callback):
    """Default callback for logging a training process.
    Records training and validation loss histories, as well as the
    trainin time and the last optimizer state."""

    log_every: int
    steps: list
    loss: list
    val_loss: list
    last_start_time: float  # start time of the last training
    last_end_time: float  # End time of the last training
    training_time: float = 0  # Total training time of all trainings
    verbose: bool
    step_offset: int = 0  # Potential offset due to previous trainings
    last_opt_state: PyTree | None = None

    def __init__(self, log_every: int = 100, verbose: bool = True):
        """Initializes the history callback.

        Args:
            log_every: Amount of steps after which the training
                and validation losses are logged.
                (Defaults to 100.)
            verbose: If true prints the training progress and losses.
                (Defaults to True.)
        """
        self.log_every = log_every
        self.verbose = verbose
        self.steps = []
        self.loss = []
        self.val_loss = []

    def __repr__(self):
        """Returns a string representation of the HistoryCallback."""
        return f"HistoryCallback(log_every={self.log_every}, verbose={self.verbose})"

    def __call__(self, state: TrainingState):
        """Called at each step during training.
        Records the training and validation loss, as well as the step count."""
        if state.step % self.log_every == 0:
            self.steps.append(self.step_offset + state.step)
            self.loss.append(state.loss)
            self.val_loss.append(state.val_loss)

            # Print message
            if self.verbose:
                message = f"Step: {state.step}, Loss: {state.loss:.3e}"
                if state.val_data is not None:
                    message += f", Validation loss: {state.val_loss:.3e}"
                print(message)

    def on_training_start(self, state: TrainingState):
        """Called at beginning of training.
        Initializes the training start time.
        """
        self.last_start_time = state.time_on_last_update
        if self.steps:  # If there are already steps, we assume that this is a continuation of a training.
            self.step_offset = self.steps[-1]
        else:
            self(state)

    def on_training_end(self, state: TrainingState):
        """Called at end of training.
        Records the training end time and the last optimizer state."""
        self.last_end_time = state.time_on_last_update
        self.training_time += self.last_end_time - self.last_start_time
        self.last_opt_state = state.opt_state
        if self.verbose:
            print(f"Training took: {datetime.timedelta(seconds=self.training_time)}")

    def plot(self, *, ax=None, loss_options: dict = {}, val_loss_options: dict = {}):
        """Plot the recorded training and validation losses.
        This method uses matplotlib.

        Args:
            ax: Matplotlib axes to plot into. If ``None`` then a new
                axis is created.
                (Defaults to None.)
            loss_options: Dictionary of keyword arguments passed to
                matplotlibs ``plot`` for the training loss.
                (Defaults to {}.)
            val_loss_options: Dictionary of keyword arguments passed to
                matplotlibs ``plot`` for the validation loss.
                (Defaults to {}.)

        Raises:
            ImportError: _description_
        """
        module_name = "matplotlib.pyplot"
        try:
            plt = importlib.import_module(module_name)
            if ax is None:
                _, ax = plt.subplots()
                ax.set(
                    xlabel="Step",
                    ylabel="Loss",
                    yscale="log",
                    title="Training History",
                )
                ax.grid(True)

            # Rename the color option to "c", if it exists. Otherwise "c" and "color" are
            # both passed to plot, which causes an error.
            if "color" in loss_options:
                loss_options["c"] = loss_options.pop("color")
            if "color" in val_loss_options:
                val_loss_options["c"] = val_loss_options.pop("color")

            loss_options = dict(label="Loss", ls="-", c="black") | loss_options
            val_loss_options = (
                dict(label="Validation loss", ls="--", c="red") | val_loss_options
            )
            ax.plot(self.steps, self.loss, **loss_options)
            if any(x is not None for x in self.val_loss):
                ax.plot(self.steps, self.val_loss, **val_loss_options)

        except ImportError as e:
            raise ImportError(
                f"Failed to import module '{module_name}'. "
                f"Install it with: pip install klax[plotting]. "
                f"Original error: {str(e)}"
            )

    def save(
        self, filename: str | Path, overwrite: bool = False, create_dir: bool = True
    ) -> None:
        """Save the HistoryCallback instance to a file using pickle.

        Args:
            filename: The file path where the instance should be saved.
            overwrite: If True, overwrite the file if it already exists.
                If False, raise a FileExistsError if the file exists.
                (Defaults to False.)
            create_dir: If True, create the parent directory if it does not exist.
                (Defaults to True.)

        Raises:
            FileExistsError: If the file already exists and overwrite is False.
            ValueError: If the provided path is not a valid file path.
        """
        filename = Path(filename)

        if filename.exists() and not overwrite:
            raise FileExistsError(
                f"The file '{filename}' already exists. Use overwrite=True to overwrite it."
            )

        if create_dir:
            filename.parent.mkdir(parents=True, exist_ok=True)

        if filename.suffix == "":
            filename = filename.with_suffix(".pkl")
        assert filename.suffix == ".pkl", "File must have a .pkl suffix."

        with filename.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str | Path) -> HistoryCallback:
        """Load a HistoryCallback instance from a file.

        Args:
            filename: The file path from which the instance should be loaded.

        Returns:
            The loaded HistoryCallback instance.

        Raises:
            ValueError: If the file is not a valid pickle file or does not contain a HistoryCallback instance.
        """
        filename = Path(filename)

        with filename.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, HistoryCallback):
            raise ValueError(
                f"The file '{filename}' does not contain a valid HistoryCallback instance."
            )

        return obj
