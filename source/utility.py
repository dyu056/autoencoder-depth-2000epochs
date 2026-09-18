"""Reusable checkpoint, tensor, analysis, and plotting utilities.

Canonical dimension conventions
-------------------------------
Image: ``[C, H, W]``
    ``C`` is the color channel (1 for grayscale, 3 for RGB, 4 for RGBA),
    ``H`` is image height, and ``W`` is image width.

Video: ``[T, C, H, W]``
    ``T`` is the batch of consecutive frames (time), ``C`` is the color
    channel, ``H`` is frame height, and ``W`` is frame width.

Channel-last ``HWC`` and ``THWC`` tensors are also supported when explicitly
selected with the ``layout`` argument.
"""

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ArrayLike = torch.Tensor | np.ndarray


def _to_numpy(value: ArrayLike, expected_ndim: int, name: str) -> np.ndarray:
    """Detach a tensor and move it to CPU without modifying the input."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray")

    if array.ndim != expected_ndim:
        raise ValueError(
            f"{name} must have {expected_ndim} dimensions, got shape {array.shape}"
        )
    return array


def _image_to_hwc(image: ArrayLike, layout: str) -> np.ndarray:
    """Convert a three-dimensional image from CHW/HWC to matplotlib format."""
    array = _to_numpy(image, expected_ndim=3, name="image")
    normalized_layout = layout.upper()
    if normalized_layout == "CHW":
        array = np.moveaxis(array, 0, -1)
    elif normalized_layout != "HWC":
        raise ValueError("image layout must be 'CHW' or 'HWC'")

    channels = array.shape[-1]
    if channels not in (1, 3, 4):
        raise ValueError(
            "image channel dimension must contain 1 (grayscale), 3 (RGB), "
            f"or 4 (RGBA) channels, got shape {array.shape}"
        )
    return array[..., 0] if channels == 1 else array


def _video_to_thwc(video: ArrayLike, layout: str) -> np.ndarray:
    """Convert a four-dimensional video from TCHW/THWC to animation format."""
    array = _to_numpy(video, expected_ndim=4, name="video")
    normalized_layout = layout.upper()
    if normalized_layout == "TCHW":
        array = np.moveaxis(array, 1, -1)
    elif normalized_layout != "THWC":
        raise ValueError("video layout must be 'TCHW' or 'THWC'")

    if array.shape[0] == 0:
        raise ValueError("video must contain at least one frame along T")
    channels = array.shape[-1]
    if channels not in (1, 3, 4):
        raise ValueError(
            "video channel dimension must contain 1 (grayscale), 3 (RGB), "
            f"or 4 (RGBA) channels, got shape {array.shape}"
        )
    return array[..., 0] if channels == 1 else array


def plot_image(
    image: ArrayLike,
    *,
    layout: str = "CHW",
    title: str | None = None,
    ax: Axes | None = None,
    show: bool = True,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
) -> tuple[Figure, Axes]:
    """Plot one image tensor.

    Args:
        image: A three-dimensional ``[C, H, W]`` tensor by default. ``C`` is
            color, ``H`` is height, and ``W`` is width. Pass ``layout="HWC"``
            for ``[H, W, C]`` input.
        layout: Dimension order, either ``"CHW"`` or ``"HWC"``.
        title: Optional title shown above the image.
        ax: Optional existing matplotlib axes on which to draw.
        show: Whether to call ``plt.show()``. Set to ``False`` when composing
            this plot into a larger figure or when testing.
        cmap: Optional matplotlib color map. Single-channel images default to
            ``"gray"``; multi-channel images ignore color maps.
        vmin: Optional lower bound of the displayed value range.
        vmax: Optional upper bound of the displayed value range.
        colorbar: Whether to add a color bar. Most useful for latent heatmaps.

    Returns:
        The matplotlib ``(figure, axes)`` pair.
    """
    display_image = _image_to_hwc(image, layout)
    if ax is None:
        figure, ax = plt.subplots(figsize=(5, 5))
    else:
        figure = ax.figure

    selected_cmap = (
        cmap if cmap is not None else ("gray" if display_image.ndim == 2 else None)
    )
    image_artist = ax.imshow(
        display_image,
        cmap=selected_cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if title is not None:
        ax.set_title(title)
    ax.axis("off")
    if colorbar:
        figure.colorbar(image_artist, ax=ax)
        figure.tight_layout()
    if show:
        plt.show()
    return figure, ax


def plot_video(
    video: ArrayLike,
    *,
    layout: str = "TCHW",
    fps: float = 30,
    title: str | Callable[[int], str] | None = None,
    max_display_frames: int | None = None,
    figsize: tuple[float, float] = (5, 5),
) -> HTML:
    """Render a four-dimensional tensor as a continuous notebook video.

    Args:
        video: A four-dimensional ``[T, C, H, W]`` tensor by default. ``T``
            is the batch of consecutive frames/time, ``C`` is color, ``H`` is
            frame height, and ``W`` is frame width. Pass ``layout="THWC"`` for
            ``[T, H, W, C]`` input.
        layout: Dimension order, either ``"TCHW"`` or ``"THWC"``.
        fps: Playback frames per second.
        title: A fixed title or a function receiving the original frame index.
        max_display_frames: Optional display-only limit. Frames are sampled
            evenly while preserving the video's original duration.
        figsize: Matplotlib figure size in inches.

    Returns:
        ``IPython.display.HTML`` containing play/pause controls and the video.
        In a notebook, return it as the last expression or call ``display``.
    """
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if max_display_frames is not None and max_display_frames <= 0:
        raise ValueError("max_display_frames must be greater than zero")

    frames = _video_to_thwc(video, layout)
    stride = (
        1
        if max_display_frames is None
        else max(1, int(np.ceil(len(frames) / max_display_frames)))
    )
    frame_indices = np.arange(0, len(frames), stride)
    effective_fps = fps / stride

    figure, ax = plt.subplots(figsize=figsize)
    first_frame = frames[frame_indices[0]]
    image_artist = ax.imshow(
        first_frame, cmap="gray" if first_frame.ndim == 2 else None
    )
    ax.axis("off")

    def update(display_index: int):
        frame_index = int(frame_indices[display_index])
        image_artist.set_data(frames[frame_index])
        if callable(title):
            ax.set_title(title(frame_index))
        elif title is not None:
            ax.set_title(title)
        return (image_artist,)

    update(0)
    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000 / effective_fps,
        blit=False,
    )
    html = HTML(animation.to_jshtml(fps=effective_fps))
    plt.close(figure)
    return html


def plot_conv_kernels(
    module: torch.nn.Module,
    *,
    max_columns: int = 16,
    show: bool = True,
) -> list[tuple[str, Figure, Axes]]:
    """Plot every output filter in each Conv2d/Conv3d layer of a module.

    PyTorch stores a Conv3d weight tensor as
    ``[C_out, C_in, K_t, K_h, K_w]``. To produce one visible 2D tile for each
    output filter, this function averages over ``C_in`` and ``K_t``. Conv2d
    weights use ``[C_out, C_in, K_h, K_w]`` and are averaged over ``C_in``.

    Tiles are placed in row-major order. For example, with 16 columns, output
    channel 17 is located at row 1, column 1. All tiles from the same layer use
    a shared, zero-centered color scale so their magnitudes remain comparable.

    Args:
        module: Model or submodule containing Conv2d and/or Conv3d layers.
        max_columns: Maximum number of output filters shown in each grid row.
        show: Whether to call ``plt.show()`` for each convolutional layer.

    Returns:
        A list of ``(layer_name, figure, axes)`` tuples, one per convolution.
    """
    if max_columns <= 0:
        raise ValueError("max_columns must be greater than zero")

    plots: list[tuple[str, Figure, Axes]] = []
    for layer_name, layer in module.named_modules():
        if not isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv3d)):
            continue

        weights = layer.weight.detach().float().cpu()
        if isinstance(layer, torch.nn.Conv3d):
            filters = weights.mean(dim=(1, 2)).numpy()
            reduction = "mean over C_in and K_t"
        else:
            filters = weights.mean(dim=1).numpy()
            reduction = "mean over C_in"

        output_channels, kernel_height, kernel_width = filters.shape
        columns = min(max_columns, output_channels)
        rows = int(np.ceil(output_channels / columns))
        vertical_step = kernel_height + 1
        horizontal_step = kernel_width + 1
        sheet = np.full(
            (rows * vertical_step - 1, columns * horizontal_step - 1),
            np.nan,
            dtype=np.float32,
        )

        scale = float(np.abs(filters).max())
        normalized_filters = filters if scale == 0 else filters / scale
        for channel, kernel in enumerate(normalized_filters):
            row, column = divmod(channel, columns)
            top = row * vertical_step
            left = column * horizontal_step
            sheet[top : top + kernel_height, left : left + kernel_width] = kernel

        figure_width = max(6.0, columns * 0.55)
        figure_height = max(3.0, rows * 0.55 + 1.5)
        figure, ax = plt.subplots(figsize=(figure_width, figure_height))
        color_map = plt.get_cmap("coolwarm").copy()
        color_map.set_bad(color="#dddddd")
        image_artist = ax.imshow(sheet, cmap=color_map, vmin=-1, vmax=1)

        ax.set_xticks(
            np.arange(columns) * horizontal_step + (kernel_width - 1) / 2,
            labels=range(columns),
        )
        ax.set_yticks(
            np.arange(rows) * vertical_step + (kernel_height - 1) / 2,
            labels=(row * columns for row in range(rows)),
        )
        ax.set_xlabel("Column offset; output channel = row label + column")
        ax.set_ylabel("First output channel in row")
        display_name = layer_name or "<root>"
        ax.set_title(
            f"{display_name}: {type(layer).__name__} {tuple(weights.shape)}\n"
            f"one tile per C_out ({reduction})"
        )
        figure.colorbar(image_artist, ax=ax, label="weight / max(abs(weight))")
        figure.tight_layout()
        if show:
            plt.show()
        plots.append((display_name, figure, ax))

    if not plots:
        raise ValueError("module does not contain any Conv2d or Conv3d layers")
    return plots


def plot_loss_history(
    history: dict[str, list[float]],
    *,
    title: str = "Train vs. validation loss",
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot train/test loss arrays stored by ``train_autoencoder``."""
    epochs = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    test_loss = history.get("test_loss", [])
    if not epochs or len(epochs) != len(train_loss) or len(epochs) != len(test_loss):
        raise ValueError(
            "history must contain equally sized epoch/train_loss/test_loss"
        )
    figure, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train loss")
    ax.plot(epochs, test_loss, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    figure.tight_layout()
    if show:
        plt.show()
    return figure, ax


def play_dataset_item(
    actions: ArrayLike,
    frames: ArrayLike,
    instruction: str,
    fps: float = 30,
    max_display_frames: int = 300,
) -> HTML:
    """Play one dataset item whose frames use ``[T, H, W, C]`` dimensions."""
    action_array = _to_numpy(actions, expected_ndim=2, name="actions")
    frame_array = _to_numpy(frames, expected_ndim=4, name="frames")
    if len(action_array) != len(frame_array):
        raise ValueError(
            f"The item has {len(frame_array)} frames but {len(action_array)} actions"
        )

    def frame_title(frame_index: int) -> str:
        action_text = np.array2string(action_array[frame_index], precision=3)
        return f"Frame {frame_index}/{len(frame_array) - 1} | action {action_text}"

    print(f"Instruction: {instruction}")
    print(f"Frames: {len(frame_array)}, actions: {action_array.shape}")
    return plot_video(
        frame_array,
        layout="THWC",
        fps=fps,
        title=frame_title,
        max_display_frames=max_display_frames,
    )


def analysis_device() -> torch.device:
    """Select the default device used by analysis scripts."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latest_epoch_checkpoint(
    experiment_name: str,
    *,
    checkpoints_root: Path | str = "checkpoints",
) -> Path:
    """Return the numerically latest ``epoch_<number>.pt`` checkpoint."""
    checkpoint_dir = Path(checkpoints_root) / experiment_name
    paths = list(checkpoint_dir.glob("epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    try:
        return max(paths, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
    except ValueError as error:
        raise ValueError(
            f"Invalid epoch checkpoint name in {checkpoint_dir}"
        ) from error


def load_experiment(
    model: torch.nn.Module,
    experiment_name: str,
    *,
    device: torch.device | None = None,
    checkpoints_root: Path | str = "checkpoints",
    results_root: Path | str = "results",
):
    """Load a model, checkpoint, and optional JSON history for analysis."""
    device = device or analysis_device()
    checkpoint_path = latest_epoch_checkpoint(
        experiment_name, checkpoints_root=checkpoints_root
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    history = checkpoint.get("history", {})
    history_path = Path(results_root) / experiment_name / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    return model, checkpoint, history, device


def sample_frames(dataset, index: int) -> torch.Tensor:
    """Extract a THWC frame tensor from either supported dataset item shape."""
    item = dataset[index]
    frames = item if isinstance(item, torch.Tensor) else item[1]
    if frames.ndim != 4:
        raise ValueError(f"Expected a THWC clip, got {tuple(frames.shape)}")
    return frames


def prepare_video_input(frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert one THWC clip or a batch of THWC clips to normalized BCTHW."""
    if frames.ndim == 4:
        frames = frames.unsqueeze(0)
    if frames.ndim != 5:
        raise ValueError(f"Expected THWC or BTHWC input, got {tuple(frames.shape)}")
    if frames.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"Expected channel-last video input, got {tuple(frames.shape)}"
        )
    return (
        frames[..., :3]
        .permute(0, 4, 1, 2, 3)
        .contiguous()
        .to(device=device, dtype=torch.float32)
        / 255.0
    )


@torch.no_grad()
def reconstruction_inference(model, frames, device):
    """Run the standard encode/decode reconstruction inference test."""
    inputs = prepare_video_input(frames, device)
    latent = model.encode(inputs)
    reconstruction = model.decode(latent)
    return inputs, latent, reconstruction


def plot_reconstruction_triptych(
    inputs,
    reconstruction,
    *,
    frame_index: int,
    output_path: Path | str | None = None,
    show: bool = True,
):
    """Plot original, reconstruction, and mean absolute RGB error."""
    frame_index = min(frame_index, inputs.shape[2] - 1)
    original = inputs[0, :, frame_index].permute(1, 2, 0).detach().cpu()
    rebuilt = reconstruction[0, :, frame_index].permute(1, 2, 0).detach().cpu()
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original.clamp(0, 1))
    axes[1].imshow(rebuilt.clamp(0, 1))
    axes[2].imshow((rebuilt - original).abs().mean(2), cmap="magma")
    for axis, title in zip(axes, ("Original", "Reconstruction", "RGB error")):
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    return figure, axes


@torch.no_grad()
def analyze_reconstruction(
    model,
    frames,
    device,
    *,
    frame_index: int = 16,
    fps: float = 30,
):
    """Run reconstruction inference and display standard image/video plots."""
    inputs, latent, reconstruction = reconstruction_inference(model, frames, device)
    mse = F.mse_loss(reconstruction, inputs).item()
    print(f"latent={tuple(latent.shape)}, reconstruction MSE={mse:.8g}")
    plot_reconstruction_triptych(
        inputs, reconstruction, frame_index=frame_index, show=True
    )
    display(plot_video(inputs[0].permute(1, 0, 2, 3), fps=fps))
    display(plot_video(reconstruction[0].permute(1, 0, 2, 3).clamp(0, 1), fps=fps))
    return inputs, latent, reconstruction


def analyze_history(history, experiment_name: str):
    """Display the standard train/test loss history for one experiment."""
    return plot_loss_history(history, title=f"{experiment_name}: loss history")


def compare_latents(latent_a, latent_b):
    """Compute and print standard pairwise latent-vector metrics."""
    a, b = latent_a.flatten(), latent_b.flatten()
    metrics = {
        "cosine_similarity": F.cosine_similarity(a, b, dim=0).item(),
        "l2_distance": torch.linalg.vector_norm(b - a).item(),
        "mean_absolute_difference": (b - a).abs().mean().item(),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def write_json_summary(value, output_path: Path | str) -> Path:
    """Create the parent directory and write an analysis JSON artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output_path


def save_figure(figure: Figure, output_path: Path | str, *, dpi: int = 180) -> Path:
    """Create the parent directory, save a figure, and close it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path
