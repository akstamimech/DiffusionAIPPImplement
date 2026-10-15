from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from visualtorch import layered_view

import SparseDiffusion as diffusion


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "architecture_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_box(draw, xy, title, body="", fill="#f7fbff", outline="#22577a", font=None, small_font=None):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=outline, width=2)
    if font is None:
        font = ImageFont.load_default()
    if small_font is None:
        small_font = font

    title_w, _ = text_size(draw, title, font)
    draw.text((x0 + (x1 - x0 - title_w) / 2, y0 + 12), title, fill="#102a43", font=font)

    if body:
        lines = body.split("\n")
        y = y0 + 42
        for line in lines:
            line_w, _ = text_size(draw, line, small_font)
            draw.text((x0 + (x1 - x0 - line_w) / 2, y), line, fill="#243b53", font=small_font)
            y += 18


def draw_arrow(draw, start, end, fill="#334e68", width=3):
    draw.line([start, end], fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):
        direction = 1 if x1 >= x0 else -1
        points = [(x1, y1), (x1 - 12 * direction, y1 - 6), (x1 - 12 * direction, y1 + 6)]
    else:
        direction = 1 if y1 >= y0 else -1
        points = [(x1, y1), (x1 - 6, y1 - 12 * direction), (x1 + 6, y1 - 12 * direction)]
    draw.polygon(points, fill=fill)


def add_caption(image, caption):
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    caption_h = 38
    out = Image.new("RGB", (w, h + caption_h), "white")
    out.paste(image, (0, caption_h))
    draw = ImageDraw.Draw(out)
    text_w, _ = text_size(draw, caption, font)
    draw.text(((w - text_w) / 2, 12), caption, fill="#102a43", font=font)
    return out


def make_visualtorch_panels(model):
    map_branch = nn.Sequential(
        model.mean_var_cnn.conv1,
        model.mean_var_cnn.act1,
        model.mean_var_cnn.pool1,
        model.mean_var_cnn.conv2,
        model.mean_var_cnn.act2,
        model.mean_var_cnn.pool2,
        model.mean_var_cnn.conv3,
        model.mean_var_cnn.act3,
        model.mean_var_cnn.pool3,
    )

    denoiser_main_path = nn.Sequential(
        nn.Conv1d(2, 64, kernel_size=3, padding=1),
        nn.GroupNorm(8, 64),
        nn.SiLU(),
        nn.Conv1d(64, 64, kernel_size=3, padding=1),
        nn.GroupNorm(8, 64),
        nn.SiLU(),
        nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.Conv1d(128, 128, kernel_size=3, padding=1),
        nn.GroupNorm(8, 128),
        nn.SiLU(),
        nn.Conv1d(128, 128, kernel_size=3, padding=1),
        nn.GroupNorm(8, 128),
        nn.SiLU(),
        nn.Conv1d(128, 64, kernel_size=3, padding=1),  # visualization proxy for ConvTranspose1d upsample
        nn.Conv1d(64, 128, kernel_size=1),  # visualization proxy for concatenating the skip path
        nn.Conv1d(128, 64, kernel_size=3, padding=1),
        nn.GroupNorm(8, 64),
        nn.SiLU(),
        nn.Conv1d(64, 64, kernel_size=3, padding=1),
        nn.GroupNorm(8, 64),
        nn.SiLU(),
        nn.Conv1d(64, 2, kernel_size=1),
    )

    map_panel = layered_view(
        map_branch,
        input_shape=(2, 51, 51),
        to_file=str(OUTPUT_DIR / "MeanVarCNN_map_branch_visualtorch.png"),
        draw_volume=True,
        legend=True,
        scale_xy=2.2,
        scale_z=0.7,
        spacing=18,
    ).convert("RGB")

    denoiser_panel = layered_view(
        denoiser_main_path,
        input_shape=(1, *diffusion.TARGET_SHAPE),
        to_file=str(OUTPUT_DIR / "NoisePredictor_main_path_visualtorch.png"),
        draw_volume=False,
        legend=True,
        min_xy=28,
        max_xy=170,
        scale_xy=36,
        spacing=28,
    ).convert("RGB")

    return add_caption(map_panel, "VisualTorch: Mean/Log-Var Map CNN Branch"), add_caption(
        denoiser_panel,
        "VisualTorch: 1D Waypoint Denoiser Main Path (FiLM and skip connections annotated below)",
    )


def resize_to_width(image, width):
    if image.width == width:
        return image
    height = max(1, int(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def build_overview(model):
    map_panel, denoiser_panel = make_visualtorch_panels(model)
    map_panel = resize_to_width(map_panel, 760)
    denoiser_panel = resize_to_width(denoiser_panel, 760)

    W = 1700
    H = 1200 + map_panel.height + denoiser_panel.height
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    small = ImageFont.load_default()

    title = "SparseDiffusion Architecture Overview"
    tw, _ = text_size(draw, title, font)
    draw.text(((W - tw) / 2, 24), title, fill="#102a43", font=font)

    draw_box(
        draw,
        (60, 80, 390, 180),
        "Raw Dataset Fields",
        "current_mean [B,51,51]\ncurrent_var [B,51,51]\ncurrent_position [B,2]\ncontrol_waypoints [B,2,8]",
        fill="#fff8e6",
        font=font,
        small_font=small,
    )
    draw_box(
        draw,
        (500, 80, 860, 180),
        "Normalization",
        "log_var = log(current_var + eps)\nmean/log_var standardized\nposition scaled to [-1,1]\nwaypoints scaled to [-1,1]",
        fill="#f0fff4",
        font=font,
        small_font=small,
    )
    draw_arrow(draw, (390, 130), (500, 130))

    draw_box(
        draw,
        (980, 80, 1310, 180),
        "Diffusion Corruption",
        "x_t = sqrt(alpha_bar_t) x_0\n    + sqrt(1-alpha_bar_t) eps\nt sampled from 0..999",
        fill="#f7fbff",
        font=font,
        small_font=small,
    )
    draw_arrow(draw, (860, 130), (980, 130))

    draw_box(
        draw,
        (60, 250, 390, 350),
        "Map Tensor",
        "meanvar_map [B,2,51,51]\nchannels: mean, log_var",
        fill="#edf2ff",
        font=font,
        small_font=small,
    )
    draw_box(
        draw,
        (60, 400, 390, 500),
        "Current Position MLP",
        "[B,2] -> Linear 32 -> SiLU\n-> Linear 32 -> SiLU\noutput [B,32]",
        fill="#edf2ff",
        font=font,
        small_font=small,
    )
    draw_box(
        draw,
        (60, 550, 390, 650),
        "Timestep MLP",
        "sin/cos embedding [B,64]\nLinear 256 -> SiLU\nLinear 256 -> SiLU",
        fill="#edf2ff",
        font=font,
        small_font=small,
    )

    canvas.paste(map_panel, (500, 230))
    draw_box(
        draw,
        (1300, 300, 1620, 430),
        "Map Encoder Output",
        "[B,64,12,12] -> flatten [B,9216]\nconcat position [B,32]\nLinear 9248 -> 256\nLayerNorm + Linear -> [B,256]",
        fill="#e6fffa",
        font=font,
        small_font=small,
    )
    draw_arrow(draw, (390, 300), (500, 300))
    draw_arrow(draw, (390, 450), (1300, 370))

    draw_box(
        draw,
        (1300, 530, 1620, 650),
        "Condition Vector",
        "concat time [B,256]\n+ map/position [B,256]\nLinear 512 -> 256 -> 256\nused as FiLM condition",
        fill="#e6fffa",
        font=font,
        small_font=small,
    )
    draw_arrow(draw, (390, 600), (1300, 590))
    draw_arrow(draw, (1460, 430), (1460, 530))

    denoiser_y = 760
    canvas.paste(denoiser_panel, (500, denoiser_y))
    draw_box(
        draw,
        (60, denoiser_y + 90, 390, denoiser_y + 210),
        "Noisy Sparse Waypoints",
        "x_t [B,2,8]\nConv1d over 8 control columns\n2 coordinate channels",
        fill="#fff0f6",
        font=font,
        small_font=small,
    )
    draw_arrow(draw, (390, denoiser_y + 150), (500, denoiser_y + 150))

    draw_box(
        draw,
        (1300, denoiser_y + 90, 1620, denoiser_y + 250),
        "FiLM Modulation + Skips",
        "ConditionalResBlock1D receives\ncondition [B,256]\ncond_layer -> gamma,beta\nx = (1+gamma)x + beta\nskip connection around blocks",
        fill="#f3f0ff",
        font=font,
        small_font=small,
    )
    draw_arrow(draw, (1460, 650), (1460, denoiser_y + 90))

    draw_box(
        draw,
        (1300, denoiser_y + 340, 1620, denoiser_y + 450),
        "Prediction + Loss",
        "noise_pred [B,2,8]\nweighted epsilon loss\n+ spline-noise auxiliary loss",
        fill="#fff8e6",
        font=font,
        small_font=small,
    )
    draw_arrow(draw, (1260, denoiser_y + 270), (1300, denoiser_y + 395))

    footer = f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    fw, _ = text_size(draw, footer, font)
    draw.text(((W - fw) / 2, H - 50), footer, fill="#102a43", font=font)

    output_path = OUTPUT_DIR / "SparseDiffusion_visualtorch_overview.png"
    canvas.save(output_path)
    return output_path


def main():
    torch.set_grad_enabled(False)
    model = diffusion.NoisePredictor().cpu().eval()
    output_path = build_overview(model)
    print(f"Saved architecture overview to {output_path}")


if __name__ == "__main__":
    main()
