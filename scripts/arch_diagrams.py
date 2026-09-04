"""Generate architecture diagrams for YOLO11 model variants using matplotlib."""

import yaml
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = Path("reports/architecture")
MODELS_DIR = Path("segab_yolo/cfg/models")

COLORS = {
    "backbone": "#D6EAF8",
    "neck": "#D5F5E3",
    "attention": "#F9E79F",
    "detect": "#E8DAEF",
    "sppf": "#ABEBC6",
    "concat": "#ABB2B9",
    "edge_fpn": "#5DADE2",
    "edge_seq": "#2C3E50",
}

MODEL_INFO = {
    "yolo11n": {"file": "11/yolo11.yaml", "title": "YOLO11n (Baseline)", "desc": "Standard YOLO11 nano senza attenzione"},
    "yolo11n-gam": {"file": "11/yolo11-gam.yaml", "title": "YOLO11n-GAM", "desc": "GAM nel neck dopo fusioni top-down"},
    "yolo11n-gam-bbone": {"file": "11/yolo11-gam-bbone.yaml", "title": "YOLO11n-GAM-BBone", "desc": "GAM alla fine del backbone"},
    "yolo11n-simam": {"file": "11/yolo11-simam.yaml", "title": "YOLO11n-SimAM", "desc": "SimAM nel neck dopo prima fusione"},
    "yolo11n-simam-bbone": {"file": "11/yolo11-simam-bbone.yaml", "title": "YOLO11n-SimAM-BBone", "desc": "SimAM alla fine del backbone"},
    "yolo26n": {"file": "26/yolo26.yaml", "title": "YOLO26n", "desc": "YOLO26 nano (end2end, reg_max=1)"},
}

MODULE_SHORT = {
    "Conv": "Conv",
    "C3k2": "C3k2",
    "SPPF": "SPPF",
    "C2PSA": "C2PSA",
    "nn.Upsample": "Up",
    "Concat": "Concat",
    "Detect": "Detect",
    "GAM": "GAM",
    "SimAM": "SimAM",
    "EMA": "EMA",
    "ECAAttention": "ECA",
    "ShuffleAttention": "SA",
    "BiLevelRoutingAttention": "BRA",
    "ResBlock_CBAM": "Res+CBAM",
}

def get_module_color(module, section):
    if module in ("GAM", "SimAM", "EMA", "ECAAttention", "ShuffleAttention", "BiLevelRoutingAttention", "ResBlock_CBAM"):
        return COLORS["attention"]
    if module == "Detect":
        return COLORS["detect"]
    if module == "Concat":
        return COLORS["concat"]
    if module == "SPPF":
        return COLORS["sppf"]
    if section == "backbone":
        return COLORS["backbone"]
    return COLORS["neck"]


def generate_diagram(model_name, data, info, scale_n):
    """Generate a single architecture diagram."""
    _, width, max_ch = scale_n
    backbone = data.get("backbone", [])
    head = data.get("head", [])
    offset = len(backbone)

    fig, ax = plt.subplots(figsize=(8, max(6, (len(backbone) + len(head)) * 0.5)))
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, len(backbone) + len(head) + 2)
    ax.axis("off")
    ax.set_title(f"{info['title']}\n{info['desc']}", fontsize=13, fontweight="bold", pad=10)

    box_h = 0.7
    box_w = 3.0

    def draw_block(y, text, color, edgecolor="#555", linewidth=1.5, fontsize=9):
        rect = FancyBboxPatch(
            (0.5, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor=edgecolor, linewidth=linewidth,
        )
        ax.add_patch(rect)
        ax.text(0.5 + box_w / 2, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")

    def draw_arrow(y_from, y_to, x_offset=0, style="solid", color="#2C3E50"):
        ax.annotate(
            "", xy=(0.5 + box_w / 2 + x_offset, y_to),
            xytext=(0.5 + box_w / 2 + x_offset, y_from),
            arrowprops=dict(arrowstyle="->", linestyle=style, color=color, lw=1.5),
        )

    def draw_fpn_arrow(y_from, y_to, label=""):
        mid_y = (y_from + y_to) / 2
        ax.annotate(
            "", xy=(4.5, y_to), xytext=(4.5, y_from),
            arrowprops=dict(arrowstyle="->", color=COLORS["edge_fpn"], lw=1.2, linestyle="dashed"),
        )
        if label:
            ax.text(4.7, mid_y, label, fontsize=7, color=COLORS["edge_fpn"], va="center")

    # Draw backbone layers (bottom to top)
    back_y = {}
    current_y = len(backbone) + len(head) - 1
    for i, layer in enumerate(reversed(backbone)):
        _, repeats, module, args = layer[0], layer[1], layer[2], layer[3]
        label = MODULE_SHORT.get(module, module)
        if args and isinstance(args[0], (int, float)):
            c = min(round(args[0] * width), max_ch)
            label += f" ({c}ch)"
        if repeats > 1:
            label += f" x{repeats}"
        color = get_module_color(module, "backbone")
        draw_block(current_y, label, color)
        back_y[len(backbone) - 1 - i] = current_y
        current_y -= 1

    # Draw head layers
    head_y = {}
    for i, layer in enumerate(head):
        _, repeats, module, args = layer[0], layer[1], layer[2], layer[3]
        label = MODULE_SHORT.get(module, module)
        if args and isinstance(args[0], (int, float)):
            c = min(round(args[0] * width), max_ch)
            label += f" ({c}ch)"
        if repeats > 1:
            label += f" x{repeats}"
        color = get_module_color(module, "head")
        draw_block(current_y, label, color)
        head_y[i] = current_y
        current_y -= 1

    # Draw sequential edges
    all_ys = sorted(set(back_y.values()) | set(head_y.values()), reverse=True)
    for i in range(len(all_ys) - 1):
        if all_ys[i] - all_ys[i+1] < 2:  # adjacent blocks
            draw_arrow(all_ys[i] - 0.35, all_ys[i+1] + 0.35)

    # Draw FPN connections
    for i, layer in enumerate(head):
        from_idx = layer[0]
        if isinstance(from_idx, list):
            for src in from_idx:
                src_y = None
                if src == -1:
                    continue  # sequential already drawn
                elif src < offset:
                    src_y = back_y.get(src)
                else:
                    src_h = src - offset
                    if 0 <= src_h < len(head):
                        src_y = head_y.get(src_h)

                dst_y = head_y.get(i)
                if src_y is not None and dst_y is not None and abs(src_y - dst_y) > 1:
                    pos = src_y if src_y > dst_y else src_y
                    label = f"[{src}]"
                    draw_fpn_arrow(src_y, dst_y, label)

    # Legend
    legend_y = current_y - 1
    legend_items = [
        ("Backbone", COLORS["backbone"]),
        ("Neck (FPN)", COLORS["neck"]),
        ("Attenzione", COLORS["attention"]),
        ("Detection Head", COLORS["detect"]),
    ]
    ax.text(0.5, legend_y, "Legenda:", fontsize=9, fontweight="bold", va="top")
    for li, (lbl, clr) in enumerate(legend_items):
        y_pos = legend_y - 0.8 - li * 0.6
        rect = FancyBboxPatch(
            (0.8, y_pos - 0.2), 0.5, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=clr, edgecolor="#555", linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(1.5, y_pos, lbl, fontsize=8, va="center")

    output_path = str(OUTPUT_DIR / f"{model_name}_arch.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_comparison():
    """Generate a side-by-side comparison of all variants."""
    models = [
        ("yolo11n", "YOLO11n"),
        ("yolo11n-gam", "+GAM(Neck)"),
        ("yolo11n-simam", "+SimAM(Neck)"),
        ("yolo11n-gam-bbone", "+GAM(BBone)"),
        ("yolo11n-simam-bbone", "+SimAM(BBone)"),
        ("yolo26n", "YOLO26n"),
    ]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models * 2.8, 8))
    fig.suptitle("Confronto Architetture — Modifiche rispetto a YOLO11n", fontsize=14, fontweight="bold", y=0.98)

    for idx, (model_name, label) in enumerate(models):
        ax = axes[idx]
        data = load_yaml(model_name)
        scale = data.get("scales", {}).get("n", [0.5, 0.25, 1024])
        backbone = data.get("backbone", [])
        head = data.get("head", [])
        offset = len(backbone)

        ax.set_xlim(-0.5, 4)
        ax.set_ylim(-1, len(backbone) + len(head) + 1)
        ax.axis("off")
        ax.set_title(label, fontsize=9, fontweight="bold")

        box_h = 0.5
        box_w = 2.5

        def draw_block(y, text, color, fontsize=7):
            rect = FancyBboxPatch(
                (0, y - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="#555", linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(box_w / 2, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")

        current_y = len(backbone) + len(head) - 1

        for i, layer in enumerate(reversed(backbone)):
            module = layer[2]
            label_text = MODULE_SHORT.get(module, module)
            color = get_module_color(module, "backbone")
            draw_block(current_y, label_text, color)
            current_y -= 1

        for i, layer in enumerate(head):
            module = layer[2]
            label_text = MODULE_SHORT.get(module, module)
            color = get_module_color(module, "head")
            draw_block(current_y, label_text, color, fontsize=7 if module not in ("GAM", "SimAM") else 8)
            current_y -= 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = str(OUTPUT_DIR / "yolo11_comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def load_yaml(name):
    info = MODEL_INFO[name]
    path = MODELS_DIR / info["file"]
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating individual architecture diagrams...")
    for model_name in MODEL_INFO:
        data = load_yaml(model_name)
        scale = data.get("scales", {}).get("n", [0.5, 0.25, 1024])
        generate_diagram(model_name, data, MODEL_INFO[model_name], scale)

    print("Generating comparison diagram...")
    generate_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
