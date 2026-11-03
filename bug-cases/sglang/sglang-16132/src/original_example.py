     
import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

import json
import torch
from diffusers import DiffusionPipeline

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(_REPO_ROOT)
_ACTIVATION_DIR = os.environ.get("EKKA_ACTIVATION_DIR", os.path.join(_REPO_ROOT, "activations"))
from activation_collector.activation_collector import activation_collector
from scripts.generate_module_tree.get_model_tree import get_module_tree_and_flat_paths


prompt_lst = [
    "Create a PowerPoint slide with a 16:9 ratio featuring Tianjin's famous attractions and culture. The background should showcase the illuminated Tianjin Eye Ferris wheel at night, with the Hai River flowing beneath it. The Ferris wheel should be glowing with colorful lights against a deep blue evening sky.\n\nAt the top center of the slide, display the title \"天津：海河明珠\" in a bold, child-friendly sans-serif font (like Baloo or similar), size 36pt, in deep red color (#B31942), with a subtle text shadow to help it stand out against the background.\n\nOn the left side of the slide, include a semi-transparent white text box (opacity 70%, rounded corners) containing the following information:\n\n\"天津位于海河五大支流交汇处，靠近渤海\n• 天津之眼：世界唯一建在桥上的摩天轮\n• 特色美食：狗不理包子，皮薄馅多，汁水丰富\n• 文化特色：相声艺术，幽默诙谐的传统说唱艺术\"\n\nThe text should be in a clean, readable sans-serif font (like Source Han Sans), size 24pt, in dark blue color (#1A237E).\n\nOn the right side of the slide, include three small illustrated elements:\n1. At the top right: A cartoon drawing of the Tianjin Eye Ferris wheel with sparkling lights\n2. In the middle right: A steaming basket of Goubuli baozi (steamed buns) with one bun partially opened to show the filling\n3. At the bottom right: Two cartoon characters performing crosstalk comedy (xiangsheng), one speaking into a microphone while the other makes an exaggerated expression\n\nAdd a decorative border along the bottom of the slide using a traditional Chinese wave pattern in light blue (#81D4FA), symbolizing the Hai River.\n\nIn the bottom right corner, include a small cartoon character of a child enjoying a steamed bun with a speech bubble saying \"好吃！\" (Delicious!).\n\nThe overall style should maintain consistency with previous slides - educational, colorful, and engaging for children, with a clean layout.",
    # "Create a professional PowerPoint slide with a 16:9 ratio titled \"市场渗透率大幅提升\" in large bold Source Han Sans font (28pt) at the top left of the slide. Use deep blue color (#0B3D91) for the title.\n\nBelow the title, add a thin horizontal line in orange (#FF7F00) extending about one-third of the slide width.\n\nThe slide should have a clean white background (#FFFFFF) with subtle blue-green gradient elements (#0B3D91 to #00A86B) at the bottom right corner as a decorative accent.\n\nOn the left side of the slide, include the following bullet points in Source Han Sans font (18pt) in dark gray color (#333333):\n\n• \"新能源汽车市场渗透率达40.9%，较2023年提升9.3个百分点\"\n• \"新能源汽车已从\"新兴选择\"转变为\"主流选择\"\"\n• \"消费者接受度大幅提高，使用信心增强\"\n• \"中国在全球新能源汽车市场的领先地位进一步巩固\"\n• \"汽车产业电动化转型进程加快\"\n\nOn the right side of the slide, create a visual representation showing the penetration rate increase:\n- Include a large circular gauge chart showing the progression from 31.6% (2023) to 40.9% (2024)\n- The gauge should use a gradient from blue to green, with 0% at the bottom and 50% at the top\n- Label the 31.6% point in blue (#0B3D91) and the 40.9% point in green (#00A86B)\n- Include a \"+9.3%\" indicator in orange (#FF7F00) with an upward arrow between the two values\n\nAt the bottom of the gauge chart, add a small caption in Source Han Sans font (14pt): \"2023-2024年市场渗透率变化\"\n\nThe overall layout should be clean and corporate, with adequate white space, maintaining the modern tech-forward aesthetic established in the previous slides.\n\nOnly render the text enclosed in double quotes in this prompt on the image. All style specifications are for the model only.",
    # "Create a professional PowerPoint slide with a 16:9 ratio analyzing the main consumer groups and their demand characteristics for humanoid robots. The slide should have the following elements:\n\nBackground: Light gray (#F0F0F0) with subtle tech pattern overlay at 10% opacity\n\nTitle: Position at the top of the slide in deep blue (#003366), sans-serif font, 32pt, bold: \"主要消费群体与需求特征\"\n\nSubtitle: Below the title in dark gray (#333333), sans-serif font, 20pt: \"制造业企业、商业服务机构、科研院所等目标客户的采购行为分析\"\n\nMain content: A clean, modular layout with 5 sections, each representing a different consumer group. Each section should include:\n- An icon representing the consumer group\n- The name of the consumer group in deep blue (#003366), sans-serif font, 22pt\n- Two subsections labeled \"需求特点\" and \"购买行为\" in bright orange (#FF9900), sans-serif font, 18pt\n- Bullet points under each subsection in dark gray (#333333), serif font, 16pt\n\nThe 5 consumer groups with their specific content:\n\n1. Manufacturing Enterprises (left top section):\n   - Icon: Factory/industrial robot icon\n   - Group name: \"制造业企业\"\n   - 需求特点:\n     \"• 提高生产效率，降低人力成本\"\n     \"• 适应柔性化生产\"\n   - 购买行为:\n     \"• 倾向于选择能在非结构化环境中执行复杂任务的机器人\"\n   - Representative: \"代表企业：汽车制造商、3C电子产品厂商\"\n\n2. Commercial Service Institutions (right top section):\n   - Icon: Service/hospitality icon\n   - Group name: \"商业服务机构\"\n   - 需求特点:\n     \"• 提升服务体验，增加吸引力\"\n     \"• 提高运营效率\"\n   - 购买行为:\n     \"• 优先考虑具备良好人机交互能力的机器人\"\n   - Representative: \"代表机构：商场、酒店、展馆、银行等\"\n\n3. Scientific Research Institutions (left middle section):\n   - Icon: University/lab icon\n   - Group name: \"科研院所\"\n   - 需求特点:\n     \"• 支持前沿科研与实验验证\"\n     \"• 高可扩展性与可编程性\"\n   - 购买行为:\n     \"• 关注平台开放性与二次开发能力\"\n   - Representative: \"代表机构：高校实验室、国家重点实验室\"\n\n4. Healthcare Institutions (right middle section):\n   - Icon: Medical cross icon\n   - Group name: \"医疗机构\"\n   - 需求特点:\n     \"• 辅助护理与康复训练\"\n     \"• 提高服务质量与效率\"\n   - 购买行为:\n     \"• 优先考虑安全性与合规性\"\n   - Representative: \"代表机构：医院、康复中心\"\n\n5. Home Users (bottom center section):\n   - Icon: Home icon\n   - Group name: \"家庭用户\"\n   - 需求特点:\n     \"• 生活助理与陪伴\"\n     \"• 友好交互与易用性\"\n   - 购买行为:\n     \"• 更关注价格、外观与品牌口碑\"\n   - Representative: \"代表人群：老年人家庭、育儿家庭\"\n\nUse clean icons and a consistent modular grid layout. Only render the text enclosed in double quotes in this prompt on the image. All style specifications are for the model only.",
    # "Create a PowerPoint slide with a 16:9 ratio titled \"向日葵系列概览\" for a presentation about Van Gogh's Sunflowers series. The slide should maintain the artistic style established in the presentation, with a textured deep blue background similar to Van Gogh's night scenes and subtle canvas texture throughout.\n\nPosition the title \"向日葵系列概览\" at the top center of the slide in a bold, artistic white sans-serif font (size approximately 44pt) with a subtle golden glow effect.\n\nDivide the main content area into two sections:\n\nOn the left side (approximately 50% of the content area), display two stylized sunflower paintings side by side - one representing the Paris period style (more muted, with sunflowers lying on a table) and one representing the Arles period (brighter, sunflowers in a vase). Include a clear visual distinction between the two styles. Below each painting, add the following text in a clean white sans-serif font (approximately 24pt):\n\"巴黎时期 (1886-1887): 4幅\"\n\"阿尔勒时期 (1888-1889): 7幅\"\n\nOn the right side (approximately 50% of the content area), create a stylized world map in muted gold/sepia tones with highlighted locations marking the museums where the paintings are displayed. Include small sunflower icons at these locations. Below the map, add the text in the same white sans-serif font (approximately 24pt):\n\"作品现分布于全球多家知名美术馆\"\n\nAt the bottom center of the slide, add a subtle text in a slightly smaller font (approximately 20pt) in a light golden color:\n\"梵高一生共创作11-12幅向日葵主题作品\"\n\nThroughout the slide, incorporate subtle brushstroke textures and patterns reminiscent of Van Gogh's painting style, particularly around the borders and as dividers between sections.\n\nThe overall look should be artistic yet professional, suitable for an art history presentation, with a balanced composition that allows both text and visuals to be clearly visible.\n\nOnly render the text enclosed in double quotes in this prompt on the image. All style specifications are for the model only.",
]

MODEL_NAME = "Qwen/Qwen-Image"
OUT_DIR = Path("outputs/qwen-image-compare")

POSITIVE_MAGIC = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图.",
}

ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _prompt_magic(prompt: str) -> str:
    return POSITIVE_MAGIC["zh"] if _has_cjk(prompt) else POSITIVE_MAGIC["en"]


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        # fp32
        # return "cuda", torch.float32
        # bf16
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def run_diffusers(
    prompts: list[str],
    *,
    width: int,
    height: int,
    steps: int,
    true_cfg_scale: float,
    seed: int,
) -> None:
    _ensure_out_dir()
    device, torch_dtype = _device_and_dtype()
    pipe = DiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype, device_map="balanced")
    # pipe = pipe.to(device)
    
    # activation collector
    enable_activation_collector = False

    if enable_activation_collector:
        activation_path = os.path.join(_ACTIVATION_DIR, "bug-cases/sglang/sglang-16132")
        # save_prefix = "hf_activation_fp32_transformer"
        save_prefix = "hf_activation_vae"
        os.makedirs(activation_path, exist_ok=True)
        
        rank = int(os.environ.get("RANK", "0"))
        activation_collector.set_rank(rank)
        # compressed, flat_paths = get_module_tree_and_flat_paths(pipe.transformer)  # Hook into UNet
        compressed, flat_paths = get_module_tree_and_flat_paths(pipe.vae)  # Hook into UNet
        
        if activation_collector.get_rank() == 0:
            activation_save_path = f"{activation_path}/model_tree"
            os.makedirs(activation_save_path, exist_ok=True)
            with open(f"{activation_save_path}/model_tree_hf_vae.json", "w") as f:
                json.dump(compressed, f, indent=2)
        
        hook_fns = [[None, None] for name in flat_paths]
        hook_modes = ["both"] * len(flat_paths)
        
        activation_collector.register_hooks(
            # model=pipe.transformer,
            model=pipe.vae,
            target_modules=flat_paths,
            hook_fns=hook_fns,
            hook_modes=hook_modes,
            save_path=f"{activation_path}/{save_prefix}",
            framework="hf"
        )
        print(f"Activation collector enabled. Saving to {activation_path}")

    for name, value in pipe.__dict__.items():
        if hasattr(value, "named_modules"):
            print(name, "->", type(value))
            
    # exit()
    
    negative_prompt = " "
    for i, p in enumerate(prompts):
        out_path = OUT_DIR / f"diffusers_p{i+1}.png"
        g = torch.Generator(device=device)
        print(f"seed: {seed + i}")
        g.manual_seed(seed + i)

        image = pipe(
            prompt=p + _prompt_magic(p),
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=true_cfg_scale,
            generator=g,
        ).images[0]
        image.save(out_path)
        print(f"[diffusers] saved: {out_path}")
        
    if enable_activation_collector:
        activation_collector.save_all()

    # Best-effort cleanup to free VRAM before launching other GPU-heavy processes.
    try:
        if device == "cuda":
            pipe = pipe.to("cpu")
    except Exception:
        pass
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def run_sglang(
    prompts: list[str],
    *,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    extra_server_args: list[str] | None = None,
) -> None:
    _ensure_out_dir()
    extra_server_args = extra_server_args or []
    print(f"steps: {steps}")
    
    # activation collector
    enable_activation_collector = False
    if enable_activation_collector:
        activation_path = os.path.join(_ACTIVATION_DIR, "bug-cases/sglang/sglang-16132")
        activation_prefix = "sglang_imgtxt_latents_replaced_activation_transformer"
        # activation_prefix = "sglang_textlen_patched_activation_vae"
        os.makedirs(activation_path, exist_ok=True)
        os.environ["SGLANG_ACTIVATION_SAVE_PATH"] = activation_path
        os.environ["SGLANG_ACTIVATION_SAVE_PREFIX"] = activation_prefix


    for i, p in enumerate(prompts):
        out_file = f"sglang_p{i+1}.png"
        cmd = [
            "sglang",
            "generate",
            "--model-path",
            MODEL_NAME,
            "--prompt",
            p,
            "--negative-prompt",
            " ",
            "--save-output",
            "--output-path",
            str(OUT_DIR),
            "--output-file-name",
            out_file,
            "--width",
            str(width),
            "--height",
            str(height),
            "--num-inference-steps",
            str(steps),
            "--guidance-scale",
            str(guidance_scale),
            "--seed",
            str(seed + i),
            "--num-frames",
            "1",
        ]
        cmd = cmd[:4] + extra_server_args + cmd[4:]

        print(f"[sglang] running prompt {i+1}/{len(prompts)} -> {OUT_DIR / out_file}")
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, choices=["diffusers", "sglang", "both"], default="both")
    parser.add_argument("--order", type=str, choices=["diffusers-first", "sglang-first"], default="diffusers-first")
    parser.add_argument("--diffusers-subprocess", action="store_true")
    parser.add_argument("--aspect", type=str, default="16:9")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument(
        "--sglang-extra-server-args",
        type=str,
        default=os.environ.get("SGLANG_EXTRA_SERVER_ARGS", ""),
    )
    args = parser.parse_args()

    if args.aspect not in ASPECT_RATIOS:
        raise ValueError(f"Unknown aspect {args.aspect}. Choose from: {list(ASPECT_RATIOS.keys())}")
    width, height = ASPECT_RATIOS[args.aspect]

    extra = [x for x in args.sglang_extra_server_args.split(" ") if x]

    def _run_diffusers_maybe_subprocess() -> None:
        if not args.diffusers_subprocess:
            run_diffusers(
                prompt_lst,
                width=width,
                height=height,
                steps=args.steps,
                true_cfg_scale=args.cfg,
                seed=args.seed,
            )
            return

        cmd = [
            sys.executable,
            __file__,
            "--only",
            "diffusers",
            "--aspect",
            args.aspect,
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--cfg",
            str(args.cfg),
        ]
        print("[diffusers] running in subprocess to guarantee VRAM is released on exit")
        subprocess.run(cmd, check=True)

    def _run_sglang() -> None:
        run_sglang(
            prompt_lst,
            width=width,
            height=height,
            steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed,
            extra_server_args=extra,
        )

    if args.only == "diffusers":
        _run_diffusers_maybe_subprocess()
        return
    if args.only == "sglang":
        _run_sglang()
        return

    # both
    if args.order == "diffusers-first":
        _run_diffusers_maybe_subprocess()
        _run_sglang()
    else:
        _run_sglang()
        _run_diffusers_maybe_subprocess()


if __name__ == "__main__":
    main()