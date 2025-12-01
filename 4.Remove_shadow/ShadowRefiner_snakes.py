# https://github.com/movingforward100/Shadow_R

import os
import torch
from PIL import Image
from torchvision import transforms
import argparse
import psutil
from model import final_net
import torch.nn.functional as F

# ----------------------------------
# Utility: print memory usage
# ----------------------------------
def print_mem(tag=""):
    mem = psutil.virtual_memory()
    used = (mem.total - mem.available) / (1024**3)
    free = mem.available / (1024**3)
    print(f"[MEM] {tag}: used={used:.2f}GB  free={free:.2f}GB")


# ----------------------------------
# Wrapper around Shadow_R model
# ----------------------------------
class ShadowRWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = final_net()

    def forward(self, x):
        return self.model(x)


# ----------------------------------
# Resize keeping aspect ratio
# ----------------------------------
def safe_resize(im, max_side=1200):
    w, h = im.size
    if max(w, h) <= max_side:
        return im, (w, h)

    scale = max_side / float(max(w, h))
    new_w = int(w * scale)
    new_h = int(h * scale)

    print(f" - Resizing from {w}x{h} → {new_w}x{new_h} (safe mode)")

    return im.resize((new_w, new_h), Image.LANCZOS), (w, h)


# ----------------------------------
# Pad to next multiple of 16
# ----------------------------------
def pad_to_multiple_of_16(tensor):
    b, c, h, w = tensor.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded, pad_h, pad_w


# ----------------------------------
# Unpad after model output
# ----------------------------------
def unpad_output(tensor, pad_h, pad_w):
    if pad_h > 0:
        tensor = tensor[..., :-pad_h, :]
    if pad_w > 0:
        tensor = tensor[..., :, :-pad_w]
    return tensor


# ----------------------------------
# Main image processing
# ----------------------------------
def process_image(img_path, model, device, output_dir):
    print("\n================================")
    print(f"Processing: {os.path.basename(img_path)}")

    print_mem("Before loading image")
    try:
        im = Image.open(img_path).convert("RGB")
    except:
        print("❌ FAILED: Could not read image")
        return

    # 1) Safe resize
    im, orig_size = safe_resize(im)

    transform = transforms.ToTensor()
    img_tensor = transform(im).unsqueeze(0).float().to(device)

    # 2) Pad so H,W divisible by 16
    img_tensor, pad_h, pad_w = pad_to_multiple_of_16(img_tensor)

    print_mem("Before model forward")

    try:
        with torch.no_grad():
            out = model(img_tensor)
    except Exception as e:
        print(f"❌ FAILED on {os.path.basename(img_path)}: {e}")
        torch.cuda.empty_cache()
        print_mem("After failure")
        return

    # 3) Remove padding
    out = unpad_output(out, pad_h, pad_w)

    # 4) Convert to image
    out = torch.clamp(out, 0, 1).cpu().squeeze(0)
    out_img = transforms.ToPILImage()(out)

    # 5) Save
    save_path = os.path.join(output_dir, os.path.basename(img_path))
    out_img.save(save_path)
    print(f"✔ Saved: {save_path}")

    print_mem("After saving")


# ----------------------------------
# Main
# ----------------------------------
def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model = ShadowRWrapper()
    model.model.remove_model.load_state_dict(
        torch.load(os.path.join('weights', 'shadowremoval.pkl'), map_location='cpu')
    )
    model.model.enhancement_model.load_state_dict(
        torch.load(os.path.join('weights', 'refinement.pkl'), map_location='cpu')
    )
    model = model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)

    imgs = sorted(os.listdir(args.input_dir))

    for imgname in imgs:
        if not imgname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        process_image(os.path.join(args.input_dir, imgname),
                      model, device, args.output_dir)


# ----------------------------------
# CLI
# ----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    main(args)
