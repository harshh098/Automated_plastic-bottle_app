from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from PIL import Image, ImageDraw

# ——— Load SAM model ——————————————
SAM_CHECKPOINT = r"C:/Users/harsh/OneDrive/Desktop/project/New folder/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam_predictor = SamPredictor(sam)

# ——— Correct detect_with_sam function —————————
def detect_with_sam(image: Image.Image, boxes: list) -> Image.Image:
    """
    Takes an image and YOLO-detected boxes,
    applies SAM to get masks,
    overlays red masks + green bounding boxes.
    """
    img_rgb = image.convert("RGB")
    img_np  = np.array(img_rgb)

    # Load image to SAM
    sam_predictor.set_image(img_np)

    # Create overlay
    base    = img_rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for box in boxes:
        mask, _, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(box)[None, :],  # (1, 4)
            multimask_output=False
        )
        mask_np = mask[0]

        # Red mask creation
        red_layer = Image.new("RGBA", base.size, (255, 0, 0, 255))
        mask_img  = Image.fromarray((mask_np * 255).astype(np.uint8))

        # Apply red mask
        overlay.paste(red_layer, (0, 0), mask_img)

        # Draw green bounding box
        draw.rectangle(box, outline=(0, 255, 0, 255), width=2)

    # Merge and return final output
    return Image.alpha_composite(base, overlay).convert("RGB")
