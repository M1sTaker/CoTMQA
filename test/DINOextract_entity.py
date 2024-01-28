import sys
sys.path.append("/root/autodl-tmp/fqy/Code/MQA")
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("/root/autodl-tmp/fqy/Code/MQA/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "/root/autodl-tmp/fqy/Code/MQA/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
IMAGE_PATH = "/root/autodl-tmp/fqy/Code/MQA/Obama.bmp"
TEXT_PROMPT = "person in the left of the image"
BOX_TRESHOLD = 0.20 #0.35
TEXT_TRESHOLD = 0.20 #0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    remove_combined = True,
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)