import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

print(model)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score/home/thuy0050/code/vlmcl/src/tevatron/hyperbolic/dev.py
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
