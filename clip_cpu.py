# for GPU inferencing:
#   sudo pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html ftfy regex scikit-image

# for CPU inferencing:
#   sudo pip3 install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html ftfy regex scikit-image

import numpy as np
from importlib import reload
import torch

MODELS = {
    "ViT-B-32":       "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B-32-cpu":       "https://battle.shawwn.com/sdb/models/ViT-B-32-cpu.pt",
}

import os
for name, url in MODELS.items():
  if not os.path.isfile(name+'.pt'):
    cmd = 'wget {url} -O {name}.pt -nc'.format(name=name, url=url)
    print(cmd)
    os.system(cmd)

#model_gpu = torch.jit.load("model.pt").cuda().eval()
model = torch.jit.load('ViT-B-32-cpu.pt', map_location=torch.device('cpu')).float().eval()
input_resolution = model.input_resolution.item()
context_length = model.context_length.item()
vocab_size = model.vocab_size.item()

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cpu()
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cpu()





import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# images in skimage to use and their textual descriptions
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse", 
    "coffee": "a cup of coffee on a saucer"
}


images = []
texts = []
# plt.figure(figsize=(16, 5))

for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue

    image = preprocess(Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB"))
    images.append(image)
    texts.append(descriptions[name])

    # plt.subplot(2, 4, len(images))
    # plt.imshow(image.permute(1, 2, 0))
    # plt.title(f"{filename}\n{descriptions[name]}")
    # plt.xticks([])
    # plt.yticks([])

# plt.tight_layout()


image_input = torch.tensor(np.stack(images)).cpu()
image_input -= image_mean[:, None, None]
image_input /= image_std[:, None, None]

from clip import encoder

tokenizer = encoder.get_encoder()
text_tokens = [tokenizer.encode("This is " + desc + "<|endoftext|>") for desc in texts]

text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)

for i, tokens in enumerate(text_tokens):
    text_input[i, :len(tokens)] = torch.tensor(tokens)

text_input = text_input.cpu()


with torch.no_grad():
  image_features = model.encode_image(image_input)
  text_features = model.encode_text(text_input)


image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T


def plot_similarity():
  count = len(descriptions)

  plt.figure(figsize=(20, 14))
  plt.imshow(similarity, vmax=0.3)
  # plt.colorbar()
  plt.yticks(range(count), texts, fontsize=18)
  plt.xticks([])
  for i, image in enumerate(images):
      plt.imshow(image.permute(1, 2, 0), extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
  for x in range(similarity.shape[1]):
      for y in range(similarity.shape[0]):
          plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

  for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

  plt.xlim([-0.5, count - 0.5])
  plt.ylim([count + 0.5, -2])

  plt.title("Cosine similarity between text and image features", size=20)

  plt.show()


  # from torchvision.datasets import CIFAR100
  # import os

  # cifar100 = CIFAR100(os.path.expanduser(".cache"), transform=preprocess, download=True)

  # text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
  # text_tokens = [tokenizer.encode(desc + "<|endoftext|>") for desc in text_descriptions]
  # text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)

  # for i, tokens in enumerate(text_tokens):
  #     text_input[i, :len(tokens)] = torch.tensor(tokens)

  # text_input = text_input.cuda()
  # text_input.shape



if __name__ == '__main__':
  plot_similarity()
