#%%
from PIL import Image
import numpy as np

#%%
image = Image.open('../assets/cat.jpg')
image = image.resize((224, 224))
# save image as binary for C code to read
im_arr = np.array(image)

# print total size of im_arr in bytes
im_arr.tofile('cat.bin')
# %%
res = np.fromfile('gray.bin', dtype=np.uint8)
res = res.reshape(224, 224)
Image.fromarray(res)
# %%
# im_arr to grayscale 
im_arr_g = np.dot(im_arr[...,:3], [0.21, 0.72, 0.07]).astype(np.uint8)
# print not equal elements
res[np.not_equal(im_arr_g, res)]
im_arr_g[np.not_equal(im_arr_g, res)]
# %%

res = np.fromfile('blur.bin', dtype=np.uint8)
res = res.reshape(224, 224)
Image.fromarray(res)
# %%
# blur image
import torch.nn.functional as F
import torch
torch_blur = F.conv2d(torch.tensor(im_arr_g).unsqueeze(0).float(), torch.ones(3, 3).view(1, 1, 3, 3) / 9, padding=1)
blur_arr = torch_blur.squeeze().numpy().astype(np.uint8)
print(np.equal(blur_arr, res).sum() / (224 * 224))
print(blur_arr, res)
# %%
