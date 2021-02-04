from PIL import Image
import os
import sys


def make_square(im, min_size=512, fill_color=(0, 0, 0)):
    x, y = im.size
    
    if x>y:
      fill_color = im.getpixel((int(x/2),y-3))
    else:
      fill_color = im.getpixel((x-3,int(y/2)))  
      
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

path = r"/data/image/training_set_SRGAN/"
abs_path = os.path.abspath('.')
files= os.listdir(abs_path+path)
for img_name in files:  
  test_image = Image.open(abs_path+path+img_name)
  new_image = make_square(test_image)
  new_image.save(abs_path+"/data/image/training_set_SRGAN/%s"%img_name)  