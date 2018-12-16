import numpy as np
import torch
from PIL import Image

from scipy import ndimage
import scipy.misc
from torchvision import transforms
from StyleCNN import StyleCNN
import argparse
from matplotlib import cm

CONTENT_PATH = 'img/content.jpg' 
STYLE_PATH = 'img/style.jpg'
EPOCHS = 31
ALPHA = 1
BETA = 1e6


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_arguments():
    parser = argparse.ArgumentParser(description="Style convolutional neural networs with vgg19")
    parser.add_argument("--content-path", type=str, default=CONTENT_PATH,
                        help="Path to the content image.")
    parser.add_argument("--style-path", type=str, default=STYLE_PATH,
                        help="Path to the style image.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of iterations to update your image.")
    parser.add_argument("--alpha", type=int, default=ALPHA,
                        help="The content error weight.")
    parser.add_argument("--beta", type=int, default=BETA,
                        help="The style error weight.")

    return parser.parse_args()

def load_image(path,maxsize=400,shape=None):
    tensor = Image.open(path).convert('RGB')
    
    if max(tensor.size) > maxsize:
        size = maxsize
    else:
        size = max(tensor.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), 
                                                             (0.229, 0.224, 0.225))])
    tensor = in_transform(tensor)[:3,:,:].unsqueeze(0)
    return tensor

def convert_image(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    image = image * 255
    return image

def save_image(image,path):
    image = convert_image(image)
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image.save(path)
    
    
def main():
    args = get_arguments()
    contentImage = load_image(args.content_path)
    styleImage = load_image(args.style_path,shape=contentImage.shape[-2:])
    
    style_cnn = StyleCNN(contentImage,styleImage,Alpha=args.alpha,Beta=args.beta)
    
    for i in range(args.epochs):
        target = style_cnn.train()
    
        if i % 2 == 0:
            print("Iteration: %d" % (i))
            
            path = "outputs/%d.png" % (i)
            save_image(target, path)
    
    
if __name__ == '__main__':
    main()
