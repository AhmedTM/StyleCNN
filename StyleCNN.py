import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models

def get_features(image,model,layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
    features = {}
            
    x = image
    #print(x)
    for name,layer in model._modules.items():
        x = layer(x)
        #print(name)
        if name in layers:
            features[layers[name]] = x
    #print(features)
    return features
        
def gram_matrix(tensor):
    
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    
    return gram 

class StyleCNN():
    def __init__(self,content_image,style_image,Alpha=1,Beta=1e6):
        

        self.model = models.vgg19(pretrained=True).features
        for param in self.model.parameters():
            param.requires_grad_(False)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        content_image.to(device)
        style_image.to(device)            
       
            
        self.contentFeatures = get_features(content_image,self.model)
        self.styleFeatures = get_features(style_image,self.model)
        
        self.target = content_image.clone().requires_grad_(True).to(device)
        
        
        self.styleGrams = {layer: gram_matrix(self.styleFeatures[layer]) for layer in self.styleFeatures}
        
        
        self.optimizer = optim.LBFGS([self.target])
        
        self.styleWeights = {'conv1_1': 1.,
                             'conv2_1': 0.8,
                             'conv3_1': 0.5,
                             'conv4_1': 0.3,
                             'conv5_1': 0.1}

        self.contentWeight = Alpha  # alpha
        self.styleWeight = Beta  # beta
        
        
    def train(self,epochs = 30):
        def closure():
            self.optimizer.zero_grad()
            targetFeatures = get_features(self.target,self.model)
            #print(targetFeatures)
            #print(self.contentFeatures)
            contentLoss = torch.mean((targetFeatures["conv4_2"] - self.contentFeatures["conv4_2"])**2)
            
            styleLoss = 0
            
                
            
            for layer in self.styleWeights:
                    
                targetFeature = targetFeatures[layer]
                styleGram = self.styleGrams[layer]
                    
                targetGram = gram_matrix(targetFeature)
                    
                _, d, h, w = targetFeature.shape
                    
                layerStyleLoss = self.styleWeights[layer]*torch.mean((targetGram - styleGram)**2)
                    
                styleLoss += layerStyleLoss / (d * h * w)
                    
            totalLoss = (self.contentWeight * contentLoss) + (self.styleWeight*styleLoss)
            
            
            totalLoss.backward()
            return totalLoss
            
            
        self.optimizer.step(closure)
        return self.target