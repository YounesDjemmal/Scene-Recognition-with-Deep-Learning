import torch
import torch.nn as nn
from torchvision.models import alexnet


class MyAlexNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Note: Do not forget to freeze the layers of alexnet except the last one. Otherwise the training will take a long time. To freeze a layer, set the
        weights and biases of a layer to not require gradients.

        Note: Map elements of alexnet to self.cnn_layers and self.fc_layers.

        Note: Remove the last linear layer in Alexnet and add your own layer to 
        perform 15 class classification.

        Note: Download pretrained alexnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()
        list  = [0,3,6,8,10]
        list2 = [1,4]
        
        model = alexnet(pretrained = True)
        model.classifier[6] = nn.Linear(4096, 7)

        for i in list : 
            model.features[i].weight.requires_grad=False
            model.features[i].bias.requires_grad=False
        for i in list2 : 
            model.classifier[i].weight.requires_grad=False
            model.classifier[i].bias.requires_grad=False


        # for param in model.features.parameters() : 
        #     param.requires_grad= False

        # for param in model.classifier.parameters() : 
        #     param.requires_grad= False

        # for param in model.classifier[6].parameters() : 
        #     param.requires_grad= True

        # for param in model.avgpool.parameters():
        #     param.requires_grad= False
        
        self.conv_layers = nn.Sequential(model.features)
        #self.avgpool_layer = nn.Sequential(model.avgpool)
        self.fc_layers = nn.Sequential(model.classifier)

        self.loss_criterion = self.loss_criterion = nn.CrossEntropyLoss(reduction = 'sum')

        ############################################################################
        # Student code begin
        ############################################################################

        
        
        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Perform the forward pass with the net

        Note: do not perform soft-max or convert to probabilities in this function

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        y = None
        x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
        ############################################################################
        # Student code begin
        ############################################################################

        conv_features = self.conv_layers(x)
        #pooled_features = self.avgpool_layer(conv_features)
        flattened_conv_features = torch.reshape(conv_features, (conv_features.size()[0],9216))
        y = self.fc_layers(flattened_conv_features)

        ############################################################################
        # Student code end
        ############################################################################

        return y
