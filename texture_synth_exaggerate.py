from texture_synth_new import *

class TextureLoss_exaggerate(nn.Module):

    def __init__(self, target_feature1, target_feature2, alpha):
        super(TextureLoss_exaggerate, self).__init__()
        target1 = gram_matrix(target_feature1).detach()
        target2 = gram_matrix(target_feature2).detach()
        self.target = target1 + (target2 - target1)*alpha

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = LOSS_SCALING * F.mse_loss(G, self.target)

        return input
    

def get_texture_model_and_losses_exaggerate(cnn, texture_img1, texture_img2,
                                            alpha, texture_layers, device):
    cnn = copy.deepcopy(cnn)
    texture_losses = []

    cnn_normalization_mean = torch.tensor(IMAGENET_MEAN).to(device)
    cnn_normalization_std = torch.tensor([1.0, 1.0, 1.0]).to(device)

    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a convolution
    j = 0  # increment every time we see a pooling

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)  # in-place version clashes with TextureLoss ?
        elif isinstance(layer, nn.MaxPool2d):
            j += 1
            # replace every max-pooling by an average-pooling
            layer = nn.AvgPool2d(layer.kernel_size, stride=layer.stride,
                                 padding=layer.padding, ceil_mode=layer.ceil_mode)
            name = 'pool_{}'.format(j)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        
        if name in texture_layers:
            target_feature1 = model(texture_img1).detach()
            target_feature2 = model(texture_img2).detach()
            texture_loss = TextureLoss_exaggerate(target_feature1, target_feature2, alpha=alpha)
            model.add_module("texture_loss_{}".format(i + j), texture_loss)
            texture_losses.append(texture_loss)

    # remove the layers after the last one contributing to the texture loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], TextureLoss_exaggerate):
            break

    model = model[:(i + 1)]

    return model, texture_losses

def run_texture_synthesis_exaggerate(cnn, texture_image1, texture_image2, alpha, image_size, 
                                     num_steps, device, init_img=None, lambda_reg=None,
                                     verbose=True, return_PIL=False):
    
    if type(texture_image1) is torch.Tensor:
        pass
    elif type(texture_image1) is np.ndarray:
        texture_img1 = torch.from_numpy(texture_image1)
        texture_img1 = texture_img1.to(device, torch.float)
    else:
        texture_img1 = pre_processing(texture_image1, image_size, device)

    if type(texture_image2) is torch.Tensor:
        pass
    elif type(texture_image2) is np.ndarray:
        texture_img2 = torch.from_numpy(texture_image2)
        texture_img2 = texture_img2.to(device, torch.float)
    else:
        texture_img2 = pre_processing(texture_image2, image_size, device)

    rescale = transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    if init_img is None:
        synthesized_img = torch.randn(texture_img1.data.size(), device=device)
    else:
        if type(init_img) is np.ndarray:
            synthesized_img = torch.from_numpy(init_img)
            synthesized_img = synthesized_img.to(device, torch.float)
        elif type(init_img) is torch.Tensor:
            synthesized_img = init_img
        else:
            synthesized_img = pre_processing(init_img, image_size, device)
    
    synthesized_img = rescale(synthesized_img)
    init_img = synthesized_img.clone()
    
    if verbose:
        print('Building the texture model..\n')

    texture_layers = ['pool_4', 'pool_3', 'pool_2', 'pool_1', 'conv_1']

    model, texture_losses = get_texture_model_and_losses_exaggerate(cnn, texture_img1, 
                                                                    texture_img2, alpha, 
                                                                    texture_layers, device)
    
    optimizer = optim.LBFGS([synthesized_img.requires_grad_()])

    if verbose:
        print('Optimizing..\n')
        
    run = [0]  # weird local variable behaviour
    while run[0] <= num_steps:
        def closure():
            
            optimizer.zero_grad()
            model(synthesized_img)
            texture_score = 0

            for tl in texture_losses:
                texture_score += tl.loss
            loss = texture_score
            
            if lambda_reg is not None:
                l2norm = torch.sum((init_img-synthesized_img)**2)
                l2norm = float(l2norm.detach().numpy())
                loss += lambda_reg*l2norm
            
            loss.backward()

            run[0] += 1
            
            if run[0] % (500 if torch.cuda.is_available() else 100) == 0:
                if verbose:
                    print("run {}".format(run[0]))    
                    if lambda_reg is not None:
                        print('loss : {:.2e}, L2reg : {:.2e}\n'.format(loss.item(),
                                                                       lambda_reg*l2norm))
                    else:
                        print('loss : {:.2e}\n'.format(texture_score.item()))

            return texture_score

        optimizer.step(closure)
        
    if return_PIL:
        return post_processing(synthesized_img)
    
    return synthesized_img.detach().numpy()