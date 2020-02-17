from texture_synth import *

def run_texture_synthesis_modified(cnn, texture_image, image_size, num_steps, device,
                          init_img=None, verbose=True, return_PIL=False):
    
    if type(texture_image) is torch.Tensor:
        pass
    elif type(texture_image) is np.ndarray:
        texture_img = torch.from_numpy(texture_image)
        texture_img = texture_img.to(device, torch.float)
    else:
        texture_img = pre_processing(texture_image, image_size, device)

    rescale = transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    if init_img is None:
        synthesized_img = torch.randn(texture_img.data.size(), device=device)
    else:
        if type(init_img) is np.ndarray:
            synthesized_img = torch.from_numpy(init_img)
            synthesized_img = synthesized_img.to(device, torch.float)
        elif type(init_img) is torch.Tensor:
            synthesized_img = init_img
        else:
            synthesized_img = pre_processing(init_img, image_size, device)
    
    synthesized_img = rescale(synthesized_img)
    
    if verbose:
        print('Building the texture model..\n')

    texture_layers = ['pool_4', 'pool_3', 'pool_2', 'pool_1', 'conv_1']

    model, texture_losses = get_texture_model_and_losses(cnn, texture_img, texture_layers, device)
    
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
            loss.backward()

            run[0] += 1
            
            if run[0] % (500 if torch.cuda.is_available() else 100) == 0:
                if verbose:
                    print("run {}".format(run[0]))
                    print('loss : {:.2e}\n'.format(texture_score.item()))

            return texture_score

        optimizer.step(closure)
    
    if return_PIL:
        return post_processing(synthesized_img)
    
    return synthesized_img.detach().numpy()

def circle_map(radius, width, size, center=None):
    Nx, Ny = size
    if center is None:
        x0 = (Nx-1)/2
        y0 = (Ny-1)/2
    else:
        x0, y0 = center

    yy,xx = np.meshgrid(np.arange(Ny), np.arange(Nx))
    
    circlemap = np.zeros([Nx,Ny])
    radmap = np.sqrt((xx-x0)**2 + (yy-y0)**2)
    circlemap[(radmap>radius-width/2) & (radmap<radius+width/2)] = 1
    
    return circlemap

def rectangle_map(side, width, size, center=None):
    Nx, Ny = size
    if center is None:
        x0 = (Nx-1)/2
        y0 = (Ny-1)/2
    else:
        x0, y0 = center
    
    yy,xx = np.meshgrid(np.arange(Ny), np.arange(Nx))
    recmap = np.zeros([Nx,Ny])
    sx = side[0]/2 - width[0]/2    
    sy = side[1]/2 - width[1]/2    
    recmap[(xx>x0+sx)|(xx<x0-sx)|(yy>y0+sy)|(yy<y0-sy)] = 1
    sx = side[0]/2 + width[0]/2    
    sy = side[1]/2 + width[1]/2    
    recmap[(xx>x0+sx)|(xx<x0-sx)|(yy>y0+sy)|(yy<y0-sy)] = 0
    
    return recmap

def gen_rand_im(powerDropoff=3, outputsize=128, genmapsize=64, priortype='flat'):
    
    import ehtim as eh
    import re
    import time
    import ehtim.imaging.dynamical_imaging as di
    from ehtim.imaging import starwarps as sw
    import copy
    import ehtim.scattering as so
    from PIL import Image

    zbl = 0.6

    # IMAGE PARAMETERS
    fov = 150.0 * eh.RADPERUAS  #field of view of the reconstructed image
    npixels = genmapsize#50                #number of pixels in the x and y dimension to reconstruct

    # IMAGE INITILIZATION PARAMETERS
    #priortype = 'flat'          #initilization image
    fwhm = 80 * eh.RADPERUAS
    #powerDropoff=3.0           #covariance smoothness term(larger values are smoother)
    covfrac = 0.4              #helps to constrain fraction of pixels > 0

    ra = 12.513728717168174
    dec = 12.39112323919932

    res = 15 * eh.RADPERUAS

    emptyprior = eh.image.make_empty(npixels, fov, ra, dec, rf=230000000000.0, source='M87')
    if priortype == 'gauss':
        gaussprior = emptyprior.add_gauss(zbl, (fwhm, fwhm, 0, 0, 0))
        meanimg = gaussprior.copy()
    elif priortype == 'flat':
        meanimg = emptyprior.copy()
        meanimg.imvec = np.ones(meanimg.imvec.shape)*zbl/(npixels**2)
    elif priortype == 'disk':
        tophat = emptyprior.add_tophat(zbl, fwhm/2.0)
        tophat = tophat.blur_circ(res)
        meanimg = tophat.copy()
    elif priortype == 'ring':
        ringim = emptyprior.add_ring_m1(zbl, 0, 21*eh.RADPERUAS, 0, 10*eh.RADPERUAS)
        ringim = ringim.blur_circ(res)
        meanimg = ringim.copy()
    elif priortype == 'asringim_left':
        asringim = emptyprior.add_ring_m1(zbl, 0.999, 21*eh.RADPERUAS, np.pi/3., 10*eh.RADPERUAS);
        asringim = asringim.blur_circ(res)
        meanimg = asringim.copy()
    elif priortype == 'asringim_right':
        asringim2 = emptyprior.add_ring_m1(zbl, 0.999, 21*eh.RADPERUAS, np.pi*2./3., 10*eh.RADPERUAS);
        asringim2 = asringim2.blur_circ(res)
        meanimg = asringim2.copy()

    imCov =  sw.gaussImgCovariance_2(meanimg, powerDropoff=powerDropoff, frac=covfrac)

    imgen = meanimg.copy()
    imgen.imvec = np.random.multivariate_normal(meanimg.imvec, imCov)
    #imgen.display()
    
    im = imgen.imvec.reshape(npixels,npixels)
    impil = Image.fromarray(im)
    impil = impil.resize((outputsize,outputsize), Image.BICUBIC)
    
    im_reshape = np.array(impil)
    
    return im_reshape