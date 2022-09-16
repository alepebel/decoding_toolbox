#    Created by Alexis Pérez Bellido, 2022
import numpy as np

# Creating folds containing test and train indexes
def CreateFolds(X,Y,nfold):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=nfold,shuffle=False)
    Folds = [None] * nfold
    i = 0
    numN = X.shape[0]
    for train_index, test_index  in  skf.split(X = np.zeros(numN), y = X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Folds[i] = { 'train_index': train_index, 'test_index': test_index}
        i += 1

    return Folds


#   [design] = stim_features(cfg, phi)
#    Returns hypothetical channel responses given a presented orientation, cf. Brouwer & Heeger.
#
#    phi         Array of length N, where N is the number of trials, that specifies the presented
#                orientation on each trial. Orientation is specified in degrees and is expected to
#                have a range of 0-180.
#    cfg         Configuration dictionary that can possess the following fields:
#                ['NumC']                               The number of hypothetical channels C to use. The
#                                                    channels are equally distributed across the circle,
#                                                    starting at 0.
#                ['Tuning']                        The tuning curve according to which the responses
#                                                    are calculated.
#                ['Tuning'] = 'vonMises'           Von Mises curve. Kappa: concentration parameter.
#                ['Tuning'] = 'halfRectCos'        Half-wave rectified cosine. Kappa: power.
#                ['Tuning'] = [function_handle]    User-specified function that can take a matrix as input,
#                                                    containing angles in radians with a range of 0-pi.
#                ['kappa']                              Parameter(s) to be passed on to the tuning curve.
#                ['offset']                            The orientation of the first channel. (default = 0)
#           
#    design      The design matrix, of size C x N, containing the hypothetical channel responses.
#    sortedesign A sorted version of the design matrix, sorted by the presented orientation to improve model visualization.

#
# Creating folds containing test and train indexes
def CCreateFolds(X,Y,nfold):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=nfold,shuffle=False)
    CrossValIdx = [None] * nfold
    i = 0
    numN = X.shape[0]
    for train_index, test_index  in  skf.split(X = np.zeros(numN), y = X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        CrossValIdx[i] = { 'train_index': train_index, 'test_index': test_index}
        i += 1

    folds = dict()
    folds['X_train'] = X[ CrossValIdx[0]['train_index']][:,np.newaxis]
    folds['Y_train'] = np.squeeze(Y[:,sel_t, CrossValIdx[0]['train_index']])
    folds['phi_train'] = phi[ CrossValIdx[0]['train_index']][:,np.newaxis]

    folds['X_test'] = X[ CrossValIdx[0]['test_index']][:,np.newaxis]
    folds['Y_test'] = np.squeeze(Y[:,sel_t, CrossValIdx[0]['test_index']])
    folds['phi_test'] = phi[ CrossValIdx[0]['test_index']][:,np.newaxis]
    return folds


#   [design] = stim_features(cfg, phi)
#    Returns hypothetical channel responses given a presented orientation, cf. Brouwer & Heeger.
#
#    phi         Array of length N, where N is the number of trials, that specifies the presented
#                orientation on each trial. Orientation is specified in degrees and is expected to
#                have a range of 0-180.
#    cfg         Configuration dictionary that can possess the following fields:
#                ['NumC']                               The number of hypothetical channels C to use. The
#                                                    channels are equally distributed across the circle,
#                                                    starting at 0.
#                ['Tuning']                        The tuning curve according to which the responses
#                                                    are calculated.
#                ['Tuning'] = 'vonMises'           Von Mises curve. Kappa: concentration parameter.
#                ['Tuning'] = 'halfRectCos'        Half-wave rectified cosine. Kappa: power.
#                ['Tuning'] = [function_handle]    User-specified function that can take a matrix as input,
#                                                    containing angles in radians with a range of 0-pi.
#                ['kappa']                              Parameter(s) to be passed on to the tuning curve.
#                ['offset']                            The orientation of the first channel. (default = 0)
#           
#    design      The design matrix, of size C x N, containing the hypothetical channel responses.
#    sortedesign A sorted version of the design matrix, sorted by the presented orientation to improve model visualization.

#    Created by Pim Mostert, 2016 in Matlab. Exported to Python by Alexis Pérez Bellido, 2022

def stim_features(phi, cfg):
    kappa = cfg['kappa']
    NumC = cfg['NumC']
    if 'offset' not in cfg:
        offset = 0
    else:
         offset = cfg['offset']

    sort_idx = phi.argsort(axis = 0)

    NumN = phi.size
    phi = phi - offset
    design = np.arange(0,NumC)[np.newaxis,:].T * np.ones([1, NumN]) * 180/NumC
    design = design - np.ones([NumC,1])*phi.T
    design = design * (np.pi/180) # transforming to radians

    if cfg['Tuning'] == 'halfRectCos':
        fn = lambda x: np.abs(np.cos(x))**kappa
    if cfg['Tuning'] == 'vonmises':
        mu = 0
        fn = lambda x: np.exp(kappa*np.cos(2*x-mu))/(2*np.pi*np.i0(kappa)) 
    
    design = fn(design)
    
    sortedesign = design[:, sort_idx] #np.take_along_axis(design, sort_idx, axis = 1).copy()
    
    return [design , sortedesign]


#   [decoder] = train_encoder(X, Y, cfg)
#    Trains a linear decoder "beamformer style" to optimally recover the latent components as 
#    prescribed in X. Several decoders may be trained indepedently, corresponding to several
#    latent components.
#
#    X           Vector or matrix of size C x N, where C is the number of components and N is
#                the number of trials, that contains the expected/prescribed component activity
#                in the training data.
#    Y           Matrix of size F x N, where F is the number of features, that contains the
#                training data.
#    cfg         Configuration struct that can possess the following fields:
#                ['gamma'] = [scalar]                Shrinkage regularization parameter, with range [0 1]. 
#                                                 No default given.
#                ['returnPattern'] = 'yes' or 'no'   Whether the spatial patterns of the components should
#                                                 be returned. Default = 'no';
#                ['demean'] = 'yes' or 'no'          Whether to demean the data first (per feature, over
#                                                 trials). Default = 'yes';.
#
#    decoder     The (set of) decoder(s), that may be passed on to an appropriate decoding function,
#                e.g. decode_beamformer. It may contain a field .pattern of size C x F
#
#    See also DECODE_BEAMFORMER.

#    Created by Pim Mostert, 2016 in Matlab. Exported to Python by Alexis Pérez Bellido, 2022

def train_encoder(X, Y, cfg): 
    Y = Y.T
    X = X.T
    numC = X.shape[1]
    numF = Y.shape[1]
    decoder = dict()
    gamma = cfg['gamma']

    # demean activity in each trial
    if 'demean' not in cfg:
        cfg['demean'] = True

    if 'returnPattern' not in cfg:
        cfg['returnPattern'] = False

    if  cfg['demean']:
        Ym = Y.mean(axis=0, keepdims=True)
        Y = Y - np.repeat(Ym, axis = 0, repeats = Y.shape[0])
        decoder['dmY'] = Ym.T # save demeaned Y for posterior inspection
    
    if cfg['returnPattern']:
        decoder['pattern'] = np.zeros([numF, numC]) 

    decoder['W'] = np.zeros([numC, numF]) # weights empty matrix


    for ic in range(numC):
        # Estimate leadfield for current channel
        l = ((X[:,ic].T @ X[:,ic])**-1) * (X[:,[ic]].T @ Y) 

        if cfg['returnPattern']:
            decoder['pattern'][:, ic] = l
        # Estimate noise (what is not explained by the regressors coefficients)
        N = Y - X[:,[ic]] * l
        # Estimate noise covariance
        S = np.cov(N,rowvar=False).copy() # rowvar is necessary to get the correct covariance matrix shape
        #  Regularize
        S = (1-gamma)*S + gamma*np.eye(numF) * np.trace(S)/numF #% [w,d] = eig(S);  eigenvalues -> pdiag(d)
        decoder['W'][ic, :] = np.dot(l,np.linalg.inv(S))

    return decoder


def test_encoder( decoder, Y, cfg):
    NumN = Y.shape[1]
    NumC = decoder['W'].shape[0]
    NumF = Y.shape[0]

    # Demean or not demean
    if cfg['demean'] == 'traindata': # demean test data with mean of training data
        Y = Y - np.repeat(decoder['dmY'], axis = 1, repeats = NumN)
    if cfg['demean'] == 'testdata':
        Ym = Y.mean(axis = 0, keepdims=True)
        Y = Y - np.repeat(Ym, axis = 1, repeats = NumN)

    # Decode
    # Inverting the calculated weights for each channel to decode the stimuli

    decoder['iW'] = 0*decoder['W'].copy() # empty array to store the inverted weights

    for ic in range(NumC):
        W = decoder['W'][ic,:]
        decoder['iW'][ic,:] = W / (W @ W.T) # inverting the weights

    Xhat = decoder['iW'] @ Y # Predicting stim based on activity multiplying by the inverted decoder -> Y = WX + N, doing this is like calculating X = Y/N
    return Xhat