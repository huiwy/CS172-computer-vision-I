import numpy as np
import cv2 as cv

from sklearn.cluster import KMeans
import random
import utils


# referenced impl, fast
def generating_patches_rep(image, step, patch):
    sift = cv.SIFT_create()
    _, ret = sift.detectAndCompute(image, None)
    return ret

# naive impl, maybe wrong and slow
def generating_patches_rep_hand_coded(image, step, patch):
    # first use gaussian to smooth the image
    image = cv.GaussianBlur(image, (5,5), 0)
    # Then calculate the norm and direction of gradients
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
    sobelx = np.mean(sobelx, axis = 2)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
    sobely = np.mean(sobely, axis = 2)
    gradient_norm = np.sqrt(sobelx**2 + sobely**2)
    gradient_dir = np.round((np.arctan2(sobelx, sobely) + np.pi) * 180/ np.pi /45)\
                            .astype(np.int32)
    # some may be round up to 8, but it is 0 actually
    gradient_dir[gradient_dir == 8] = 0
    
    patches = []
    sub_size = int(patch/4)
    x, y, _ = image.shape
    i, j = 0, 0
    # calculate the subpatches
    subpatches = np.ones(np.floor(np.array([x/sub_size, y/sub_size, 8])).astype(np.int32))
    a, b, _ = subpatches.shape
    for i in range(a):
        for j in range(b):
            # extract the dirs and norms of gradients in this local region
            # flatten it
            subgradient_dir = gradient_dir[i*sub_size:(i+1)*sub_size,
                                          j*sub_size:(j+1)*sub_size]
            subgradient_dir = subgradient_dir.reshape(-1)
            subgradient_norm = gradient_norm[i*sub_size:(i+1)*sub_size,
                                          j*sub_size:(j+1)*sub_size]
            subgradient_norm = subgradient_norm.reshape(-1)
            # since some are added multiple times, use np.add.at for correct results
            np.add.at(subpatches[i,j], subgradient_dir, subgradient_norm)
            # normalize
            subpatches[i,j,:] /= np.sum(subpatches[i,j,:])
    xx = int(np.floor((x - patch)/step + 1))
    yy = int(np.floor((y - patch)/step + 1))
    for i in range(xx):
        for j in range(yy):
            patch_vec = np.zeros(128)
            for k in range(4):
                for l in range(4):
                    patch_vec[8*(4*k+l):8*(4*k+l+1)] = subpatches[2*i+k,2*j+l,:]
            patches.append(patch_vec)
    return patches

def generate_kmeans_model(data, dim, patch_size, step_size, 
          number_of_samples, verbose = False):
    
    print('number of images:', len(data))
    k = 0
    all_patches = []
    for i in range(0, len(data)):
        patches = generating_patches_rep(data[i], patch_size, 
                        step_size)
        try:
            k += patches.shape[0]
        except AttributeError as e:
            continue
            
        all_patches.append(patches)
        
        if i % 500 == 0:
            print(i, 'images finished.')
    
    
    print("number of all patches", k)
    # make the patches an ndarray
    all_patches_arr = np.zeros([k, 128])
    k = 0
    for i in all_patches:
        l = i.shape[0]
        all_patches_arr[k:k+l, :] = i
        k += l
    print(all_patches_arr.shape)
    
    # random sample the indices
    all_patches = np.random.choice(all_patches_arr.shape[0], number_of_samples, 
                                   replace = False)
    
    all_patches_arr = all_patches_arr[all_patches]
    print("number of samples used in Kmeans clustering:", all_patches.shape[0])
    print("Start training Kmeans model.")
    kmeans = KMeans(n_clusters=dim, random_state=0,
                    verbose = verbose * 2).fit(all_patches_arr)
    return kmeans

# feature function uninitialized
def feature_function_model_unfeeded(image, dim, step, batch, kmeans_model):
    a = generating_patches_rep(image, step, batch)
    feature_vec = np.zeros(dim)
    a = np.array(a).astype(np.float)
    kmeans_model.predict(a)
    np.add.at(feature_vec, kmeans_model.predict(a), 1)
    if np.sum(feature_vec) == 0:
        return feature_vec
    feature_vec /= np.sum(feature_vec)
    return feature_vec 