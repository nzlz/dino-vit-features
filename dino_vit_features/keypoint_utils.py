import mediapy as media
import numpy as np
import cv2
import open3d as o3d

import sys
sys.path.append('/home/norman/dino-vit-features')

import matplotlib.pyplot as plt
import torch
from correspondences import find_correspondences, draw_correspondences
from extractor import ViTExtractor
from PIL import Image
import glob

num_pairs = 8
load_size = 480
layer = 7 
facet = 'key'
bin=True 
thresh=0.2
model_type='dino_vits8'
stride = 8 
patch_size = 8 
device = "cuda"

extractor = ViTExtractor(model_type, stride, device=device)

def extract_descriptors(image_1_path, image_2_path, num_pairs = num_pairs, load_size = load_size):
    """
    Given a pair of image paths, extracs descriptors and returns the descriptor vectors.
    Inputs: image_1_path, image_2_path: paths to the images.
            num_pairs: number of pairs to extract.
            load_size: size to load the images.
    Outputs: patches_xy: indices of the extracted patches.
              desc1: descriptor map of the first image.
              descriptor_vectors: descriptor vectors of the first image.
              num_patches: number of patches in the x and y direction.    
    """
    with torch.no_grad():
        points1, points2, image1_pil, image2_pil, \
        patches_xy, desc1, desc2, num_patches = find_correspondences(image_1_path, image_2_path, num_pairs, load_size, layer,
                                                                       facet, bin, thresh, model_type, stride,
                                                                       return_patches_x_y = True)
        desc1 = desc1.reshape((num_patches[0],num_patches[1],6528))
        descriptor_vectors = desc1[patches_xy[0], patches_xy[1]]
        print("num patches", num_patches)
        return patches_xy, desc1, descriptor_vectors, num_patches



def extract_desc_maps(image_paths, load_size = load_size):
    """
    Given a list of image paths, extracts descriptor maps and returns them.
    Inputs: image_paths: list of image paths.
            load_size: size to resize the images.
    Outputs: descriptors_list: list of descriptor maps.
              org_images_list: list of the images.
    """
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    path = image_paths[0]
    if isinstance(path, str):
        pass
    else:
        paths = []
        for i in range(len(image_paths)):
            paths.append(f"image_{i}.png")
            media.write_image( f"image_{i}.png", image_paths[i])
        image_paths = paths

    descriptors_list = []
    org_images_list = []
    with torch.no_grad():
        for i in range(len(image_paths)):
            image_path = image_paths[i]

            image_batch, image_pil = extractor.preprocess(image_path, load_size)
            image_batch_transposed = np.transpose(image_batch[0], (1,2,0))

            print("image1_batch.size", image_batch.size())
            descriptors = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin)
            patched_shape = extractor.num_patches
            descriptors = descriptors.reshape((patched_shape[0],
                                patched_shape[1],
                                -1))

            descriptors_list.append(descriptors.cpu())
            image_batch_transposed = image_batch_transposed - image_batch_transposed.min()
            image_batch_transposed = image_batch_transposed/image_batch_transposed.max()
            image_batch_transposed = np.array(image_batch_transposed*255, dtype = np.uint8)
            org_images_list.append(media.resize_image(image_batch_transposed, (image_batch_transposed.shape[0]//patch_size,
                                                                   image_batch_transposed.shape[1]//patch_size)))
    return descriptors_list, org_images_list


def extract_descriptor_nn(descriptors, emb_im, patched_shape, return_heatmaps = False):
    """
    Given a list of descriptors and an embedded image, extracts the nearest neighbor descriptor and returns the keypoints.
    Inputs: descriptors: list of descriptor vectors.
            emb_im: embedded image.
            patched_shape: shape of the patches.
            return_heatmaps: if True, returns the heatmaps of the similarities.
    Outputs: cs_ys_list: y coordinates of the keypoints.
             cs_xs_list: x coordinates of the keypoints.
             cs_list: list of heatmaps if return_heatmaps
    """
    cs_ys_list = []
    cs_xs_list = []
    cs_list = []
    cs = torch.nn.CosineSimilarity(dim=-1)
    print("emb_im shape", emb_im.shape)
    for i in range(len(descriptors)):
        cs_i = cs(descriptors[i].cuda(), emb_im.cuda())
        print("cs_i.shape", cs_i.shape)
        cs_i = cs_i.reshape((-1))
        cs_i_y = cs_i.argmax().cpu()//patched_shape[1]
        cs_i_x = cs_i.argmax().cpu()%patched_shape[1]

        cs_ys_list.append(int(cs_i_y)*stride)
        cs_xs_list.append(int(cs_i_x)*stride)

        cs_list.append(np.array(cs_i.cpu()).reshape(patched_shape))

        cs_i = cs_i.reshape(patched_shape)
    if return_heatmaps:
        return cs_ys_list, cs_xs_list, cs_list
    return cs_ys_list, cs_xs_list


def draw_keypoints(image, key_y, key_x, colors):
    """
    Given an image and keypoints, draws the keypoints on the image.
    Inputs: image: image to draw the keypoints on.
            key_y: y coordinates of the keypoints.
            key_x: x coordinates of the keypoints.
            colors: colors of the keypoints.
    Outputs: canvas: image with the keypoints drawn on it
    """
    assert len(key_y) == len(key_x)
    canvas = np.zeros((image.shape[0],image.shape[1],3))
    if len(image.shape) < 3 or image.shape[-1] == 1:
        canvas[:,:,0] = image
    else:
        canvas = np.array(image)
    for i in range(len(key_y)):
        color = colors[i]
        canvas = canvas.astype(np.uint8)
        canvas[key_y[i]-5:key_y[i]+5,key_x[i]-5:key_x[i]+5,:] = np.array(color)
    return canvas


def draw_numbers(image, key_y, key_x, colors, font_scale = 0.5, font_thickness = 1):
    """
    Given an image and keypoints, draws the numbers of the keypoints on the image.
    Inputs: image: image to draw the keypoints on.
            key_y: y coordinates of the keypoints.
            key_x: x coordinates of the keypoints.
            colors: colors of the keypoints.
            font_scale : size of font
            font_thickness : thickness of font
    Outputs: image: image with the numbers of the keypoints drawn
    """
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 0, 0)  # Green color in BGR

    # Draw numbers for each keypoint
    for i, (x, y) in enumerate(zip(key_x, key_y)):
        x, y = int(x)*8, int(y)*8
                
        print( (colors[i][0], colors[i][1], colors[i][2]))
        cv2.putText(image, str(i+1), (x, y), font, font_scale, font_color, font_thickness)
    return image

def project_in_3d(points, depth, intrinsics_mat, scaling_factor = 1000):
  """
  Given a list of points, a depth map and intrinsics matrix, projects the points in 3D.
  Inputs: points: list of 2D points.
          depth: (H,W) depth map.
          intrinsics_mat: intrinsics matrix of the camera.
          scaling_factor: scaling factor of the depth map. (default: millimeters)
  Outputs: x3d: x coordinates of the 3D points.
           y3d: y coordinates of the 3D points.
           z3d: z coordinates of the 3D points.
  """
  x3d, y3d, z3d = [], [], []

  for p in points:
    u, v = p
    v = int(v)
    u = int(u)
    fx = intrinsics_mat[0,0]
    fy = intrinsics_mat[1,1]
    cx = intrinsics_mat[0,2]
    cy = intrinsics_mat[1,2]
    d = depth[u,v]
    z = d / scaling_factor
    x = (v - cx) * z / fx
    y = (u - cy) * z / fy
    x3d.append(x)
    y3d.append(y)
    z3d.append(z)
  return x3d, y3d, z3d


def clean_zero_kps(k):
    """
    Given a list of keypoints, removes the keypoints that are too far or have a zero x coordinate
    among all the images.
    Inputs: k: list of 3D keypoints.
    Outputs: k: cleaned list of keypoints.
             non_zero_ids: indices of the keypoints that are kept
    """
    k = np.array(k)
    print("shape k", k.shape)
    non_zero_ids = np.arange(0, k.shape[2])
    for k_i in k:
        k_i_x = k_i[0]
        k_i_z = k_i[2]
        zero_ids = np.argwhere(np.array(k_i_x) ==  0)
        for z_i in zero_ids:
            non_zero_ids = np.delete(non_zero_ids, np.argwhere(non_zero_ids == z_i[0]))
        far_ids = np.argwhere(np.array(k_i_z) >  1.5)
        for f_i in far_ids:
            non_zero_ids = np.delete(non_zero_ids, np.argwhere(non_zero_ids == f_i[0]))
    k = k[:,:,non_zero_ids]
    return k, non_zero_ids


def extract_data_for_training(paths_rgb, paths_depth, intrinsics_mat, num_pairs = 15, clean = True, show = True, 
                              matches_between_same_image = False, manual_first_frame = None):
    """
    Given a list of paths to the RGB and depth images, extracts the keypoints and descriptor vectors.
    Inputs: paths_rgb: list of paths to the RGB images.
            paths_depth: list of paths to the depth images.
            intrinsics_mat: intrinsics matrix of the camera.
            num_pairs: number of keypoints to extract.
            clean: if True, removes the keypoints that are too far or have a zero x coordinate.
            show: if True, shows the images with the keypoints.
            matches_between_same_image: if True, extracts the matches between the same image (UNSTABLE, only use if a single image is available).
            manual_first_frame: if not None, uses this frame as the first frame.
    Outputs: kps: list of 3D keypoints.
             descriptor_vectors: descriptor vectors of the first image.
             num_patches: number of patches in the x and y direction.
             non_zero_ids: indices of the keypoints that are kept.
             colors: colors of the keypoints to draw.
    """
    assert len(paths_rgb) > 1
    assert len(paths_rgb) == len(paths_depth)

    video1 = np.load(glob.glob(paths_rgb[0]))
    video2 = np.load(glob.glob(paths_rgb[1]))
    if matches_between_same_image:
      video1 = np.load(paths_rgb[1])
    if not manual_first_frame is None:
        video1 = manual_first_frame
    media.write_image("a.png", video1[0])
    media.write_image("b.png", video2[0])
    patches_xy, desc_map, descriptor_vectors, num_patches = extract_descriptors("a.png",
                                                                            "b.png", 
                                                                            load_size = load_size, 
                                                                            num_pairs = num_pairs)
    kps = []
    colors = [np.random.randint(0,255,size=3) for i in range(num_pairs)]
    for path_i in range(len(paths_rgb)):
        video = np.load(paths_rgb[path_i])
        descriptors_list, org_images_list = extract_desc_maps(video[0])
        key_y, key_x = extract_descriptor_nn(descriptor_vectors, emb_im=descriptors_list[0], patched_shape=num_patches, return_heatmaps = False)
        points = [(y,x) for y, x in zip(np.array(key_y)*stride, np.array(key_x)*stride)]
        depth = np.load(paths_depth[path_i])
        x3d, y3d, z3d = project_in_3d(points, depth, intrinsics_mat)
        if show:
          i=0
          media.show_image(np.array(org_images_list[i], dtype = np.uint8))
          image_w_keypoints = draw_keypoints(np.array(org_images_list[i], dtype = np.uint8), key_y, key_x,  colors)
          media.show_image(media.resize_image(image_w_keypoints, (image_w_keypoints.shape[0]*4,image_w_keypoints.shape[1]*4)))
        kps.append([x3d, y3d, z3d])
    if clean:
      kps, non_zero_ids = clean_zero_kps(kps)
    else:
      non_zero_ids = np.arange(0, num_pairs)
    return kps, descriptor_vectors, num_patches, non_zero_ids, colors


def keypoints_using_descriptors(image, depth, descriptors, patches, non_zero_ids, stride, intrinsics_mat):
    """
    Given an image, depth map, and descriptrs, extract keypoints and project them in 3D.
    Inputs: image: image to extract the keypoints.
            depth: depth map of the image.
            descriptors: descriptor vectors.
            patches: shape of the patches.
            non_zero_ids: indices of the keypoints that are kept.
            stride: stride of the patches.
            intrinsics_mat: intrinsics matrix of the camera.
    Outputs: x3d: x coordinates of the 3D points.
             y3d: y coordinates of the 3D points.
             z3d: z coordinates of the 3D points.
    """
    descriptors_list, org_images_list = extract_desc_maps([image])
    key_y, key_x = extract_descriptor_nn(descriptors, emb_im=descriptors_list[0], patched_shape=patches, return_heatmaps = False)
    key_y = np.array(key_y)[non_zero_ids]
    key_x = np.array(key_x)[non_zero_ids]
    points = [(y,x) for y, x in zip(np.array(key_y)*stride, np.array(key_x)*stride)]
    x3d, y3d, z3d = project_in_3d(points, depth, intrinsics_mat)
    return x3d, y3d, z3d

def clean_reshape(k):
    """
    Given a list of keypoints, reshapes them.
    Inputs: k: list of 3D keypoints.
    Outputs: clean_k_reshape: reshaped list of keypoints.  
    """
    clean_k_reshape = []
    for i in range(len(k)):
        new = []
        for j in range(len(k[0,0])):
            new.append([k[i,0,j], k[i,1,j], k[i,2,j]])
        clean_k_reshape.append(new)
    return clean_k_reshape


def find_transformation(X, Y):
    """
    Find transformation given two sets of correspondences between 3D points.

    Inputs: X, Y: lists of 3D points
    Outputs: R - 3x3 rotation matrix, t - 3-dim translation array.
    """
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    Xc = X - cX
    Yc = Y - cY
    C = np.dot(Xc.T, Yc)
    U, S, Vt = np.linalg.svd(C)
    R = np.dot(Vt.T, U.T)
    t = cY - np.dot(R, cX)
    return R, t
    
    
