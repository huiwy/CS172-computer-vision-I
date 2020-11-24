from numpy.core.defchararray import count
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Given a depth map, retrieve its point cloud
def pointcloud(depthmap, fov = 45, v2h = 1, focus = 1, orthogonal = True):
  xx, yy = depthmap.shape

  points = np.zeros([xx*yy, 3])
  counter = 0

  midx = xx // 2
  midy = yy // 2
  
  vertical_stride = focus / np.tan(focus * np.cos(fov/180 * np.pi)) / xx
  horizental_stride = vertical_stride * v2h

  if orthogonal:
    for i in range(xx):
      for j in range(yy):
        points[counter, 0] = depthmap[i,j] 
        points[counter, 1] = vertical_stride*(midy - j)
        points[counter, 2] = horizental_stride*(midx - i)
        counter += 1
    return points


  for i in range(xx):
    for j in range(yy):
      film_loc = np.array([focus, horizental_stride*(i-midx), vertical_stride*(j-midy)])
      depth = depthmap[i,j]
      world_loc = film_loc * depth / np.linalg.norm(film_loc)
      points[counter,:] = world_loc
      counter += 1
  
  return points

def print3Dscatter(points):
  xs = points[:,0]
  ys = points[:,1]
  zs = points[:,2]
  
  fig = plt.figure(figsize=(12, 16))
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(xs, ys, zs, c=xs, s=7, cmap='tab20c')

  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  plt.show()

def betterscatter(img, depthmap):
  color = o3d.geometry.Image((img.transpose([1, 2, 0])* 255).astype(np.uint8))
  depth = o3d.geometry.Image(depthmap.astype(np.float32))
  rgbd_image = o3d.geometry.RGBDImage.\
                   create_from_color_and_depth(color, depth, 
                                              convert_rgb_to_intensity = False)

  pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
  # Flip it, otherwise the pointcloud will be upside down
  pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
  o3d.visualization.draw_geometries([pcd], width=800, height=600)