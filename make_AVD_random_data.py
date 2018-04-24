import os
import cv2
import numpy as np
import glob
import selectivesearch
from scipy import interpolate
from scipy import misc
import scipy.io as sio
import sys
import png

def write_numpy_to_ply(vertices, outfile):

    outfile = open(outfile,'w')
    outfile.write('ply\n')
    outfile.write('format ascii 1.0\n')
    outfile.write('element vertex {}\n'.format(vertices.shape[1]))
    outfile.write('property float x\n')
    outfile.write('property float y\n')
    outfile.write('property float z\n')
    outfile.write('end_header\n')
    
    for v_ind in range(vertices.shape[1]):
        v = vertices[:,v_ind]
        outfile.write('{} {} {}\n'.format(v[0], v[1],v[2]))

    outfile.close()


def project_points_to_image(world_coords,img_struct,intrinsic,distortion,img_shape,d_img=None):
    world_coords_homog = np.concatenate((world_coords,np.ones((1,world_coords.shape[1]))))
    R = img_struct[2]
    t = img_struct[1]
    if len(t) == 0:
        print('No t')
        return [] 

    P = np.concatenate((R,t),axis=1) 
    points = np.matmul(P ,world_coords_homog)
    zs = points[2,:]
    zs[zs<0] = 0
    cur_world_points = world_coords[:,zs.nonzero()[0]]
    if cur_world_points.size == 0:
        print('No orinetation')
        return []

    #project points back onto second image to get new bounding box
    XC = np.matmul(R,cur_world_points) + t
    a = XC[0,:] / XC[2,:]
    b = XC[1,:] / XC[2,:]

    r = np.sqrt(a*a + b*b)
    theta = np.arctan(r)
    thetad = theta * (1 + distortion[0]*(np.power(theta,2)) +
                          distortion[1]*(np.power(theta,4)) +
                          distortion[2]*(np.power(theta,6)) +
                          distortion[3]*(np.power(theta,8)))

    xx = (thetad/r) * a
    yy = (thetad/r) * b

    up = intrinsic[0,0] * (xx + 0*yy) + intrinsic[0,2]
    vp = intrinsic[1,1] *yy + intrinsic[1,2]

    ui = [x for x in up if x>=0 and x<=img_shape[1]]
    vi = [x for x in vp if x>=0 and x<=img_shape[0]]

    if len(ui) == 0 or len(vi) == 0:
        print('No inside image')
        return [] 

    xvals = np.round(np.asarray(ui))
    yvals = np.round(np.asarray(vi))

    depth_vals = d_img[yvals,xvals]

    new_box = [int(xvals.min()), int(yvals.min()), int(xvals.max()), int(yvals.max())]
    return new_box 


def prune_outliers(array, dim, nstd):
    values = array[dim,:]
    return array[:,abs(values - values.mean()) < nstd*values.std()]



max_box = 50000
min_box = 1500
#set paths to data
avd_root2 = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
#avd_root2 = '/playpen/ammirato/Data/RohitMetaData/'
avd_root = '/playpen/ammirato/Data/RohitData/'
#avd_root = avd_root2 
#pick a scene
scene = 'Home_003_1'
#pick an image and semi-random bbox in that image
img_paths = glob.glob(os.path.join(avd_root,scene,'jpg_rgb','*.jpg'))
img_names = os.listdir(os.path.join(avd_root,scene,'jpg_rgb'))
img_path = img_paths[np.random.choice(len(img_paths))]
img_name = img_path[img_path.rfind('/')+1:]
#img_name = '000320004350101.jpg'
#img_name = '0008570101.jpg'
img_path = os.path.join(avd_root,scene,'jpg_rgb',img_name)
d_img_path = img_path.replace('jpg_rgb','high_res_depth').replace('01.jpg','03.png')
rgb_img = cv2.imread(img_path)
org_img_shape = rgb_img.shape
rgb_img = cv2.resize(rgb_img,(0,0),fx=.5,fy=.5)
#d_img = cv2.imread(d_img_path)
reader = png.Reader(d_img_path)
pngdata = reader.read()
depth_vals = np.array(map(np.uint16, pngdata[2]))
#choose a region proposal
img_lbl, regions = selectivesearch.selective_search(rgb_img, scale=500, sigma=0.9, min_size=10)
box = []
counter = 0
while len(box) == 0:
    counter +=1
    ind = np.random.choice(len(regions))
    bsize =regions[ind]['size'] 
    bsize*= 4
    if bsize > min_box and bsize < max_box:
        x,y,w,h= regions[ind]['rect'] 
        x *=2
        y *=2
        w *=2
        h *=2
        if w/h > 1.5 or h/w > 1.5:
            continue
        box = [x,y,x+w,y+h]
print('Prop search count {}'.format(counter))

#xs = np.random.randint(100,high=rgb_img.shape[1]-100, size=2)
#ys = np.random.randint(100,high=rgb_img.shape[0]-100, size=2)
#box = [1160,460, 1265, 740]
#box = [xs.min(),ys.min(),xs.max(),ys.max()]
#project points in bbox to world coordinates
#camera follows:  http://docs.opencv.org/master/db/d58/group__calib3d__fisheye.html#gsc.tab=0 

#camera_params_fid = open(os.path.join(avd_root2,scene,'reconstruction_results/colmap_results/0/','cameras.txt'))
camera_params_fid = open(os.path.join(avd_root2,scene,'cameras.txt'))
lines = [x for x in camera_params_fid]
lines = lines[-1].split()
intrinsic = np.zeros((3,3))
intrinsic[0,0] = float(lines[4])
intrinsic[0,2] = float(lines[6])
intrinsic[1,1] = float(lines[5])
intrinsic[1,2] = float(lines[7])
intrinsic[2,2] = float(1)

distortion = np.zeros(4)
distortion[0] = float(lines[8])
distortion[1] = float(lines[9]) 
distortion[2] = float(lines[10])
distortion[3] = float(lines[11])
camera_params_fid.close()

pixel_pos = np.mgrid[0:org_img_shape[0],0:org_img_shape[1]]
pixel_pos = pixel_pos.reshape((2,-1))

aa = np.linspace(-1,1,2000)
bb = np.linspace(-1,1,2000)
rr = np.sqrt(np.square(aa) + np.square(bb))
theta = np.arctan(rr)
thetad = theta * (1 + distortion[0]*(np.power(theta,2)) +
                      distortion[1]*(np.power(theta,4)) +
                      distortion[2]*(np.power(theta,6)) +
                      distortion[3]*(np.power(theta,8)))
xx = (thetad/rr) * aa
yy = thetad/rr *bb

index_map_mult = 1000
index_map_add = 1000
a_map = np.zeros(aa.shape)
b_map = np.zeros(bb.shape)
a_map[np.floor(xx*index_map_mult + index_map_add).astype(np.int)] = aa
b_map[np.floor(yy*index_map_mult + index_map_add).astype(np.int)] = bb 
x = np.arange(0,len(a_map))
idx = a_map.nonzero()
fa = interpolate.interp1d(x[idx],a_map[idx])
fb = interpolate.interp1d(x[idx],b_map[idx])
offset = 29
xnew = np.arange(offset,len(a_map)-offset)
a_map[offset:-offset] = fa(xnew)
b_map[offset:-offset] = fb(xnew)


#load image struct
img_structs = sio.loadmat(os.path.join(avd_root,scene,'image_structs.mat'))
scale = img_structs['scale'][0][0]
img_structs = img_structs['image_structs']
img_structs_dict = {}
for img_st in img_structs[0,:]:
    img_structs_dict[img_st[0][0]] = img_st

img_struct = img_structs_dict[img_name]
t = img_struct[1]
R = img_struct[2]
if len(t) == 0:
    sys.exit(0)


depth_vals = depth_vals.reshape(1,-1)
depth_vals[depth_vals > 4500] = 0
pixel_pos = np.mgrid[0:org_img_shape[0],0:org_img_shape[1]]
#pixel_pos = np.mgrid[box[1]:box[3], box[0]:box[2]]
pixel_pos = pixel_pos.reshape((2,-1))
pixel_pos = pixel_pos[:,depth_vals.nonzero()[1]]
depth_vals = depth_vals[:,depth_vals.nonzero()[1]]




vv = pixel_pos[0,:]
uu = pixel_pos[1,:]
#crop around box
vv[vv>box[3]] = 0
vv[vv<box[1]] = 0
uu = uu[vv.nonzero()[0]]
depth_vals = depth_vals[:,vv.nonzero()[0]]
vv = vv[vv.nonzero()[0]]

uu[uu>box[2]] = 0
uu[uu<box[0]] = 0
vv = vv[uu.nonzero()[0]]
depth_vals = depth_vals[:,uu.nonzero()[0]]
uu = uu[uu.nonzero()[0]]


xx = (uu-intrinsic[0,2])/intrinsic[0,0]
yy = (vv-intrinsic[1,2])/intrinsic[1,1]

xx[xx>1] = 0
xx[xx<-1] = 0
depth_vals = depth_vals[:,xx.nonzero()[0]]
yy = yy[xx.nonzero()[0]]
xx = xx[xx.nonzero()[0]]

yy[yy>1] = 0
yy[yy<-1] = 0
depth_vals = depth_vals[:,yy.nonzero()[0]]
xx = xx[yy.nonzero()[0]]
yy = yy[yy.nonzero()[0]]

aa = a_map[np.floor(xx*index_map_mult + index_map_add).astype(np.int)]
bb = b_map[np.floor(yy*index_map_mult + index_map_add).astype(np.int)]

xc3 = np.asarray(depth_vals) /float(scale)
xc1 = aa * xc3
xc2 = bb * xc3
cam_coords = np.stack((xc1,xc2,xc3)).squeeze()

world_coords = np.matmul(R.transpose(), (cam_coords-t))

#prune point cloud
world_coords = prune_outliers(world_coords,0,2) 
world_coords = prune_outliers(world_coords,1,2) 
world_coords = prune_outliers(world_coords,2,2) 


write_numpy_to_ply(world_coords, '/playpen/ammirato/test.ply')
#world_coords_homog = np.concatenate((world_coords,np.ones((1,world_coords.shape[1]))))


box = project_points_to_image(world_coords,img_struct,intrinsic,distortion,org_img_shape)


#find an image that sees the projected points, and is within 90 veiwing angle, and check scale
for img_name in img_names:
    try:
        img_struct = img_structs_dict[img_name]
    except:
        print('no image struct: {}'.format(img_name))
        continue
    depth_name = img_name.replace('0101.jpg','0103.jpg')
    reader = png.Reader(os.path.join(avd_root,scene,'high_res_depth', depth_name))
    pngdata = reader.read()
    new_depth_img = np.array(map(np.uint16, pngdata[2]))

    new_box = project_points_to_image(world_coords,img_struct,intrinsic,distortion,org_img_shape,new_depth_img)
    if new_box == []:
        continue

    new_w = new_box[2] - new_box[0]
    new_h = new_box[3] - new_box[1]
    if new_w == 0 or new_h==0 or new_w/new_h > 1.5 or new_h/new_w > 1.5 or new_h*new_w<min_box or new_h*new_w>max_box:
        continue
    ##projected_points = np.round(np.stack((u,v)))    
    


#    break




#if True:
    new_img = cv2.imread(os.path.join(avd_root,scene,'jpg_rgb', img_name))
    cv2.rectangle(rgb_img,(box[0]/2,box[1]/2),(box[2]/2,box[3]/2),(255,0,0),4)
    cv2.rectangle(new_img,(new_box[0],new_box[1]),(new_box[2],new_box[3]),(255,0,0),4)
    cv2.imshow("org_image", rgb_img)
    cv2.imshow("new_image", new_img)
    k =  cv2.waitKey(0)
    if k == ord('q'):
        break
