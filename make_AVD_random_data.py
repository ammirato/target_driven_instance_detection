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
import random

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


def project_points_to_image(world_coords,img_struct,intrinsic,distortion,img_shape,d_img=None, scale=None):
    try:
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

        up[up<0] = 0
        up[up>=img_shape[1]] = 0
        vp = vp[up.nonzero()]
        up = up[up.nonzero()]
        world_coords = world_coords[:,up.nonzero()]
        vp[vp<0] = 0
        vp[vp>=img_shape[0]] = 0
        up = up[vp.nonzero()]
        vp = vp[vp.nonzero()]
        world_coords = cur_world_points[:,vp.nonzero()].squeeze()

        ui = up
        vi = vp

        if len(ui) == 0 or len(vi) == 0:
            print('No inside image')
            return [] 

        xvals = np.floor(np.asarray(ui))
        yvals = np.floor(np.asarray(vi))

        #try:
        #    assert(world_coords.shape[1] == xvals.shape[0])
        #except:
        #    breakp = 1
        #    return []
        if d_img is not None:
            reader = png.Reader(d_img)
            pngdata = reader.read()
            d_img = np.array(map(np.uint16, pngdata[2]))
            depth_vals = d_img[yvals.astype(np.int), xvals.astype(np.int)]
            depth_vals[depth_vals>4500] = 0
            xvals = xvals[depth_vals.nonzero()]        
            yvals = yvals[depth_vals.nonzero()]        
            world_coords = world_coords[:,depth_vals.nonzero()]
            depth_vals = depth_vals[depth_vals.nonzero()]

            if len(xvals) ==0 or len(yvals) == 0:
                print('Bad depth')
                return []  

            if scale is not None:
                cam_pos = img_struct[3]*scale
                cam_dir = img_struct[4]
                world_coords *= scale
                cam_point_vecs = world_coords.squeeze() - cam_pos
                dists = np.matmul(cam_point_vecs.transpose(), cam_dir).squeeze()
                diffs = abs(dists - depth_vals) + 5
                diffs[diffs > 500] = 0
                xvals = xvals[diffs.nonzero()] 
                yvals = yvals[diffs.nonzero()] 
                if len(xvals) ==0 or len(yvals) == 0:
                    print('occuleded')
                    return [] 

        new_box = [int(xvals.min()), int(yvals.min()), int(xvals.max()), int(yvals.max())]
        return new_box 
    except:
        print('Unknow exception')
        return []


def prune_outliers(array, dim, nstd):
    values = array[dim,:]
    return array[:,abs(values - values.mean()) < nstd*values.std()]


outfile_path = '/playpen/ammirato/AVD_extra_10.txt'

max_box = 50000
min_box = 1500
#set paths to data
avd_root2 = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
#avd_root2 = '/playpen/ammirato/Data/RohitMetaData/'
avd_root = '/playpen/ammirato/Data/RohitData/'
#scene_list = ['Home_003_1','Home_001_1', 'Home_004_1', 'Home_002_1', 'Home_005_1']
scene_list = ['Home_003_1','Home_001_1', 'Home_004_1', 'Home_002_1']
num_runs = 5
#scene_list = ['Home_005_1', 'Home_005_2', 'Home_006_1']
#avd_root = avd_root2 
#pick a scene
#scene = 'Home_003_1'

for scene in scene_list:
    #pick an image and semi-random bbox in that image
    img_paths = glob.glob(os.path.join(avd_root,scene,'jpg_rgb','*.jpg'))
    img_names = os.listdir(os.path.join(avd_root,scene,'jpg_rgb'))
    num_written = 0
    #load image structs
    img_structs = sio.loadmat(os.path.join(avd_root,scene,'image_structs.mat'))
    scale = img_structs['scale'][0][0]
    img_structs = img_structs['image_structs']
    img_structs_dict = {}
    for img_st in img_structs[0,:]:
        img_structs_dict[img_st[0][0]] = img_st

    for xl in range(num_runs):    
        random.shuffle(img_paths)
        for img_path in img_paths:

            img_name = img_path[img_path.rfind('/')+1:]
            org_img_name = img_name
            img_path = os.path.join(avd_root,scene,'jpg_rgb',img_name)

            img_struct = img_structs_dict[img_name]
            t = img_struct[1]
            R = img_struct[2]
            if len(t) == 0:
                print('no org t')
                continue
            org_cam_vec = img_struct[3] + img_struct[4]
            org_cam_pos = img_struct[3]
            org_cam_dir = img_struct[4]  + org_cam_pos
            org_slope = (org_cam_pos[2] - org_cam_dir[2]) / (org_cam_pos[0] - org_cam_dir[0])
            org_intercept = org_cam_pos[2] - org_slope*org_cam_pos[0]

            d_img_path = img_path.replace('jpg_rgb','high_res_depth').replace('01.jpg','03.png')
            rgb_img = cv2.imread(img_path)
            org_img_shape = rgb_img.shape
            rgb_img = cv2.resize(rgb_img,(0,0),fx=.5,fy=.5)
            reader = png.Reader(d_img_path)
            pngdata = reader.read()
            depth_vals = np.array(map(np.uint16, pngdata[2]))

            #choose a region proposal
            img_lbl, regions = selectivesearch.selective_search(rgb_img, scale=500, sigma=0.9, min_size=10)
            box = []
            counter = 0
            max_props_searched = 200
            while len(box) == 0:
                counter +=1
                if counter == max_props_searched:
                    break
                ind = np.random.choice(len(regions))
                bsize =regions[ind]['size'] 
                bsize*= 4
                if bsize > min_box and bsize < max_box:
                    x,y,w,h= regions[ind]['rect'] 
                    x *=2
                    y *=2
                    w *=2
                    h *=2
                    if w==0 or h==0 or w/h > 1.5 or h/w > 1.5:
                        continue
                    box = [x,y,x+w,y+h]
            print('Prop search count {}'.format(counter))
            if counter == max_props_searched:
                continue 
            #project points in bbox to world coordinates
            #camera follows:  http://docs.opencv.org/master/db/d58/group__calib3d__fisheye.html#gsc.tab=0 
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




            depth_vals = depth_vals.reshape(1,-1)
            depth_vals[depth_vals > 4500] = 0
            pixel_pos = np.mgrid[0:org_img_shape[0],0:org_img_shape[1]]
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
            #write_numpy_to_ply(world_coords, '/playpen/ammirato/test.ply')
            #world_coords_homog = np.concatenate((world_coords,np.ones((1,world_coords.shape[1]))))

            box = project_points_to_image(world_coords,img_struct,intrinsic,distortion,org_img_shape)
            if box == []:
                print('bad org box')
                continue
            b_w = box[2]-box[0]
            b_h = box[3]-box[1]
            if box == [] or b_w == 0 or b_h==0 or b_w/b_h > 1.5 or b_h/b_w > 1.5:
                continue 

            #find an image that sees the projected points, and is within 90 veiwing angle, and check scale
            counter= 0 
            random.shuffle(img_names)
            for img_name in img_names:
                if img_name == org_img_name:
                    continue
                counter += 1
                if counter > 200:
                    break
                try:
                    img_struct = img_structs_dict[img_name]
                except:
                    print('no image struct: {}'.format(img_name))
                    continue
                #check image directions
                cur_cam_vec = img_struct[3] + img_struct[4]
                cur_cam_pos = img_struct[3]
                cur_cam_dir = img_struct[4] + cur_cam_pos
                cur_slope =    (cur_cam_pos[2] - cur_cam_dir[2]) / (cur_cam_pos[0] - cur_cam_dir[0])
                cur_intercept = cur_cam_pos[2] - cur_slope*cur_cam_pos[0]
                x_intersection = (cur_intercept - org_intercept) / (org_slope - cur_slope)
                y_intersection = org_slope * x_intersection + org_intercept
                #intersection should be on same side as direction
                if     ((org_cam_dir[0] > org_cam_pos[0] and x_intersection < org_cam_pos[0]) or
                        (org_cam_dir[0] < org_cam_pos[0] and x_intersection > org_cam_pos[0]) or
                        (cur_cam_dir[0] < cur_cam_pos[0] and x_intersection > cur_cam_pos[0]) or
                        (cur_cam_dir[0] < cur_cam_pos[0] and x_intersection > cur_cam_pos[0])):
                    print('bad interection')
                    continue
                #get interseciton angle
                sideA = np.sqrt(np.square(org_cam_pos[0]-cur_cam_pos[0]) + np.square(org_cam_pos[2]-cur_cam_pos[2]))
                sideB = np.sqrt(np.square(org_cam_pos[0]-x_intersection) + np.square(org_cam_pos[2]-y_intersection))
                sideC = np.sqrt(np.square(cur_cam_pos[0]-x_intersection) + np.square(cur_cam_pos[2]-y_intersection))
                angle = np.rad2deg(np.arccos((sideC*sideC + sideB*sideB - sideA*sideA) / (2*sideC*sideB)))
                if angle < 5 or angle > 75:
                    print('bad angle')
                    continue
     

                depth_name = img_name.replace('0101.jpg','0103.png')
                depth_path = os.path.join(avd_root,scene,'high_res_depth', depth_name)

                new_box = project_points_to_image(world_coords,img_struct,intrinsic,distortion,org_img_shape,d_img=depth_path, scale=scale)
                if new_box == []:
                    print('empty new box')
                    continue

                new_w = new_box[2] - new_box[0]
                new_h = new_box[3] - new_box[1]
                if new_w == 0 or new_h==0 or new_w/new_h > 1.5 or new_h/new_w > 1.5 or new_h*new_w<min_box or new_h*new_w>max_box:
                    print('new box bad dims')
                    continue

                outfid = open(outfile_path, 'a')
                outfid.write('{}#{}#{}#{}\n'.format(org_img_name,box,img_name,new_box))
                outfid.close()
                num_written+=1
                print('sucess {}'.format(num_written))
                
                #new_img = cv2.imread(os.path.join(avd_root,scene,'jpg_rgb', img_name))
                #cv2.rectangle(rgb_img,(box[0]/2,box[1]/2),(box[2]/2,box[3]/2),(255,0,0),4)
                #cv2.rectangle(new_img,(new_box[0],new_box[1]),(new_box[2],new_box[3]),(255,0,0),4)
                #cv2.imshow("org_image", rgb_img)
                #cv2.imshow("new_image", new_img)
                #k =  cv2.waitKey(0)
                #if k == ord('q'):
                #    break
