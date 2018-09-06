#!/usr/bin/env python3
#
# A script to label images of the form [id],[x-euler],[y-euler],[z-euler]
#

import os, random, shutil, math, re, errno
import numpy as np

#TODO: holy crap gotta figure out all these frames
def cart2sph(x, y, z):
    '''cartesian to spherical coords'''
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def sph2cart(az, el, r):
    '''spherical to cartesian coords'''
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def parse_ln(ln):
    '''simple but might become more complex'''
    return ln.split(',')

def make_labels(filename):
    '''create dictionary of label: azimuth, elevation correspondence'''
    lbl = 0
    dct_cart = {}
    dct_sph = {}
    with open(filename,'r') as f:
        for line in f:
            x,y,z = parse_ln(line)
            dct_cart[lbl] = [float(x),float(y),float(z)]
            dct_sph[lbl] = cart2sph(float(x),float(y),float(z))
            lbl += 1
    return (dct_cart, dct_sph)

def dist_sq(v1, v2):
    '''calculate the squared norm since we're doing this just for comparison'''
    return ((v1[0]-v2[0])**2 + \
            (v1[1]-v2[1])**2 + \
            (v1[2]-v2[2])**2)

def closest_cam(x, y, z, dct_cart):
    '''find the camera location closest to the servicer'''
    #TODO: can be done faster with numpy arrays
    minkey = 0
    minval = 2**2       # highest distance in the unit sphere squared

    for key, value in dct_cart.items():
        tmp_dist = dist_sq([x,y,z], value)
        if (tmp_dist < minval):
            minval = tmp_dist
            minkey = key

    return minkey

def label_image(filename, dct_cart):
    # Parse the filename
    fn_l = filename.split('.')
    s = '.'.join(fn_l[:len(fn_l)-1])      # get rid of file extension
    s_l = s.split(',')
    s_l = s_l[1:]
    float_l = list(map(float, s_l))
    float_l = list(map(math.radians, float_l))
    float_l[2] = 1                      # currently no z-rotation, r = 1
    float_l = sph2cart(*float_l)        # change to cartesian

    # Find the closest camera
    label = closest_cam(*float_l, dct_cart)
    return str(label)

def is_png(filename):
    '''check if the file is a png'''
    # can probably find a better way
    file_l = filename.split('.')
    if (file_l[len(file_l)-1] == 'png'):
        return True
    return False

def main():
    input_dir = "/media/mannykoum/Data/synthetic_data/AcrimSat_random_no_grain_all/"
    output_dir = "/media/mannykoum/Data/synthetic_data/AcrimSat_random_no_grain/"
    camera_pos_fn = \
            "/media/mannykoum/Data/synthetic_data/coords_pts_300_its_10000.txt"

    # Make a dictionary (key=label, val=coordinates)
    dct_cart, dct_sph = make_labels(camera_pos_fn)
    l_files = os.listdir(input_dir)
    for file in l_files:
        # Skip if it isn't a png file
        if not(is_png(file)):
            continue

        # Label the image
        label = label_image(file, dct_cart)
        src_path = os.path.join(input_dir, file)
        dst_path = os.path.join(output_dir, label, file)

        # Make the dir if it doesn't exist
        if not os.path.exists(os.path.dirname(dst_path)):
            try:
                os.makedirs(os.path.dirname(dst_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # Copy the photo in the appropriate directory
        shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    main()
