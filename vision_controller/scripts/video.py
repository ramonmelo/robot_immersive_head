#!/usr/bin/env python

'''
Video capture sample.

Sample shows how VideoCapture class can be used to acquire video
frames from a camera of a movie file. Also the sample provides
an example of procedural video generation by an object, mimicking
the VideoCapture interface (see Chess class).

'create_capture' is a convinience function for capture creation,
falling back to procedural video in case of error.

Usage:
    video.py [--shotdir <shot path>] [source0] [source1] ...'

    sourceN is an
     - integer number for camera capture
     - name of video file
     - synth:<params> for procedural video

Synth examples:
    synth:bg=../cpp/lena.jpg:noise=0.1
    synth:class=chess:bg=../cpp/lena.jpg:noise=0.1:size=640x480

Keys:
    ESC    - exit
    SPACE  - save current frame to <shot path> directory

'''

import numpy as np
import cv2
from time import clock
from numpy import pi, sin, cos
import common

class VideoSynthBase(object):
    def __init__(self, size=None, noise=0.0, bg = None, **params):
        self.bg = None
        self.frame_size = (640, 480)
        if bg is not None:
            self.bg = cv2.imread(bg, 1)
            h, w = self.bg.shape[:2]
            self.frame_size = (w, h)

        if size is not None:
            w, h = map(int, size.split('x'))
            self.frame_size = (w, h)
            self.bg = cv2.resize(self.bg, self.frame_size)

        self.noise = float(noise)

    def render(self, dst):
        pass

    def read(self, dst=None):
        w, h = self.frame_size

        if self.bg is None:
            buf = np.zeros((h, w, 3), np.uint8)
        else:
            buf = self.bg.copy()

        self.render(buf)

        if self.noise > 0.0:
            noise = np.zeros((h, w, 3), np.int8)
            cv2.randn(noise, np.zeros(3), np.ones(3)*255*self.noise)
            buf = cv2.add(buf, noise, dtype=cv2.CV_8UC3)
        return True, buf

    def isOpened(self):
        return True

class Chess(VideoSynthBase):
    def __init__(self, **kw):
        super(Chess, self).__init__(**kw)

        w, h = self.frame_size

        self.grid_size = sx, sy = 10, 7
        white_quads = []
        black_quads = []
        for i, j in np.ndindex(sy, sx):
            q = [[j, i, 0], [j+1, i, 0], [j+1, i+1, 0], [j, i+1, 0]]
            [white_quads, black_quads][(i + j) % 2].append(q)
        self.white_quads = np.float32(white_quads)
        self.black_quads = np.float32(black_quads)

        fx = 0.9
        self.K = np.float64([[fx*w, 0, 0.5*(w-1)],
                        [0, fx*w, 0.5*(h-1)],
                        [0.0,0.0,      1.0]])

        self.dist_coef = np.float64([-0.2, 0.1, 0, 0])
        self.t = 0

    def draw_quads(self, img, quads, color = (0, 255, 0)):
        img_quads = cv2.projectPoints(quads.reshape(-1, 3), self.rvec, self.tvec, self.K, self.dist_coef) [0]
        img_quads.shape = quads.shape[:2] + (2,)
        for q in img_quads:
            cv2.fillConvexPoly(img, np.int32(q*4), color, cv2.CV_AA, shift=2)

    def render(self, dst):
        t = self.t
        self.t += 1.0/30.0

        sx, sy = self.grid_size
        center = np.array([0.5*sx, 0.5*sy, 0.0])
        phi = pi/3 + sin(t*3)*pi/8
        c, s = cos(phi), sin(phi)
        ofs = np.array([sin(1.2*t), cos(1.8*t), 0]) * sx * 0.2
        eye_pos = center + np.array([cos(t)*c, sin(t)*c, s]) * 15.0 + ofs
        target_pos = center + ofs

        R, self.tvec = common.lookat(eye_pos, target_pos)
        self.rvec = common.mtx2rvec(R)

        self.draw_quads(dst, self.white_quads, (245, 245, 245))
        self.draw_quads(dst, self.black_quads, (10, 10, 10))


classes = dict(chess=Chess)

presets = dict(
    empty = 'synth:',
    lena = 'synth:bg=../cpp/lena.jpg:noise=0.1',
    chess = 'synth:class=chess:bg=../cpp/lena.jpg:noise=0.1:size=640x480'
)

def create_capture(source = 0, fallback = presets['chess']):
    '''source: <int> or '<int>|<filename>|synth [:<param_name>=<value> [:...]]'
    '''
    source = str(source).strip()
    chunks = source.split(':')
    # hanlde drive letter ('c:', ...)
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )

    cap = None
    if source == 'synth':
        Class = classes.get(params.get('class', None), VideoSynthBase)
        try: cap = Class(**params)
        except: pass
    else:
        cap = cv2.VideoCapture(source)
        if 'size' in params:
            w, h = map(int, params['size'].split('x'))
            cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, h)
    if cap is None or not cap.isOpened():
        print 'Warning: unable to open video source: ', source
        if fallback is not None:
            return create_capture(fallback, None)
    return cap


class StereoVision(object):

    def __init__(self):

        self.res = (960, 600)

        # Intrinsic of the original view
        int_params_0 = np.array([[self.res[0]*0.5, 0, self.res[0]*0.5], [0, self.res[1]*0.5, self.res[1]*0.5], [0, 0, 1]])
        int_params_1 = np.array([[self.res[0]*0.5, 0, self.res[0]*0.5], [0, self.res[1]*0.5, self.res[1]*0.5*(190.0/240.0)], [0, 0, 1]])

        self.int_params_original = np.array([ int_params_0, int_params_1 ])

        # Intrinsic of the new view
        int_params_0 = np.array([[self.res[0]*0.2, 0, self.res[0]*0.2863248], [0, self.res[1]*0.30, self.res[1]*0.5], [0, 0, 1]])
        int_params_1 = np.array([[-self.res[0]*0.2, 0, self.res[0]*0.7136753], [0, -self.res[1]*0.30, self.res[1]*0.5], [0, 0, 1]])

        self.int_params_new = np.array([ int_params_0, int_params_1 ])

        #Distortion parameters
        self.distortion_params = np.array([0.21, -0.05, 0, 0, 0.03])

    def create_stereo_image(self, right_image, left_image):

        r_id = 1
        l_id = 0

        right_dst = cv2.resize( right_image, self.res )
        left_dst = cv2.resize( left_image, self.res )

        right_dst = cv2.undistort(right_dst, self.int_params_original[r_id], self.distortion_params, None, self.int_params_new[r_id])

        left_dst = cv2.undistort(left_dst, self.int_params_original[l_id], self.distortion_params, None, self.int_params_new[l_id])

        h, w = right_dst.shape[:2]

        draw = np.zeros((h, w, 3), np.uint8)
        draw = right_dst + left_dst

        return draw

if __name__ == '__main__':
    import sys
    import getopt

    print __doc__

    args, sources = getopt.getopt(sys.argv[1:], '', 'shotdir=')
    args = dict(args)
    shotdir = args.get('--shotdir', '.')
    if len(sources) == 0:
        sources = [ 0 ]

    res = (960,600)

    #Intrinsic of the original view
    oK1 = np.array([[res[0]*0.5, 0, res[0]*0.5], [0, res[1]*0.5, res[1]*0.5*(190.0/240.0)], [0, 0, 1]])
    oK0 = np.array([[res[0]*0.5, 0, res[0]*0.5], [0, res[1]*0.5, res[1]*0.5], [0, 0, 1]])

    K = np.array([oK0,oK1])

    ##d = np.array([1, 0.22, 0, 0, 0.25])

    #Intrinsic of the new view
    K1 = np.array([[res[0]*0.2, 0, res[0]*0.2863248], [0, res[1]*0.30, res[1]*0.5], [0, 0, 1]])
    K0 = np.array([[-res[0]*0.2, 0, res[0]*0.7136753], [0, -res[1]*0.30, res[1]*0.5], [0, 0, 1]])

    print oK1
    print K1

    M = np.array([K0,K1])

    #oK1 = np.array([[320, 0, 320], [0, 240, 190], [0, 0, 1]])
    #oK0 = np.array([[320, 0, 320], [0, 240, 240], [0, 0, 1]])

    #K = np.array([oK0,oK1])

    #d = np.array([1, 0.22, 0, 0, 0.25])

    #K1 = np.array([[180, 0, 640*0.2863248], [0, 210, 240], [0, 0, 1]])
    #K0 = np.array([[-180, 0, 640*0.7136753], [0, -210, 240], [0, 0, 1]])

    #M = np.array([K0,K1])
    # k0 = 0
    # k1 = 0
    # k2 = 0

    caps = map(create_capture, sources)
    print enumerate(caps)
    shot_idx = 0
    while True:
        #Distortion parameters
        d = np.array([0.21, -0.05, 0, 0, 0.03])
        imgs = []
        for i, cap in enumerate(caps):

            ret, img = cap.read()
            h, w = img.shape[:2]

            dst = cv2.resize(img,res)


            newimg = cv2.undistort(dst, K[i], d, None, M[i])
            imgs.append(newimg)

           # cv2.imshow('capture %d' % i, img)
           # cv2.imshow('distort %d' % i, newimg)

        h,w = imgs[0].shape[:2]


        draw = np.zeros((h, w, 3), np.uint8)

        draw = imgs[0] + imgs[1]



        cv2.namedWindow('draw', 0)
        cv2.imshow('draw', draw)

               #for i, img in enumerate(imgs):

        # ch = 0xFF & cv2.waitKey(1)
        # if ch == ord('q'):
        #     k0+=0.01
        # if ch == ord('a'):
        #     k0-=0.01
        # if ch == ord('w'):
        #     k1+=0.01
        # if ch == ord('s'):
        #     k1-=0.01
        # if ch == ord('e'):
        #     k2+=0.01
        # if ch == ord('d'):
        #     k2-=0.01

        # print k0,k1,k2

        #    break
        #if ch == ord(' '):
        #    for i, img in enumerate(imgs):
        #        fn = '%s/shot_%d_%03d.bmp' % (shotdir, i, shot_idx)
        #        cv2.imwrite(fn, img)
        #        print fn, 'saved'
        #    shot_idx += 1
    cv2.destroyAllWindows()
