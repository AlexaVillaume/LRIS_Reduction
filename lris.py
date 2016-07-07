import sys
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
from astropy.io import fits
from astropy.io.fits import getheader
from astropy.io.fits import writeto
from astropy.convolution import Gaussian1DKernel, convolve


blue_side = {'l1': 3300, 'l2': 6900, 'pixsize': 0.135, 'outpix': 2700,
            'fully': 4096, 'dbar': 193, 'order': [1, 2, 3, 4],
            'bars': [409, 459, 639, 691], 'ressubx': [1953, 2281],
            'ressuby': [75,200,616,728], 'l_abs': [6270, 6330],
            # gain is approximate average of 4 detectors; varies by ~5%
            # this is electrons / ADU
            'gain': 1.6,
            # fudge factor to bring noise in line with model - it seems to
            # be close to sqrt(2), but not sure what cause is ..
            # could be combination of too low sky model (as sky_min) and
            # some smoothing because of resampling ..
            'noise_fac': 1.0,
            # vacuum wavelength of strong OH line in ~ middle of spectrum
            #  from ESO page:
            #  http://www.eso.org/observing/dfo/quality/UVES/pipeline/sky_spectrum.html
            'sky_ref': np.array([5578.88,6302.04]),
            # coefficients for s distortion correction - derived in sdist/
            'sdist_a': np.array([0.0058983, -7.870595])
            }

red_side = {'l1': 7450, 'l2': 10750, 'pixsize': 0.270, 'outpix': 4096,
            'fully': 2048, 'dbar': 162, 'order': [2, 1, 4, 3],
            'bars': [234, 254, 309, 331], 'ressubx': [1953, 2281],
            'ressuby': [75, 200, 616, 728], 'l_abs': [9300, 9700],
            # gain is approximate average of 4 detectors; varies by ~5%
            # this is electrons / ADU
            'gain': 1.2,
            # fudge factor to bring noise in line with model - it seems to
            # be close to sqrt(2), but not sure what cause is ..
            # could be combination of too low sky model (as sky_min) and
            # some smoothing because of resampling ..
            'noise_fac': 1.4,
            # from ESO page:
            # http://www.eso.org/observing/dfo/quality/UVES/pipeline/sky_spectrum.html
            # sky_ref = np.array([7964.65, 8399.18, 8885.86, 8919.64, 9914.66])
            # ESO sky is air, not vacuum !!
            'sky_ref': np.array([7966.84, 8401.49, 8888.30, 8922.09, 9917.38]),
            # coefficients for s distortion correction - derived in sdist/
            'sdist_a': np.array([0.000520129, -1.161418])
            }

class Reduction(object):
    def __init__(self, galaxy):

        self.outroot = 'out/'
        self.lisroot = 'lists/'
        self.in_root = 'raw_cut/'
        self.lamroot = 'lambda_'

        self.galaxy = galaxy

        self.im1_lis = None
        self.im2_lis = None
        self.im3_lis = None
        self.arc_lis = None
        self.out_root = None
        self.off = None
        self.doskylam = None
        self.do1sub = None
        self.docal = None
        self.doabs = None

        #self.l1 = None
        #self.l2 = None
        #self.pixsize = None
        #self.outpix = None
        #self.fully = None
        #self.dbar = None
        #self.order = None
        #self.bars = None
        #self.ressubx = None
        #self.ressuby = None
        #self.l_abs = None
        #self.gain = None
        #self.noise_fac = None
        #self.sky_ref = None
        #self.sdist_a = None

    def read_lis(self):
        path = ('../{0}{1}.lis'.format(self.lisroot, self.galaxy))
        allsets = np.genfromtxt(path, dtype='str')

        self.im1_lis = allsets[:,0]
        self.im2_lis = allsets[:,1]
        self.im3_lis = allsets[:,2]
        self.arc_lis = allsets[:,3]
        self.out_root = allsets[:,4]

        dither = allsets[:,5]
        dither = dither.astype(np.int)
        self.doff = np.amax(dither) - dither

        doskylam = allsets[:,6]
        self.doskylam = doskylam.astype(np.int)

        do1dsub = allsets[:,7]
        self.do1dsub = do1dsub.astype(np.int)

        docal = allsets[:,8]
        self.docal = docal.astype(np.int)

        doabs = allsets[:,9]
        self.doabs = doabs.astype(np.float)

class Frame(object):
    def __init__(self, arc_lis):
        if arc_lis[0] == 'r':
            self.l1 = red_side['l1']
            self.l2 = red_side['l2']
            self.pixsize = red_side['pixsize']
            self.outpix = red_side['outpix']
            self.fully = red_side['fully']
            self.dbar = red_side['dbar']
            self.order = red_side['order']
            self.bars = red_side['bars']
            self.ressubx = red_side['ressubx']
            self.ressuby = red_side['ressuby']
            self.l_abs = red_side['l_abs']
            self.gain = red_side['gain']
            self.noise_fac = red_side['noise_fac']
            self.sky_ref = red_side['sky_ref']
            self.sdist_a = red_side['sdist_a']
        elif arc_lis[0] == 'b':
            self.l1 = blue_side['l1']
            self.l2 = blue_side['l2']
            self.pixsize = blue_side['pixsize']
            self.outpix = blue_side['outpix']
            self.fully = blue_side['fully']
            self.dbar = blue_side['dbar']
            self.order = blue_side['order']
            self.bars = blue_side['bars']
            self.ressubx = blue_side['ressubx']
            self.ressuby = blue_side['ressuby']
            self.l_abs = blue_side['l_abs']
            self.gain = blue_side['gain']
            self.noise_fac = blue_side['noise_fac']
            self.sky_ref = blue_side['sky_ref']
            self.sdist_a = blue_side['sdist_a']

    def get_sky_scaling(self, reduction, i):
        filename = ('{0}{1}_{2}.fits'.format(reduction.in_root,
                reduction.im1_lis[i], self.order[0]))
        print filename
        sys.exit()
        # Median filter file

        filename = ('{0}{1}_{2}.fits'.format(reduction.in_root,
                reduction.im2_lis[i], self.order[0]))
        # Median filter file

        filename = ('{0}{1}_{2}.fits'.format(reduction.in_root,
                reduction.im3_lis[i], self.order[0]))
        # Median filter file

        return 0

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(outdir)

if __name__=='__main__':
    galaxy = 'NGC1407_r'

    reduction = Reduction(galaxy)
    reduction.read_lis()

    for i, name in enumerate(reduction.im1_lis):
        frame = Frame(reduction.arc_lis[i])
        frame.get_sky_scaling(reduction, i)
