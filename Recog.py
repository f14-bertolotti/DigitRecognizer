
import numpy as np
import cv2 as cv
import tensorflow as tf
import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.models import load_model
from optparse import OptionParser
from itertools import count

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=1)

drawing = False
point   = (-1,-1)
image   = np.full((512,512),0, dtype=np.uint8)
resized = np.full((28,28),  0, dtype=np.uint8)


def progres_bars(results, fill ='█'):
    string = "{:=<65}\n".format("")
    string += "Best guess: {}, confidence: {:2}\n".\
              format(list(results).index(max(results)), max(results))
    for e,i in zip(results,count()):
        string += "{}: {:35.35}:{}\n".format(i, fill*int(e*30),e)
    string += "{:=<65}\n".format("")
    return string;


def draw_wrapper(modeldir, verbose):

    committee = [load_model(modeldir+filename) for filename in os.listdir(modeldir) if 'best' in filename]

    def draw(event,x,y,flags,param):

        global drawing, point, image

        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            point = (x,y)
            for i in range(512):
                for j in range(512):
                    image[i][j] = 0
                    
        elif event == cv.EVENT_MOUSEMOVE and drawing == True:
             cv.line(image,point,(x,y),250,25)
             point = (x,y)

        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            resized = (cv.resize(image, dsize=(28,28), interpolation = cv.INTER_AREA)/255)
            if verbose: print(resized.reshape(28,28))
            res = np.zeros(10,dtype=float)
            for nn in committee:
                try: res += list(nn.predict(resized.reshape(1,28,28,1))[0])
                except(ValueError): res += list(nn.predict(resized.reshape(1,784))[0])
            res/=len(committee)
            print(progres_bars(res))


    return draw

if __name__ == '__main__':


    parser = OptionParser()
    parser.add_option("-f", "--modeldir", dest="modeldir", 
                      default="./mnist_models/CNN2D_type0-batch128/",
                        help="keras directory where modelx.best.hdf5 are stored")
    parser.add_option("-v", "--verbose", dest="verbose", 
                      default=False, action="store_true",
                      help="keras directory where modelx.best.hdf5 are stored")
    (options, args) = parser.parse_args()


    cv.namedWindow('Canvas')
    cv.setMouseCallback('Canvas',draw_wrapper(options.modeldir, options.verbose))
    while cv.getWindowProperty('Canvas', 0) >= 0:
        cv.imshow('Canvas',image)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()