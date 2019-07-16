# name: predict_emotion_server
# made: multimedia communication lab.
# date: 18. 7. 3

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from time import time
from socketserver import ThreadingTCPServer, StreamRequestHandler
import numpy as np
import os, sys, logging, json, requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
#model_path = 'samplewise_norm_model+aug-Copy1.hdf5'
model_path ="v_t_aug__input_114.hdf5"
#model_mean = [[[119.09373]]]
classes = {0:'angry', 1:'fear', 2:'happy', 3:'neutral', 4:'sad', 5:'surprise'}
emotions = [e.upper() for e in list(classes.values())]

model = load_model(model_path)

"""
valid_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True, 
)
"""
valid_datagen = ImageDataGenerator(
    samplewise_std_normalization=True, 
)

def prepare(test_img):
    '''
    Usage: The first time to run server, loading utils into memory
    return: None
    '''
    valid_x, valid_y = [], []

    z = load_img(path = test_img, grayscale=True, target_size=(114,114), interpolation='nearest')
    q = np.asarray(z).astype('float32')
    q = np.asarray([q])
    
    valid_x.append(q)
    valid_x = np.asarray(valid_x)
    valid_x_moveaxis = np.moveaxis(valid_x, 1, 3)  
    
    valid_y.append([0, 0, 0, 0, 0, 0])
    valid_y = np.asarray(valid_y)

    valid_datagen.fit(valid_x_moveaxis)
    #valid_datagen.mean = model_mean

    prob = model.predict_generator(valid_datagen.flow(valid_x_moveaxis, valid_y, batch_size=1, shuffle=False), steps=1)
    logging.debug(prob[0], classes[np.argmax(prob[0])])
    
    logging.info('Server ready')

    
def classfy(data):
    """
    Usage: Classfify into emotions with the prediction probability
    return: list of probabilities
    """
    _ = np.array([x for x in data], dtype='uint8') #vectorize
    face = np.reshape(_, (114,114)) #227x227 dims
        
    valid_x, valid_y = [], []
    reply = {}
    
    tic = time()
    q = np.asarray(face).astype('float32')
    q = np.asarray([q])
    
    valid_x.append(q)
    valid_x = np.asarray(valid_x)
    valid_x_moveaxis = np.moveaxis(valid_x, 1, 3)  
    
    valid_y.append([0, 0, 0, 0, 0, 0])
    valid_y = np.asarray(valid_y)

    valid_datagen.fit(valid_x_moveaxis)
    #valid_datagen.mean = model_mean

    prob = model.predict_generator(valid_datagen.flow(valid_x_moveaxis, valid_y, batch_size=1, shuffle=False), steps=1)
    
    for i in range(0, len(emotions)):
        reply[emotions[i]] = float(prob[0][i]) #reply formet: {"EMOTION": probability, ...}
    
    return reply


class RequestHandler(StreamRequestHandler):
    def handle(self):
        while (True):
            self.data = self.my_recv(114*114)
            self.data_len = len(self.data)
            logging.info('Client {} wrote:'.format(self.client_address[0]))
            logging.info(self.data_len)
            
            try: #request
                report = classfy(self.data)
            except Exception as e:
                reply = {'STATUS':0,
                         'REPORT':str(e)}
                logging.error(str(e))
            else:
                reply = {'STATUS':1,
                         'REPORT':[report]}
                
            reply_json = json.dumps(reply)
            try: #reply
                logging.info('Server wrote: ')
                logging.info(reply_json)
                self.wfile.write(bytes(reply_json, 'utf-8'))
                #res = requests.post(url, data=reply_json) # send HTTP post
                #logging.info('Send jason reply into HTTP post method')
            except Exception as e:
                logging.error(str(e))
                return      # cannot send. Closing connection
            if reply['STATUS'] == 1:
                continue      # cause to close connection
            else:
                return
            
    def my_recv(self, size):
        msg = b''
        while len(msg) < size:
            chunk = self.rfile.read(size-len(msg))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            msg = msg + chunk
        return msg
    
    
if __name__ == '__main__':
    prepare('bbb.png')
    server = ThreadingTCPServer(('', 50000), RequestHandler)
    server.serve_forever()


