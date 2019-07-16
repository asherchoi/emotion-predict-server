# -*- coding: utf-8 -*-

# name: kinect_emotion_server
# made: multimedia communication lab.
# date: 18. 7. 18

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from socketserver import ThreadingTCPServer, BaseRequestHandler
from threading import Timer
from threading import Lock
from time import time
import numpy as np
import sys, logging, json, cv2, json, http.client, urllib

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s')

url='http://220.67.127.70:8000/emoR/facial?prob={},{},{},{}&present={}'
host = urllib.parse.urlparse(url).netloc

stack = []
lock = Lock()
http_get_start = False

server_ip, server_port = '', 50000
classes = {0:'angry', 1:'fear', 2:'happy', 3:'neutral', 4:'sad', 5:'surprise'}
model_path = 'v_t_aug__input_114.hdf5'
model = load_model(model_path)

valid_datagen = ImageDataGenerator(
    samplewise_std_normalization=True, 
)


def prepare(test_img):
    '''
    Usage: The first time to run server, loading utils into memory
    return: None
    '''
    valid_x, valid_y = [], []
    z = load_img(path=test_img, grayscale=True, target_size=(114,114), interpolation='nearest')
    z = np.asarray(z).astype(dtype=np.uint8)
    z = cv2.equalizeHist(z)
    q = np.asarray(z).astype('float32')
    q = np.asarray([q])
    
    valid_x.append(q)
    valid_x = np.asarray(valid_x)
    valid_x_moveaxis = np.moveaxis(valid_x, 1, 3)  
    valid_y.append([0, 0, 0, 0, 0, 0])
    valid_y = np.asarray(valid_y)

    valid_datagen.fit(valid_x_moveaxis)
    prob = model.predict_generator(valid_datagen.flow(valid_x_moveaxis, valid_y, batch_size=1, shuffle=False), steps=1)
    print(prob)
    
    
def classify(data):
    """
    Usage: Classify face image's emotions with the prediction probability
    return: dict which mapped emotions and probabilities
    """
    face = cv2.resize(data, dsize=(114,114), interpolation=cv2.INTER_CUBIC)
        
    valid_x, valid_y = [], []
    #_face = np.asarray(face).astype(dtype=np.uint8)
    #equalize_hist_face = cv2.equalizeHist(_face)
    q = np.asarray(face).astype('float32')
    q = np.asarray([q])
    
    valid_x.append(q)
    valid_x = np.asarray(valid_x)
    valid_x_moveaxis = np.moveaxis(valid_x, 1, 3)  
    valid_y.append([0, 0, 0, 0, 0, 0])
    valid_y = np.asarray(valid_y)
    valid_datagen.fit(valid_x_moveaxis)

    prob = model.predict_generator(valid_datagen.flow(valid_x_moveaxis, valid_y, batch_size=1, shuffle=False), steps=1)
    angry, fear, happy, neutral, sad, surprise = (float(x) for x in prob[0])
    reply = {
        'angry': angry,
        'fear': fear,
        'happy': happy,
        'neutral':neutral,
        'sad': sad,
        'surprise': surprise}    
    return reply


def jpeg_decoder(jpeg_data):
    """
    Usage: Decode jpeg raw data to numpy array
    return: numpy ndarray
    """
    data_bytes = np.asarray(bytearray(jpeg_data), dtype=np.uint8) 
    np_img = cv2.imdecode(data_bytes, cv2.IMREAD_GRAYSCALE)
    return np_img


def threading_http_geter():
    if stack:
        with lock: #mutual exclusion with threading tcp server
            newest_report = stack.pop()
            stack.clear()
            
        get_data = {'joy': newest_report['surprise'] + newest_report['happy'],
                    'calm': newest_report['neutral'],
                    'sad': newest_report['sad'],
                    'stress': newest_report['fear'] + newest_report['angry']}    
        get_url = url.format(get_data['joy'], get_data['calm'], get_data['sad'], get_data['stress'], 1)
    else:
        get_url = url.format(0, 0, 0, 0, 0)
    try:
        conn = http.client.HTTPConnection(host)
        u = urllib.parse.urlparse(get_url)
        conn.request("GET", u.path+'?'+u.query) #, headers={'connection': 'keep-alive'}
        r = conn.getresponse()
        conn.close()
    except OSError as e: #network error
        logging.error('Can not send {} cause of {}'.format(get_url, e))
    else:
        if r.status == 200: #200 OK
            logging.debug('Get {} is {} {}'.format(get_url, r.status, r.reason))
        else: #not 200 OK
            logging.error('Get {} is {} {}'.format(get_url, r.status, r.reason))
    if http_get_start:
        Timer(2, threading_http_geter).start()


class RequestHandler(BaseRequestHandler):    
    def setup(self):
        logging.info('Client {} connect'.format(self.client_address[0]))
        return BaseRequestHandler.setup(self)
    
    def handle(self):
        while True: #persistent connection
            try: #receive
                data_len =  int.from_bytes(self.my_recv(4), byteorder='little')
                logging.debug(data_len) 
                jpeg_data = self.my_recv(data_len)
                np_face = jpeg_decoder(jpeg_data)  
                report = classify(np_face)
                logging.debug(report) 
            except OSError as e: 
                logging.error(str(e)) 
                return 
            except Exception as e:
                logging.error(str(e))
                reply = ['ERROR', str(e)]
            else:
                reply = ['OK', report]
            
            reply_json = json.dumps(reply)
                
            try:
                self.request.send(bytes(reply_json, 'utf-8'))
                with lock: #mutual exclusion with threading http geter
                    stack.append(report) 
                global http_get_start #this variable is global
                if not http_get_start: #if not http get method is start state
                    http_get_start = True #change state
                    threading_http_geter() #start get thread
            except OSError as e:
                logging.error(str(e)) 
                return 
            
    def finish(self):
        logging.info('Client {} disconnect'.format(self.client_address[0]))
        global http_get_start
        http_get_start = False #stop http get thread
        return BaseRequestHandler.finish(self)
    
    def my_recv(self, size):
        more = size
        msgs = []
        while more > 0:
            chunk = self.request.recv(more)
            if chunk:
                msgs.append(chunk)
            else:
                raise ConnectionError
            more -= len(chunk)
        return b''.join(msgs)
    
    
if __name__ == '__main__':    
    prepare('test.png')
    server = ThreadingTCPServer((server_ip, server_port), RequestHandler)
    logging.info('Server listening on port {}'.format(server_port))
    server.serve_forever()


