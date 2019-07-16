from socketserver import ThreadingTCPServer, BaseRequestHandler
from random import random
import numpy as np
import sys, logging, json, cv2, json, time

server_ip, server_port = '', 50000
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')


def jpeg_decoder(jpeg_data):
    """
    Usage: Decode jpeg raw data to numpy array
    return: numpy ndarray
    """
    data_bytes = np.asarray(bytearray(jpeg_data), dtype=np.uint8) 
    np_img = cv2.imdecode(data_bytes, cv2.IMREAD_GRAYSCALE)
    return np_img


class RequestHandler(BaseRequestHandler):    
    def setup(self):
        logging.info('Client {} connect'.format(self.client_address[0]))
        return BaseRequestHandler.setup(self)
    
    def handle(self):
        while (True): #persistent connection
            try: #receive
                data_len =  int.from_bytes(self.my_recv(4), byteorder='little')
                jpeg_data = self.my_recv(data_len)
                np_face = jpeg_decoder(jpeg_data)
                time.sleep(0.1)
                report = {'joy': random(),
                          'calm': random(),
                          'sad': random(),
                          'stress':random()
                          } 
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
            except OSError as e:
                logging.error(str(e)) 
                return 
            
    def finish(self):
        logging.info('Client {} disconnect'.format(self.client_address[0]))
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
    server = ThreadingTCPServer((server_ip, server_port), RequestHandler)
    logging.info('Sample_server listening on port {}'.format(server_port))
    server.serve_forever()
    
