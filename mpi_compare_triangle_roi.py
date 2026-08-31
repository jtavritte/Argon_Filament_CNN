# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 00:30:43 2024

@author: jta0030

"""

#add change of entry

from glob import glob
import numpy as np

import cv2

import matplotlib.pyplot as plt
import sys
import time
from multiprocessing import Pool
import os
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import gc
import heapq

def Make_Fil(t):
    fil = Fil(imdat=cv2.imread(f'{shared_folder}/{shared_dict[t]}.tif', cv2.IMREAD_UNCHANGED),rad=shared_rad,xor=shared_xor,yor=shared_yor)
    #mode_fft = np.array([f[2] for f in fil.fil_char.values()])
    #modes,score = Fil.predict_image_fil_char(fil.fil_char,shared_model,shared_label)
    #array_fft_model_score = np.zeros((3,5))
    #mfu,mfc = np.unique(mode_fft,return_counts=True)
    #array_fft_model_score[0][mfu] = mfc
    #mu,mc = np.unique(modes,return_counts=True)
    #array_fft_model_score[1][mu] = mc
    #array_fft_model_score[2] = score.sum(axis=0)

    #return (t,array_fft_model_score)
    return (t,fil)

def convert_time(time_str):
    hr_min_sec,ms_tif = time_str.split('_')[-2:]
    ms = ms_tif.split('.')[0]
    hr = int(hr_min_sec[:2])
    minute = int(hr_min_sec[2:4])
    sec = int(hr_min_sec[4:6])
    total = float(f'{(hr*3600)+(minute*60)+sec}.{ms}')
    return total

# initialize worker processes
#def init_worker(thresh,folder,dictionary,model,labels):
def init_worker(rad,xor,yor,folder,dictionary):
    # declare scope of a new global variable

    global shared_rad
    global shared_xor
    global shared_yor

    global shared_dict
    global shared_folder
    #global shared_model
    #global shared_label

    # store argument in the global variable for this process
    shared_rad = rad
    shared_xor = xor
    shared_yor = yor

    shared_dict = dictionary
    shared_folder = folder
    #shared_model = model
    #shared_label = labels

class Timestep_MPI():
    def __init__(self,folder='',step=1,rad=None,xor=None,yor=None,model_file=[],label_file=[]):

        self.radius = rad
        self.x_origin = xor
        self.y_origin = yor

        self.folder = folder
        self.step = step
        self.model_file = model_file
        self.label_file = label_file

        frames = [os.path.basename(x) for x in glob(f'{folder}/*.tif')]
        #doesnt start at zero
        self.time = np.arange(len(frames))
        self.t2f = {t:f.split('.')[0] for t,f in enumerate(frames)}

        self.map = {f'{i}{j}':(i*4)+j for i in range(4) for j in range(4)}
        self.storage_fft_model_score = {t:np.zeros((2,len(self.map.keys()))) for t in self.time}
        self.storage = []
        if self.step:
            self.run_split(self.step)
        else:
            self.run_slow()
    
    def wave2dicindex(self,wave_list,factor):
        m1,m2 = heapq.nlargest(2,wave_list)
        if m1 < factor*m2:
            wave_lab = f'{wave_list.index(m1)}{wave_list.index(m2)}'
        else:
            wave_lab = f'{wave_list.index(m1)}{wave_list.index(m1)}'
        return self.map[wave_lab]

    def get_fil_number_data(self,t,n):
        if t not in self.storage:
            print(f'Not a valid time: {t}')
            return
        elif n not in self.storage[t]:
            return
        else:
            return self.storage[t][n]

    def run_split(self,step):
        if self.model_file:
            model = tf.keras.models.load_model(self.model_file)
            labels = np.arange(4)
        else:
            print('Doing Nothing')
            return
        time_split = np.array_split(self.time,range(step,len(self.time),step))
        print(f'Total Chunks: {len(time_split)}')
        time_start_mpi = time.time()
        for ci, t_chunk in enumerate(time_split):
        #for ci,t_chunk in enumerate(time_split[0:1]):
            print(f'Running MPI Chunk {ci+1} {t_chunk[0]} to {t_chunk[-1]} of {time_split[-1][-1]}')
            time_start_chunk = time.time()
            chunksize, extra = divmod(len(t_chunk),os.cpu_count()*4)
            if extra:
                chunksize += 1
            with Pool(os.cpu_count()-1,initializer=init_worker, initargs=(self.radius,self.x_origin,self.y_origin,self.folder,self.t2f)) as pool:
                all_fils = pool.map(Make_Fil, t_chunk,chunksize=chunksize)

            pool.close()
            pool.join()

            #print('Chunk '+Total_Time(time_start_chunk))
            #print('Done MPI Chunk')
            
            time_mode = time.time()
            for t,fil in all_fils:
                if not fil.fil_number:
                    print('-'*10+f'Empty {t}'+'-'*10)
                    continue
                scores_fft = [f[2] for f in fil.fil_char.values()]
                _,scores_model = Fil.predict_image_fil_char(fil.fil_char,model,labels)
                scores_fft_model = np.zeros((2,len(self.map.keys())))
                for s_fft,s_model in zip(scores_fft,scores_model):
                    scores_fft_model[0][self.wave2dicindex(list(s_fft),2)] += 1
                    scores_fft_model[1][self.wave2dicindex(list(s_model),9)] += 1
                self.storage_fft_model_score[t] = scores_fft_model

                #mfu,mfc = np.unique(mode_fft,return_counts=True)
                #array_fft_model_score[0][mfu] = mfc
                #mu,mc = np.unique(modes,return_counts=True)
                #array_fft_model_score[1][mu] = mc
                #array_fft_model_score[2] = score.sum(axis=0)
                #self.storage_fft_model_score[t] = array_fft_model_score
            

            """
            for t,fft_mode_score in all_fils_model:
                if t%10==0:
                    print(f'Frame {t} of {len(all_fils_model)}: {(t/len(all_fils_model)*100):0.1f}%')
                fft,mode,score = fft_mode_score
                self.storage_mode_fft[t] = fft
                self.storage_mode_model[t] = mode
                self.storage_mode_score[t] = score
            """
            #print('Mode '+Total_Time(time_mode))
            print('Chunk '+Total_Time(time_start_chunk))
            #remove to save memory
            #all_fils_model = None
            del all_fils
            gc.collect()
        self.storage_fft_model_score = np.array([*self.storage_fft_model_score.values()])
        #self.storage = np.vstack([*self.storage])
        #self.storage_mode_model = np.vstack([*self.storage_mode_model.values()])
        #self.storage_mode_score = np.vstack([*self.storage_mode_score.values()])
        #self.storage_mode_fft = np.vstack([*self.storage_mode_fft.values()])
        #np.save('mode_model_array',self.storage_mode_model)
        #np.save('mode_score_array',self.storage_mode_score)
        #np.save('mode_fft_array',self.storage_mode_fft)
        #print(self.storage_fft_model_score)
        np.save(f'MixedScores_fft_model_{self.folder}_thresh_triangle',self.storage_fft_model_score)
        np.save('MixedLabels',np.array([*list(self.map.keys())]))
        print('MPI Total '+Total_Time(time_start_mpi))
    
class Fil():
    def __init__(self,imdat=None,rad=None,xor=None,yor=None):
        self.imgData = self.blur_median(imdat)
        self.thresh = None
        self.radius = rad
        self.x_origin = xor
        self.y_origin = yor

        self.fil_char = {}
        self.num_ref = {}
        self.missing = {}

        #initialize filament location
        self.getLoc()
        #clear image for data
        self.imgData = None
        #del self.imgData
        #gc.collect()

    @staticmethod
    def blur_median(img,size=5):
        return cv2.medianBlur(img,size)
    @staticmethod
    def circle_cont(radius,x_origin,y_origin,n=200):
        cent = np.array([x_origin,y_origin]).reshape(2,1) 
        angle_array = np.linspace(0,2*np.pi,n) 
        rxy = cent+(radius*np.array([np.cos(angle_array),np.sin(angle_array)]))
        return rxy.reshape(2,n).astype(int)
    @staticmethod
    def distance_com(com1,com2):
        a = np.array(com1)
        b = np.array(com2)
        return np.sqrt(np.sum((a-b)**2,axis=0))
    @staticmethod
    def distance_cont(points,xcm,ycm):
        center = np.array([xcm,ycm])
        return np.sqrt(np.sum((points-center)**2,axis=(1,2)))
    
    def output_gray_center(self,select,cont):
        gray = self.imgData.T[*select]
        gs = gray/gray.max()
        scale = gs/gs.sum()
        com = (select*scale).sum(axis=1) 
        resize = np.ceil(self.distance_cont(cont,*com).max()*2).astype(int)
        fil_gray = np.zeros((resize,resize))
        shift = np.array((resize//2,resize//2))-com
        fil_gray.T[*(select+shift[...,None]).astype(int)] = gray
        return com, fil_gray
    
    def output_gray_tight(self,select):
        gray = self.imgData.T[*select]
        gs = gray/gray.max()
        scale = gs/gs.sum()
        com = (select*scale).sum(axis=1)  
        fil_shift = select-select.min(axis=1,keepdims=True)
        fil_gray = np.zeros(tuple(fil_shift.max(axis=1)+1))
        fil_gray[fil_shift[0],fil_shift[1]]=gray
        return com, fil_gray.T

    @staticmethod
    def predict_image_fil_char(fil_char,model,label,size=180):

        image_batch = tf.stack([tf.image.resize_with_pad(f[0].astype(float)[...,None],size,size) for f in fil_char.values()])
        image_batch *= 255/np.max(image_batch,axis=(1,2,3)).reshape(-1,1,1,1)
        predictions = model(image_batch)
        score = tf.nn.softmax(predictions).numpy()
        return np.array(label)[np.argmax(score,axis=1)], score
    @staticmethod
    def predict_image_single(img,model,label,size=180):

        image_batch = tf.image.resize_with_pad(img.astype(float)[...,None],size,size)[None,...]
        image_batch *= 255/np.max(image_batch,axis=(1,2,3)).reshape(-1,1,1,1)
        predictions = model(image_batch)
        score = tf.nn.softmax(predictions).numpy()
        return np.array(label)[np.argmax(score,axis=1)], score
   
    @staticmethod
    def get_simple(dictionary,index=0):
        return {k:dictionary[k][index] for k in dictionary}
    def output_mode_fft(self,c,com,scale=0.8):

        con_area = cv2.contourArea(c)
        _,rad = cv2.minEnclosingCircle(c)
        cir_area = np.pi*(rad**2)
        area_dif = np.round(((cir_area-con_area)/cir_area),2)
        if area_dif < .3:
            wave = np.zeros(4)
            wave[0] = 1
            return wave

        radius = self.distance_cont(c,*com).max()
        c_cont_graph = self.circle_cont(scale*radius,*com,200)
        #(center_x,center_y),radius = cv2.minEnclosingCircle(c)
        #c_cont_graph = self.circle_cont(scale*radius,center_x,center_y,200)

        c_cont_graph = c_cont_graph[:,c_cont_graph[0]<self.imgData.shape[1]]
        c_cont_graph = c_cont_graph[:,c_cont_graph[1]<self.imgData.shape[0]]
        inten = self.imgData.T[*c_cont_graph]
        inten_shift = inten-inten.mean()
        inten_shift = inten_shift/inten_shift.max()
        wave = abs(np.fft.rfft(inten_shift))[:4]
        return wave

    def getLoc(self):
        img_8bit = (self.imgData.copy()/self.imgData.max()*255).astype('uint8')
        if not self.thresh:
            self.thresh , im_bw = cv2.threshold(img_8bit, None, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        else:
            _ , im_bw = cv2.threshold(img_8bit, self.thresh, 255, cv2.THRESH_BINARY)
        contours, _ =  cv2.findContours(im_bw.astype('uint8').copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        c_area_roi = [c for c in contours if np.all(self.distance_cont(np.array(c),self.x_origin,self.y_origin) < self.radius) and cv2.contourArea(c) > 200]

        index=0
        for i,c in enumerate(c_area_roi):
            #this works for filament count less than 255 
            inten = i+1
            _ = cv2.drawContours(im_bw,c_area_roi,i,inten,thickness=-1)
            select = np.asarray(np.where(im_bw.T==inten))
            #com, fil_gray = self.output_gray_center(select,c)
            com, fil_gray = self.output_gray_tight(select)
            wave_fft = self.output_mode_fft(c,com,scale=0.8)
            self.fil_char[tuple(com)] = [fil_gray,c,wave_fft]
            self.num_ref[index+1] = [tuple(com),c]
            index+=1
        #fil number is based on number at timestep
        self.fil_number = len(self.fil_char.keys())



def Total_Time(time_start):
    return f'Total time: {int((time.time()-time_start)//3600)} hrs {int((time.time()-time_start)%3600//60)} min {(time.time()-time_start)%3600%60:.2f} sec'


if __name__ == "__main__":
    #command argv 
    #example: python mpi_track_thresh.py 0 ./ 500 model_file_path label_file_path
    # ['C:/Users/jtavr/Desktop/Fil_Training/model_t0b0_no_outliers.keras', 'C:/Users/jtavr/Desktop/Fil_Training/labels_model.txt']

    step = int(sys.argv[1])
    model_file = sys.argv[2]

    if sys.argv[3] == 'dir':
        folders = glob('*mT')
        rad = int(sys.argv[4])
        xor = int(sys.argv[5])
        yor = int(sys.argv[6])
        print('Running Multiple Directories')
        for f in folders:
            print(f)
            time_start = time.time()
            tt = Timestep_MPI(folder=f,step=step,rad=rad,xor=xor,yor=yor,model_file=model_file)
            print(Total_Time(time_start))
    else:
        folder = sys.argv[3]
        time_start = time.time()
        rad = int(sys.argv[4])
        xor = int(sys.argv[5])
        yor = int(sys.argv[6])
        tt = Timestep_MPI(folder=folder,step=step,rad=rad,xor=xor,yor=yor,model_file=model_file)
        
        print(Total_Time(time_start))




#python mpi_compare_triangle_roi.py 500 C:/Users/jtavr/Desktop/Fil_Training/Training_BG0.5/model_t0b1_stretch_03_triangle.keras . 450 856 733
