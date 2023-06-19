#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import numba 
import math
from numba import cuda


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~~                   NUMBA                     ~~
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Simple function to evaluate the acess time of a matrix in CUDA
@numba.cuda.jit
def Cuda_time(mat,out):
    
    pos_x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    pos_y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
             
    if (pos_x<mat.shape[0] and pos_y<mat.shape[1]):       
        out[pos_x,pos_y]=1

##~~~~~~~~~~~~~~~~~~~~~Sum~~~~~~~~~~~~~~~~~~~~~~~~~
#Simple sum function of two vectors at different
#optimization estrategies

@numba.vectorize(['int16(int16,int16)'], target='cuda')
def V_Sum(x,y):
    return x+y

@numba.guvectorize(['void(int16[:],int16[:],int16[:])'], '(n),(n)->(n)',nopython=True, target='cuda')
def GV_Sum(x,y,resp):
    for i in range(x.shape[0]):
        resp[i] = x[i] + y[i]

#Original approach, usually faster but requires manual block per grid calculation
@numba.cuda.jit
def Cuda_Soma(x,y,resp):
    tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
    ty = cuda.blockIdx.x  # Similarly, this is the unique block ID within the 1D grid

    block_size = cuda.blockDim.x  # number of threads per block
    grid_size = cuda.gridDim.x    # number of blocks in the grid
    
    start = tx + ty * block_size
    
    if start < x.shape[0]:
        resp[start] = x[start] + y[start]

#Alternative approach, accepts diferent input dimensions
@numba.cuda.jit
def Cuda_alt_Sum(x,y,resp):
    tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
    ty = cuda.blockIdx.x  # Similarly, this is the unique block ID within the 1D grid

    block_size = cuda.blockDim.x  # number of threads per block
    grid_size = cuda.gridDim.x    # number of blocks in the grid
    
    start = tx + ty * block_size
    stride = block_size * grid_size

    #python simplification
    #start = cuda.grid(1)      # 1 = one dimensional thread grid, returns a single value
    #stride = cuda.gridsize(1)

    #Assuming x and y inputs are same length
    for i in range(start, x.shape[0], stride):
        resp[i] = x[i] + y[i]

##~~~~~~~~~~~~~~~~~~~ImageMax~~~~~~~~~~~~~~~~~~~~~~~
@numba.guvectorize(['void(uint8[:,:,:],uint8[:])'], '(w,h,ch)->(ch)',target='cpu')
def GV_ImageMax(mat,out):    
    out[0]=np.max(mat[:,:,0])
    out[1]=np.max(mat[:,:,1]) 
    out[2]=np.max(mat[:,:,2]) 

@numba.guvectorize(['void(uint8[:,:,:],uint8[:])'], '(w,h,ch)->(ch)',target='parallel')
def GVp_ImageMax(mat,out):    
    out[0]=np.max(mat[:,:,0])
    out[1]=np.max(mat[:,:,1]) 
    out[2]=np.max(mat[:,:,2]) 

##~~~~~~~~~~~~~~ImageMaxPercentile~~~~~~~~~~~~~~~~~~
@numba.guvectorize(['void(uint8[:,:,:],uint8,uint8[:])'], '(w,h,ch),()->(ch)', target='cpu')
def GV_ImageMaxPercentile(mat,p,out):    
    out[0]=np.percentile(mat[:,:,0],p)
    out[1]=np.percentile(mat[:,:,1],p) 
    out[2]=np.percentile(mat[:,:,2],p)


##~~~~~~~~~~~~~~~Contancia de cor~~~~~~~~~~~~~~~~~~~~
@numba.vectorize(['uint16(uint8,uint8)'], target='cpu')
def V_ColorConstancyAdjust(mat,adjust):     
    out=(1.0*mat/adjust)*255
    return (out)

@numba.vectorize(['uint16(uint8,uint8)'], target='parallel')
def Vp_ColorConstancyAdjust(mat,adjust):     
    out=(1.0*mat/adjust)*255
    return (out)

@numba.vectorize(['uint16(uint8,uint8)'], target='cuda')
def Vc_ColorConstancyAdjust(mat,adjust):     
    out=(1.0*mat/adjust)*255
    return (out)

@numba.guvectorize(['void(uint8[:,:,:],float32[:],uint16[:,:,:])'], '(w,h,ch),(ch)->(w,h,ch)', target='cpu' )
def GV_ColorConstancyAdjust(mat,adjust,out):     
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            for k in range(mat.shape[2]):
                out[i,j,k]=mat[i,j,k]*adjust[k]

@numba.guvectorize(['void(uint8[:,:,:],float32[:],uint16[:,:,:])'], '(w,h,ch),(ch)->(w,h,ch)', target='parallel' )
def GVp_ColorConstancyAdjust(mat,adjust,out):     
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            for k in range(mat.shape[2]):
                out[i,j,k]=mat[i,j,k]*adjust[k]

@numba.cuda.jit
def Cuda_ColorConstancyAdjust(mat,adjust,out):    
    pos_x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    pos_y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y   
             
    if (pos_x<mat.shape[0] and pos_y<mat.shape[1]):
        out[pos_x,pos_y,0]=mat[pos_x,pos_y,0]*adjust[0]
        out[pos_x,pos_y,1]=mat[pos_x,pos_y,1]*adjust[1]
        out[pos_x,pos_y,2]=mat[pos_x,pos_y,2]*adjust[2]

@numba.cuda.jit
def Cuda3d_ColorConstancyAdjust(mat,adjust,out):    
    pos_x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    pos_y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    pos_z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
             
    if (pos_x<mat.shape[0] and pos_y<mat.shape[1] and pos_z<mat.shape[2]):
        out[pos_x,pos_y,pos_z]=mat[pos_x,pos_y,pos_z]*adjust[pos_z]

##Alternative approach involves an integer input, but showed a slower performance
#@numba.guvectorize(['void(uint8[:,:,:],uint8[:],uint16[:,:,:])'], '(w,h,ch),(ch)->(w,h,ch)', target='parallel' )
#def GVp_alt_ColorConstancyAdjust(mat,adjust,out):     
#    for i in range(mat.shape[0]):
#        for j in range(mat.shape[1]):
#            for k in range(mat.shape[2]):
#                out[i,j,k]=(1.0*mat[i,j,k]/adjust[k])*255.0

##CUDA with guvectorize showed impractical processing time
#@numba.guvectorize(['void(uint8[:,:,:],float32[:],uint16[:,:,:])'], '(w,h,ch),(ch)->(w,h,ch)', target='cuda' )
#def GVc_ColorConstancyAdjust(mat,adjust,out):     
#    for i in range(mat.shape[0]):
#        for j in range(mat.shape[1]):
#            for k in range(mat.shape[2]): 
#                out[i,j,k]=mat[i,j,k]*adjust[k]

##~~~~~~~~~~~~~~~~~~~~~~~ExG~~~~~~~~~~~~~~~~~~~~~
@numba.vectorize(['uint16(uint8,uint8,uint8)'],target='cpu')
def V_Transformation_ExG(R,G,B):
    out = (255*2)+(2*G-R-B)   
    return out

@numba.vectorize(['uint16(uint8,uint8,uint8)'], target='parallel')
def Vp_Transformation_ExG(R,G,B):
    out = (255*2)+(2*G-R-B)   
    return out

@numba.vectorize(['uint16(uint8,uint8,uint8)'], target='cuda')
def Vc_Transformation_ExG(R,G,B):
    out = (255*2)+(2*G-R-B)   
    return out

##Direct approach (slightly slower execution)
#@numba.guvectorize(['void(uint8[:,:,:],uint16[:,:])'], '(w,h,ch)->(w,h)')
#def GV_Transformation_ExG(mat,out):  
#    for i in range(mat.shape[0]):
#        for j in range(mat.shape[1]):
#            out[i,j]=255*2+(2*mat[i,j,1]-mat[i,j,2]-mat[i,j,0])  


#change target according to necessity (cpu,parallel,cuda)
@numba.guvectorize(['void(uint8[:,:,:],uint16[:,:])'], '(w,h,ch)->(w,h)',target='cpu')
def GV_Transformation_ExG(mat,out):    
    R=mat[:,:,2]
    G=mat[:,:,1]
    B=mat[:,:,0]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            out[i,j]=255*2+(2*G[i,j]-R[i,j]-B[i,j]) 

#change target according to necessity (cpu,parallel,cuda)
@numba.guvectorize(['void(uint8[:,:,:],uint16[:,:])'], '(w,h,ch)->(w,h)',target='cpu')
def GV_alt_Transformation_ExG(mat,out):    
    R=mat[:,:,2]
    G=mat[:,:,1]
    B=mat[:,:,0]   
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):            
            out[i,j]=255*2+(2*G[i,j]-R[i,j]-B[i,j])
            if out[i,j] < 510:
                out[i,j]=510
            if out[i,j] > 765:
                out[i,j]=765
            out[i,j]=out[i,j]-510


@numba.guvectorize(['void(uint8[:,:,:],uint16[:,:])'], '(w,h,ch)->(w,h)',target='parallel')
def GVp_alt_Transformation_ExG(mat,out):    
    R=mat[:,:,2]
    G=mat[:,:,1]
    B=mat[:,:,0]   
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):            
            out[i,j]=255*2+(2*G[i,j]-R[i,j]-B[i,j])
            if out[i,j] < 510:
                out[i,j]=510
            if out[i,j] > 765:
                out[i,j]=765
            out[i,j]=out[i,j]-510

@numba.cuda.jit
def Cuda_Transformation_ExG(mat,out):
    
    pos_x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    pos_y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y  
    #pos_x, pos_y = cuda.grid(2)   
        
    if (pos_x<mat.shape[0] and pos_y<mat.shape[1]):
        out[pos_x,pos_y]=255*2+(2*mat[pos_x,pos_y,1]-mat[pos_x,pos_y,2]-mat[pos_x,pos_y,0])  

@numba.cuda.jit 
def Cuda_alt_Transformation_ExG(mat,out):
    
    pos_x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    pos_y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y  
    #pos_x, pos_y = cuda.grid(2)   
        
    if (pos_x<mat.shape[0] and pos_y<mat.shape[1]):
        out[pos_x,pos_y]=255*2+(2*mat[pos_x,pos_y,1]-mat[pos_x,pos_y,2]-mat[pos_x,pos_y,0])
        if out[pos_x,pos_y] < 510:
            out[pos_x,pos_y]=510
        if out[pos_x,pos_y] > 765:
            out[pos_x,pos_y]=765
        out[pos_x,pos_y]=out[pos_x,pos_y]-510


##Alternative approach, accepts diferent input dimensions
#@numba.cuda.jit
#def Cuda_Transformation_ExG(mat,out):
    
#    pos_x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#    pos_y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y  
#    #pos_x, pos_y = cuda.grid(2)     
 
#    stride_x = cuda.blockDim.x * cuda.gridDim.x
#    stride_y = cuda.blockDim.y * cuda.gridDim.y

#    for i in range(pos_x, mat.shape[0], stride_x):
#        for j in range(pos_y, mat.shape[1], stride_y):
#            out[i,j]=255*2+(2*mat[i,j,1]-mat[i,j,2]-mat[i,j,0])
        
      
  
##~~~~~~~~~~~~~~~~~~~~~~~FMI~~~~~~~~~~~~~~~~~~~~~
@numba.vectorize(['int16(uint8,uint8,uint8)'])
def V_Transformation_FMI(R,G,B):
    FMg= 255*(2*G-R-B)/(2*G + R + B + 0.01)    
    FMcomb= -255*(2*(B*B-3*G*R+B*R+B*G)/((B+2*G+R)*(B+G+2*R)+0.01))
    #FMI=(FMg/np.max(FMg))+(FMcomb/np.max(FMcomb))
    out=FMg+FMcomb
    return out

@numba.vectorize(['int16(uint8,uint8,uint8)'], target='cuda')
def Vc_Transformation_FMI(R,G,B):
    FMg= 255*(2*G-R-B)/(2*G + R + B + 0.01)    
    FMcomb= -255*(2*(B*B-3*G*R+B*R+B*G)/((B+2*G+R)*(B+G+2*R)+0.01))
    #FMI=(FMg/np.max(FMg))+(FMcomb/np.max(FMcomb))
    out=FMg+FMcomb
    return out

@numba.vectorize(['int16(uint8,uint8,uint8)'], target='parallel')
def Vp_Transformation_FMI(R,G,B):     
    FMg= 255*(2*G-R-B)/(2*G + R + B + 0.01)    
    FMcomb= -255*(2*(B*B-3*G*R+B*R+B*G)/((B+2*G+R)*(B+G+2*R)+0.01))
    #FMI=(FMg/np.max(FMg))+(FMcomb/np.max(FMcomb))
    out=FMg+FMcomb
    return out

#change target according to necessity (cpu,parallel,cuda)
@numba.guvectorize(['void(uint8[:,:,:],int16[:,:])'], '(w,h,ch)->(w,h)',target='cpu')
def GV_Transformation_FMI(mat,out):    
    R=mat[:,:,2]
    G=mat[:,:,1]
    B=mat[:,:,0]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            out[i,j]=255*(2*G[i,j]-R[i,j]-B[i,j])/(2*G[i,j] + R[i,j] + B[i,j] + 0.01) + (-255*(2*(R[i,j]*B[i,j]-3*G[i,j]*R[i,j]+B[i,j]*R[i,j]+B[i,j]*G[i,j])/((B[i,j]+2*G[i,j]+R[i,j])*(B[i,j]+G[i,j]+2*R[i,j])+0.01)))
           
#change target according to necessity (cpu,parallel,cuda)
@numba.guvectorize(['void(uint8[:,:,:],int16[:,:])'], '(w,h,ch)->(w,h)',target='cpu')
def GV_alt_Transformation_FMI(mat,out):    
    R=mat[:,:,2]
    G=mat[:,:,1]
    B=mat[:,:,0]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            out[i,j]=255*(2*G[i,j]-R[i,j]-B[i,j])/(2*G[i,j] + R[i,j] + B[i,j] + 0.01) + (-255*(2*(R[i,j]*B[i,j]-3*G[i,j]*R[i,j]+B[i,j]*R[i,j]+B[i,j]*G[i,j])/((B[i,j]+2*G[i,j]+R[i,j])*(B[i,j]+G[i,j]+2*R[i,j])+0.01)))
            if out [i,j] < 0:
                out [i,j] = 0
            if out [i,j] > 255:
                out [i,j] = 255 

#change target according to necessity (cpu,parallel,cuda)
@numba.guvectorize(['void(uint8[:,:,:],int16[:,:])'], '(w,h,ch)->(w,h)',target='parallel')
def GVp_alt_Transformation_FMI(mat,out):    
    R=mat[:,:,2]
    G=mat[:,:,1]
    B=mat[:,:,0]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            out[i,j]=255*(2*G[i,j]-R[i,j]-B[i,j])/(2*G[i,j] + R[i,j] + B[i,j] + 0.01) + (-255*(2*(R[i,j]*B[i,j]-3*G[i,j]*R[i,j]+B[i,j]*R[i,j]+B[i,j]*G[i,j])/((B[i,j]+2*G[i,j]+R[i,j])*(B[i,j]+G[i,j]+2*R[i,j])+0.01)))
            if out [i,j] < 0:
                out [i,j] = 0
            if out [i,j] > 255:
                out [i,j] = 255 

#Direct Method
@numba.cuda.jit
def Cuda_Transformation_FMI(mat,out):    
    pos_x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    pos_y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
             
    if (pos_x<mat.shape[0] and pos_y<mat.shape[1]):       
        #FMg[pos_x,pos_y]= (2*mat[pos_x,pos_y,1]-[pos_x,pos_y,2]-mat[pos_x,pos_y,0])/(2*mat[pos_x,pos_y,1] + [pos_x,pos_y,2] + mat[pos_x,pos_y,0] + 0.01) 
        #FMcomb[pos_x,pos_y]= -255*(2*(mat[pos_x,pos_y,0]*mat[pos_x,pos_y,0]-3*mat[pos_x,pos_y,1]*[pos_x,pos_y,2]+mat[pos_x,pos_y,0]*[pos_x,pos_y,2]+mat[pos_x,pos_y,0]*mat[pos_x,pos_y,1])/((mat[pos_x,pos_y,0]+2*mat[pos_x,pos_y,1]+[pos_x,pos_y,2])*(mat[pos_x,pos_y,0]+mat[pos_x,pos_y,1]+2*[pos_x,pos_y,2])+0.01))
        #FMI=(FMg/np.max(FMg))+(FMcomb/np.max(FMcomb))
        out[pos_x,pos_y]=255*(2*mat[pos_x,pos_y,1]-mat[pos_x,pos_y,2]-mat[pos_x,pos_y,0])/(2*mat[pos_x,pos_y,1] + mat[pos_x,pos_y,2] + mat[pos_x,pos_y,0] + 0.01) + (-255*(2*(mat[pos_x,pos_y,0]*mat[pos_x,pos_y,0]-3*mat[pos_x,pos_y,1]*mat[pos_x,pos_y,2]+mat[pos_x,pos_y,0]*mat[pos_x,pos_y,2]+mat[pos_x,pos_y,0]*mat[pos_x,pos_y,1])/((mat[pos_x,pos_y,0]+2*mat[pos_x,pos_y,1]+mat[pos_x,pos_y,2])*(mat[pos_x,pos_y,0]+mat[pos_x,pos_y,1]+2*mat[pos_x,pos_y,2])+0.01)))

#Alternative method creating auxiliary R,G,B matrices (small improvement in performance)
#Aditional evaluation of interval within function with major performance improvement
@numba.cuda.jit
def Cuda_alt_Transformation_FMI(mat,out):    
    pos_x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    pos_y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    
    R=mat[:,:,2]
    G=mat[:,:,1]
    B=mat[:,:,0]

    if (pos_x<mat.shape[0] and pos_y<mat.shape[1]):
        out[pos_x,pos_y]=255*(2*G[pos_x,pos_y]-R[pos_x,pos_y]-B[pos_x,pos_y])/(2*G[pos_x,pos_y] + R[pos_x,pos_y] + B[pos_x,pos_y] + 0.01) + (-255*(2*(R[pos_x,pos_y]*B[pos_x,pos_y]-3*G[pos_x,pos_y]*R[pos_x,pos_y]+B[pos_x,pos_y]*R[pos_x,pos_y]+B[pos_x,pos_y]*G[pos_x,pos_y])/((B[pos_x,pos_y]+2*G[pos_x,pos_y]+R[pos_x,pos_y])*(B[pos_x,pos_y]+G[pos_x,pos_y]+2*R[pos_x,pos_y])+0.01)))
        if out [pos_x,pos_y] < 0:
            out [pos_x,pos_y] = 0
        if out [pos_x,pos_y] > 255:
            out [pos_x,pos_y] = 255
    


##~~~~~~~~~~~~~~~~~~FIM NUMBA~~~~~~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##~~~~~~~~~~~~~~SetCameraParameters~~~~~~~~~~~~~~~
def SetCameraParameters(cap):
    #Parameters values
    gain=70
    exposure=240
    temperature=4000

    #Capture Dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv2.CAP_PROP_FPS,60)
    
    #Open configuration window
    #cap.set(cv2.CAP_PROP_SETTINGS,0) 

    #Comands to set the parameters in OpenCV
    #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25) #0.25 means "manual exposure"
    #cap.set(cv2.CAP_PROP_EXPOSURE,10) #set value to enable manual exposure mode
    #cap.set(cv2.CAP_PROP_GAIN,50)
    #cap.set(cv2.CAP_PROP_HUE,15)
    
    #Comands to set the parameters in Terminal
    #os.popen("v4l2-ctl --set-ctrl=white_balance_temperature_auto=0") #To turn back on set value to 1
    #os.popen("v4l2-ctl --set-ctrl=white_balance_temperature={}".format(temperature)) #Default = 4000		
    #os.popen("v4l2-ctl --set-ctrl=backlight_compensation=0") #To turn back on set value to 1
    #os.popen("v4l2-ctl --set-ctrl=exposure_auto=1") #To turn back on set value to 3 (off=1)
    #os.popen("v4l2-ctl --set-ctrl=exposure_absolute={}".format(exposure)) #Adjust gain valure 3~2047
    #os.popen("v4l2-ctl --set-ctrl=gain={}".format(gain)) #Adjust gain valure 0~255
    print ("->Camera configurada")


##~~~~~~~~~~~~~~~~~~~ImageMax~~~~~~~~~~~~~~~~~~~~~
def ImageMax(mat):
    #mat=np.array(frame,np.uint8)
    
    maxB=np.max(mat[:,:,0])
    maxG=np.max(mat[:,:,1])
    maxR=np.max(mat[:,:,2])
    return (maxB,maxG,maxR)
    #Metodo se mostrou mais demorado 
    #return np.max(mat,(0,1))

##~~~~~~~~~~~~~~ImageMaxPercentile~~~~~~~~~~~~~~~~~~
def ImageMaxPercentile(mat,p):
        
    maxB=np.percentile(mat[:,:,0],p)
    maxG=np.percentile(mat[:,:,1],p)
    maxR=np.percentile(mat[:,:,2],p)
    return (maxB,maxG,maxR)

##~~~~~~~~~~~~~~Pré-Processamento~~~~~~~~~~~~~~~~~~
def ColorConstancyAdjust(mat,adjust):         
    out=np.array((mat*adjust).clip(min=0,max=255),np.uint8)
    return (out)

##~~~~~~~~~~~~~~~~~~Transformation ExG~~~~~~~~~~~~~~~~~~~
def Transformation_ExG(img):
      
    mat=np.array(img,np.uint16)
    mat= (255*2)+(2*mat[:,:,1]-mat[:,:,0]-mat[:,:,2])
    
    ExG=mat.clip(min=(255*2),max=255*3)-(255*2)
    #ExG=mat.clip(min=(255*2),max=np.max(mat))-(255*2)                
    out = np.uint8(ExG)
    return out

#@cuda.jit    
##~~~~~~~~~~~~~~~~~~Transformation FMI~~~~~~~~~~~~~~~~~~~
def Transformation_FMI(img):
      
    mat=np.array(img,np.float32)
    R=mat[:,:,2]/255
    G=mat[:,:,1]/255
    B=mat[:,:,0]/255
                    
    FMg= (2*G-R-B)/(2*G + R + B + 0.01)
    #FMg=FMg.clip(min=0,max=1) 
    FMcomb= -1*(2*(B*B-3*G*R+B*R+B*G)/((B+2*G+R)*(B+G+2*R)+0.01))
    FMI=(FMg/np.max(FMg))+(FMcomb/np.max(FMcomb))
     
    mat=FMI
    
    ExG=255*mat.clip(min=0,max=1)             
    out = np.uint8(ExG)
    
    return out

##~~~~~~~~~~~~~~~~~~Transformation l3~~~~~~~~~~~~~~~~~~~
def Transformation_l3(img):
      
    mat=np.array(img,np.float32)
    R=mat[:,:,2]/255
    G=mat[:,:,1]/255
    B=mat[:,:,0]/255
        
    l3 = (G-B)*(G-B)/((R-G)*(R-G) + (R-B)*(R-B) + (G-B)*(G-B) + 0.01)
    mat=l3        

    ExG=255*mat.clip(min=0,max=1)             
    out = np.uint8(ExG)
    
    return out

##~~~~~~~~~~~~~~~~~~Classification~~~~~~~~~~~~~~~~~~~
def Classification(img,th):
    
    if (th>=0):
        ret,out=cv2.threshold(img,th*255,255,cv2.THRESH_BINARY)
    else:
        ret,out=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)         
    
    return (out)

##~~~~~~~~~~~~~~~~~~Post Processing~~~~~~~~~~~~~~~~~~~
def PostProcessing(img,ksize):
    
    if (ksize>=1):
         kernel = np.ones((ksize,ksize),np.uint8)
         outAux = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
         out = cv2.morphologyEx(outAux, cv2.MORPH_CLOSE, kernel)
    else:
         out=img         
    
    return (out)

##~~~~~~~~~~~~~~~RelevantContorns~~~~~~~~~~~~~~~~~~~
def RelevantContorns(img,bin,maxObj,areaMin):
        
    #Find the binary image contourns
    _,contours, _ = cv2.findContours(bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #Create a structured array for the object propreties
    dtype = [('ID', int), ('area', int), ('perimeter', int),('Cx', int),('Cy', int)]
    values=np.ones((maxObj), dtype=int)*-1
    objects=np.array(values, dtype=dtype) 
    
    n=0
    cv2.drawContours(img,contours,-1,(0,0,255),3)
        
    for obj in range (0,len(contours)):
        #Verify the criteria for the selection of the objects
        if (cv2.contourArea(contours[obj])>areaMin and n<maxObj):
            #Temporary records the object moments in M
            M=cv2.moments(contours[obj])
            #Record the object ID along with the selected propreties
            objects[n]=(obj,cv2.contourArea(contours[obj]),cv2.arcLength(contours[obj],1),int(M['m10']/M['m00']),int(M['m01']/M['m00']))
            #Draw the relevant contourns
            cv2.drawContours(img,[contours[obj]],-1,(0,255,0),3) 
            #increment the relevant object counter
            n=n+1
      
    #Sort the relevant results in respect to the contourn area
    objects=np.sort(objects, order='area')[::-1]

    return (img,objects)
 
##~~~~~~~~~~~~~~~~~~~Biggest Blob~~~~~~~~~~~~~~~~~~~~~
def BiggestBlob(keypoints):
    aux=0
    ObjSize=0
    center=[0,0]
    for k in keypoints:
        if(k.size>aux):
            ObjSize=k.size
            aux=k.size
            center=k.pt
    return (ObjSize,center)

##~~~~~~~~~~~~~~~~~~~Cross~~~~~~~~~~~~~~~~~~~~~
def Cross(img,Cx,Cy,size,color):
    lineSize=int(size*10)
    cv2.line(img, (int(Cx)-lineSize, int(Cy)), (int(Cx)+lineSize, int(Cy)), color, size)
    cv2.line(img, (int(Cx), int(Cy)-lineSize), (int(Cx), int(Cy)+lineSize), color, size)

##~~~~~~~~~~~~~~~~~~~Imprime FPS~~~~~~~~~~~~~~~~~~
timeI=[time.time()]
i=[0]

def Fps(frameInterval):
    i[0]=i[0]+1
    if(i[0]==frameInterval):
        global timeI
        interval=time.time()-timeI[0]
        fps=frameInterval/interval
        print("FPS:", fps)
        i[0]=0 
        timeI=[time.time()]

 ##~~~~~~~~~~~~~~~~~~~CamShiftMod~~~~~~~~~~~~~~~~~~
def CamShiftMod(imgProb,track_window,hMin,response,inertia):

    hLimite=10
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1 )
    
    for epoca in range(0,20):
        #MeanShift
        ret_mycs, track_window = cv2.meanShift(imgProb, track_window, term_crit)        

        # ROI
        (cx,cy,hx,hy) = track_window       
        ROI= imgProb[cy:cy+hy, cx:cx+hx]        
        # Moment Calculation
        M00= np.sum(ROI)/255 
        
        # Register previous window values
        hxa=hx
        hya=hy      

        # update windows values
        hx=int((2.828*M00*response + inertia*hxa)/(1+inertia))
        hy=int((2.828*M00*response + inertia*hya)/(1+inertia))        
        
        # if new values are too small replace to minimum values
        if hx < hMin or hy < hMin:
            hx=hMin
            hy=hMin

        # Change the windows parameters
        track_window = (cx,cy,hx,hy) 
        
        # If there is no signficant change in the window terminates the loop
        if abs(hxa-hx) < hLimite or abs(hya-hy) < hLimite:
            break

    return (track_window,M00)

 ##~~~~~~~~~~~~~~~~~~~DrawCamShift~~~~~~~~~~~~~~~~~~
def DrawCamShift(img,track_window,color):
    (cxd,cyd,wd,dd) = track_window
    cv2.rectangle(img,(cxd,cyd),(cxd+wd,cyd+dd),color,2)
    Cross(img,cxd+wd/2,cyd+dd/2,2,color)
