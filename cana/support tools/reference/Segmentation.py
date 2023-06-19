import numpy as np
import cv2
import glob
import os
import errno
import time

#Command to create path, ignore if already exists
def create_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

#----------Espaços de Cor--------------
#1 - HSV
def Segmentation_1(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #h,s,v = cv2.split(img) #Avoid Split due to relative elevaded computational cost
    h=img[:,:,0]

    #ret,img=cv2.threshold(h,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bimg=cv2.inRange(h,30,85)    
    
    return (img,bimg)

#2 - HSV with Otsu
def Segmentation_2(img):    
    
    #Convertion
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    #Convert to Numpy array
    h_mat = np.array(img[:,:,0])
    img=h_mat
                    
    ret,bimg=cv2.threshold(h_mat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    
    return (img,bimg)

#3 - Lab b* - a*
def Segmentation_3(img):    
    
    #Convertion
    img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
                               
    #Convert to Numpy array
    BmA_mat = 255+np.array(img[:,:,2],np.uint16)-np.array(img[:,:,1],np.uint16)               

    #Clip the interval
    BmA_mat=BmA_mat.clip(min=255,max=255*2)-255
    BmA_mat = np.uint8(BmA_mat)
    img=BmA_mat 
    
    ret,bimg=cv2.threshold(BmA_mat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    return (img,bimg)

#4 - Luv v* - u*
def Segmentation_4(img):    
    
    #Convertion
    img = cv2.cvtColor(img,cv2.COLOR_BGR2Luv)
                               
    #Convert to Numpy array
    VmU_mat = 255+np.array(img[:,:,2],np.uint16)-np.array(img[:,:,1],np.uint16)               

    #Clip the interval
    VmU_mat=VmU_mat.clip(min=255,max=255*2)-255
    VmU_mat = np.uint8(VmU_mat)
    img=VmU_mat 
    
    ret,bimg=cv2.threshold(VmU_mat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    return (img,bimg)

#5 - YCrCb - Cr + Cb
def Segmentation_5(img):    
    
    #Convertion
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
 
    #Convert to Numpy array
    CrpCb_mat = np.array(img[:,:,1],np.uint16)+np.array(img[:,:,2],np.uint16)              

    #Clip the interval
    CrpCb_mat=CrpCb_mat.clip(max=255)
    CrpCb_mat = np.uint8(CrpCb_mat)
    img=CrpCb_mat                        
        
    ret,bimg=cv2.threshold(CrpCb_mat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bimg=cv2.bitwise_not(bimg)
        
    return (img,bimg)

#6 - l1l2l3
def Segmentation_6(img):    
    
    #Convert to Numpy array
    R=np.array(img[:,:,2],np.float)/255
    G=np.array(img[:,:,1],np.float)/255
    B=np.array(img[:,:,0],np.float)/255
    
    l3_mat = ((G-B)*(G-B)/((R-G)*(R-G)+(R-B)*(R-B)+(G-B)*(G-B)+0.01))*255             

    l3_mat = np.uint8(l3_mat)
    img=l3_mat                        
        
    ret,bimg=cv2.threshold(l3_mat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   
        
    return (img,bimg)
        
#7 - CrCgCb -Cg
def Segmentation_7(img):    
    
    #Convertion
    imgYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
 
    #Convert to Numpy array
    Cg = 255+1.211*(np.array(img[:,:,1],np.uint16)-np.array(imgYCrCb[:,:,0],np.uint16))+128             
   
    #Dont figure out yet, but Clip the interval is not to be needed in this case
    #Cg=Cg.clip(min=255,max=255*2)-255       
    Cg = np.uint8(Cg)

    img=Cg                          
        
    ret,bimg=cv2.threshold(Cg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    return (img,bimg)

#8 - CrCgCb - Excess Green
def Segmentation_8(img):    
    
    #Convertion
    imgYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
     
    #Convert to Numpy array
    Cg = 255+1.211*(np.array(img[:,:,1],np.uint16)-np.array(imgYCrCb[:,:,0],np.uint16))+128             
        
    #Clip the interval
    #Cg=Cg.clip(min=255)-255
    Cg = np.uint8(Cg)     

    #Perform Excess Green Calculations
    ExCg = (255*2)+2*np.array(Cg,np.uint16)-np.array(imgYCrCb[:,:,1],np.uint16)-np.array(imgYCrCb[:,:,2],np.uint16)
    
            
    #Clip the interval
    ExCg=ExCg.clip(min=(255*2),max=(255*3))-(255*2)  
    ExCg = np.uint8(ExCg)
    img=ExCg                        
        
    ret,bimg=cv2.threshold(ExCg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    return (img,bimg)

#----------Indices Vegetativos--------------
#9 - Excess Green in RGB
def Segmentation_9(img):
   
    RGB_mat=np.array(img,np.uint16)  
    
    #Two full scales were added in order to avoid negative results 
    EXG=(255*2)+2*RGB_mat[:,:,1]-RGB_mat[:,:,2]-RGB_mat[:,:,0]     #ExG = 2G -R -B 

    #All numbers below two full scales are clipped, the scales are them subtracted
    EXG=EXG.clip(min=(255*2),max=(255*3))-(255*2)              
    
    img = np.uint8(EXG)
    ret,bimg=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
        
    return (img,bimg)

#10 - NDI
def Segmentation_10(img):
   
    RGB_mat=np.array(img,np.float32)  
    
    #Two full scales were added in order to avoid negative results 
    NDI=(RGB_mat[:,:,1]-RGB_mat[:,:,2])/(RGB_mat[:,:,1]+RGB_mat[:,:,2]+0.01)     #NDI = (G-R)/(G+B) 
    NDI=NDI.clip(min=(0))*255 #Remove negative numbers
             
    img = np.uint8(NDI)
    
    ret,bimg=cv2.threshold(img,0,255,cv2.THRESH_BINARY) #Zero thresholding
        
    return (img,bimg)


#11 - NDI mod
def Segmentation_11(img):
   
    RGB_mat=np.array(img,np.float32)  
    
    #Two full scales were added in order to avoid negative results 
    NDI=(RGB_mat[:,:,1]-RGB_mat[:,:,2])/(RGB_mat[:,:,1]+RGB_mat[:,:,2]+0.01)     #NDI = (G-R)/(G+B) 
    NDI=NDI.clip(min=(0))*255 #Remove negative numbers
             
    img = np.uint8(NDI)
    
    ret,bimg=cv2.threshold(img,0,255,cv2.THRESH_BINARY) #Zero thresholding
        
    return (img,bimg)
#TESTS

#DEFINTIONS
FileType="PNG"
#FileType="jpg"
Ni_approach=1    #Default = 1
N_approach=10    #Default = 26
N_exec=1
#kernelRadius=3

#path = raw_input("Define the image folder path:")
path = input("Define the image folder path:")

#If left in blank consider the current folder
if(path==""):
    path="*."+FileType
        
#Else goes to the selected path
else:
    path = path + "/*."+FileType

#Function that reads all the files in the folder and assing to a string vector
filenames = [imgName for imgName in glob.glob(path)]
#Organize in alfabetical order
filenames.sort() 

#Check if the images were found
if(len(filenames)>0):
    
    ans=input("{} images were found, do you wish to continue(Y/N)? ". format(len(filenames)))
    if(ans=="y" or ans=="Y" or ans==""):
            
            #Create an output folder
            create_path("OpenCV_output")
            
            #create an index for the images
            index=0
            #For each image at the folder
            for imgName in filenames:
                #Read the image
                img= cv2.imread(imgName)
                #Executes one call to the first function to ensure the same running conditions to all approaches 
                (convImg,binImg) = eval("Segmentation_" + str(Ni_approach))(img)
                
                index=index+1
                

                #-------SEGMENTATION APPROACHES-------
                for m in range(Ni_approach,N_approach+1):

                    Approach = "Segmentation_" + str(m)
                    
                    ti=time.time()
                    for n in range(0,N_exec):
                        (convImg,binImg) = eval(Approach)(img)

                    tf=time.time() - ti
                    #Calculate the mean execution time, transform in microsecond, round and convert to string
                    tm= str(int(round(1000*1000*tf/N_exec)))
                    
                    ##-------MORFOLOGICAL OPERATION-------
                    #kernel = np.ones((kernelRadius,kernelRadius),np.uint8)
                    
                    #ti=time.time()
                    #for n in range(0,N_exec):                        
                    #    BimgM = cv2.morphologyEx(Bimg, cv2.MORPH_OPEN, kernel)
                    #    BimgM = cv2.morphologyEx(BimgM, cv2.MORPH_CLOSE, kernel)
                    
                    #tf=time.time() - ti
                    ##Calculate the mean execution time, transform in microsecond, round and convert to string
                    #tmM= str(int(round(1000*1000*tf/N_exec)+int(tm)))                                       

                    #-------FILE PREPARATION-------
                    #Separates the file name from the path and the extention 
                    outFileName= os.path.split(imgName)[1].replace("."+FileType,"")
                    
                    #Compose the output file name format:
                    #A-Approach+ m-Morpho + I-Image + T-Time + Original File Name + File Type
                    outFile= "Bin_A"+str(m)+"_I"+str(index)+"_T"+tm+"_"+outFileName+"."+FileType                    
                    outFileC= "Conv_A"+str(m)+"_I"+str(index)+"_T"+tm+"_"+outFileName+"."+FileType
                    outFileS = "Out_A" + str(m) + "_I" + str(index) + "_T" + tm + "_" + outFileName + "." + FileType
                    
                    #-------DEBUG-------                        
                    #Show image for debug purpose
                    #cv2.imshow(imgName+"_"+str(m),MBimg)

                    outImg=cv2.bitwise_or(img, img, mask=binImg)


                    #-------FILE EXPORT-------
                    #Export the results into output images
                    cv2.imwrite("OpenCV_output/"+outFile,binImg)
                    cv2.imwrite("OpenCV_output/"+outFileC,convImg)
                    cv2.imwrite("OpenCV_output/" + outFileS, outImg)
                     
            #Wait the user input to terminate the program
            while(True):

                print("Image processing Done!\n")

                if cv2.waitKey(500) & 0xFF == 27:
                    break

                if (raw_input("Press enter to continue...") == ""):
                    break

            cv2.destroyAllWindows()

    else:
            print("\nProgram terminated...\n")

else:
    print("No images found!\n")  