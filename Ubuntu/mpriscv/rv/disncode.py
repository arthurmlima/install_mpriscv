import re
import sys
# open a file for reading
addr = []
val = []
i=0
def strbigend(x):
    x=list(x)
    x1 = []
    x1[0:2]=x[6:8]
    x1[2:4]=x[4:6]
    x1[4:6]=x[2:4]
    x1[6:8]=x[0:2]
    return ''.join(x1)
with open(sys.argv[1], 'r') as f:
    pattern1=r":*[0-9a-fA-F]{8} [^.<>]"
    pattern2=r"<*>:"
    for line in f:        
        if re.search(pattern2, line.strip()):
            words = re.split('[<>]', line.strip())     
            if words[1] == ".comment":
                break
        elif re.search(pattern1, line.strip()):
            words = re.split('[:\t]', line.strip())     
            addr.append(int(words[0],16))
            val.append(int(words[2],16))

with open('diss.dsm', 'r') as f1:
    for line in f1:
        words = re.split('[ ]', line.strip()) 
        for count in range(0,4):
            try:    
                hex_int = int(words[count], 16)
                if count==0:
                    addr.append(int(words[0], 16))
                    val.append(int( strbigend(words[1]), 16))
                else:
                    val.append(int(strbigend(words[count+1]), 16))
                    addr.append(addr[-1]+4)
            except ValueError:                
                break


with open(sys.argv[2], 'w') as outfile:
    # loop over the list and write each value in hexadecimal format to the file
    message = "#include<stdint.h> \n uint32_t addrs[{0}]=".format(int(len(addr)))
    outfile.write(str(message) + str('\n{') +'\n')
    j=0
    for num in addr:
        addr[j]=int(addr[j]/4)
        if j < len(addr)-1:
            outfile.write(f"0x{addr[j]:08x}"+","+'\n')
            j=j+1
        elif j>=len(addr)-1:
            outfile.write(f"0x{addr[j]:08x}"+'\n')
            j=j+1
    outfile.write(str('\n};') +'\n')

   # loop over the list and write each value in hexadecimal format to the file
    message = "\n\nuint32_t program[{0}]=".format(int(len(val)))
    outfile.write(str(message) + str('\n{') +'\n')
    j=0
    for num in val:
        val[j]=int(val[j])
        if j < len(val)-1:
            outfile.write(f"0x{val[j]:08x}"+","+'\n')
            j=j+1
        elif j>=len(val)-1:
            outfile.write(f"0x{val[j]:08x}"+'\n')
            j=j+1
    outfile.write(str('\n};') +'\n')
