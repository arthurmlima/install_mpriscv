#include <stdint.h>

// CONFIGURAçao da PM
#define PIXEL_SIZE 16
#define PM_LENGTH   9
#define KERNEL_SIZE 5
#define KERNEL_SIZE_Z 3

/////////////////////GENERICOS IRAO VIRAR REDE//////////////////////////
////////////////////////////////////////////////////////////////////////
#define IMAGE_WIDTH                (*(volatile uint32_t*)0x02310000)
#define IMAGE_HEIGHT               (*(volatile uint32_t*)0x02310004)
#define SUB_IMAGE_WIDTH            (*(volatile uint32_t*)0x02310008)
#define SUB_IMAGE_HEIGHT           (*(volatile uint32_t*)0x0231000C)
#define X_TILES                    (*(volatile uint32_t*)0x02310010)    // quantidade de tiles em X
#define Y_TILES                    (*(volatile uint32_t*)0x02310014)    // quantidade de tiles em Y
#define X_LOCAL                    (*(volatile uint32_t*)0x02310018)    // posicionamento do tile em X
#define Y_LOCAL                    (*(volatile uint32_t*)0x0231001C)    // posicionamento do tile em Y
#define X_INIT                     (*(volatile uint32_t*)0x02310020)    // pixels iniciais do tile em X
#define Y_INIT                     (*(volatile uint32_t*)0x02310024)    // pixels iniciais do tile em Y
#define LEDS                       (*(volatile uint32_t*)0x0223F000)
///////////////////INTERFACE DE TRANSMISSAO AXI/////////////////////////
////////////////////////////////////////////////////////////////////////
#define AXI_IMAGE_PIXEL            (*(volatile uint32_t*)0x02300000)
#define AXI_IMAGE_X                (*(volatile uint32_t*)0x02300004)
#define AXI_IMAGE_Y                (*(volatile uint32_t*)0x02300008)
#define AXI_IMAGE_REQ              (*(volatile uint32_t*)0x0230000C)
#define AXI_IMAGE_ACK              (*(volatile uint32_t*)0x02300010)

///////////////////INTERFACE DE TRANSMISSAO AXI/////////////////////////
////////////////////////////////////////////////////////////////////////
#define AXI_RISCV_IMAGE_PIXEL      (*(volatile uint32_t*)0x02400000)
#define AXI_RISCV_IMAGE_REQ        (*(volatile uint32_t*)0x02400004)
#define AXI_RISCV_IMAGE_ACK        (*(volatile uint32_t*)0x02400008)

///////////////////////PROBES PARA PROFILING////////////////////////////
////////////////////////////////////////////////////////////////////////
#define ENTRADA_INIT_PROG_FIM      (*(volatile uint32_t*)0x02300014)
#define SAIDA_INIT_PROG_FIM        (*(volatile uint32_t*)0x02300018)



///////////////////////INTERFACE COM PM E ROUTER////////////////////////
////////////////////////////////////////////////////////////////////////
#define SET_PIXEL_BUFFER_TOP        (*(volatile uint32_t*)0x02230000)
#define SET_PIXEL_BUFFER            (*(volatile uint32_t*)0x02220000)
#define PE_WRITE_ADDRESS_TOP        (*(volatile uint32_t*)0x02210000)
#define PE_WRITE_ADDRESS            (*(volatile uint32_t*)0x02200000)
#define PE_READ_ADDRESS_TOP         (*(volatile uint32_t*)0x02110000)
#define PE_READ_ADDRESS             (*(volatile uint32_t*)0x02100000)


/////////////////////////CONFIGURACAO UART//////////////////////////////
////////////////////////////////////////////////////////////////////////
#define reg_uart_clkdiv             (*(volatile uint32_t*)0x02000004)
#define reg_uart_data               (*(volatile uint32_t*)0x02000008)


// ASSIGNS DE PARAMETERS
/**/
#define write_buffer_top             (*(volatile uint32_t*)0x00006204)
#define write_buffer                 (*(volatile uint32_t*)0x00006200)

#define read_message_pixelValue      (*(volatile uint32_t*)0x00006100)
#define read_message_xDest           (*(volatile uint32_t*)0x00006104)
#define read_message_yDest           (*(volatile uint32_t*)0x00006108)
#define read_message_step            (*(volatile uint32_t*)0x0000610C)
#define read_message_frame           (*(volatile uint32_t*)0x00006110)
#define read_message_xOrig           (*(volatile uint32_t*)0x00006114)
#define read_message_yOrig           (*(volatile uint32_t*)0x00006118)
#define read_message_fb              (*(volatile uint32_t*)0x0000611c)
#define read_message_req             (*(volatile uint32_t*)0x00006120)
#define read_message_ack             (*(volatile uint32_t*)0x00006124)




/////////MASCARAS PARA BIT HANDLING DE ESCRITA E LEITURA////////////////
////////////////////////////////////////////////////////////////////////
#define SET_PM_PIXEL_MASK           0x0000FFFF
#define SET_PM_X_DEST_MASK          0x7F800000
#define SET_PM_Y_DEST_MASK          0x007F8000
#define SET_PM_STEP                 0x00007C00
#define SET_PM_FRAME                0x000003FC

#define SET_ADDRESS_MASK            0x000003FE
#define SET_REQ_MASK                0x00000001

#define PE_WRITE_ADDRESS_MASK_PIXEL 0xFFFF0000
#define PE_WRITE_ADDRESS_MASK_XDEST 255UL<<8
#define PE_WRITE_ADDRESS_MASK_YDEST 255UL<<0
#define PE_WRITE_ADDRESS_MASK_STEP  31UL<<27
#define PE_WRITE_ADDRESS_MASK_FRAME 255UL<<19
#define PE_WRITE_ADDRESS_MASK_XORIG 255UL<<11
#define PE_WRITE_ADDRESS_MASK_YORIG 255UL<<3
#define PE_WRITE_ADDRESS_MASK_FB    1UL<<2
#define PE_WRITE_ADDRESS_MASK_REQ   1UL<<1
#define PE_WRITE_ADDRESS_MASK_REQ   1UL<<1
#define PE_WRITE_ADDRESS_MASK_ACK   1UL


void setPixel(uint32_t pixel_Value);
void setXdest(uint32_t x_Dest);
void setYdest(uint32_t y_Dest);
void setStep(uint32_t step);
void setFrame(uint32_t frame);
void setXorig(uint32_t x_Orig);
void setYorig(uint32_t y_Orig);
void setFb(uint32_t fb);

void set_pm_pixel(uint32_t pixel);
void set_pm_x_dest(uint32_t x_dest);
void set_pm_y_dest(uint32_t y_dest);
void set_pm_step(uint32_t step);
void set_pm_frame(uint32_t frame);

void readPixel(void);
void readXdest(void);
void readYdest(void);
void readStep(void);
void readFrame(void);
void readXorig(void);
void readYorig(void);
void readFb(void);
void readReq(void);
void readAck(void);


void read_gpio(void);
void write_gpio(uint32_t pixel, uint32_t x_dest, uint32_t y_dest, uint32_t step, uint32_t frame, uint32_t x_orig, uint32_t y_orig, uint32_t fb);
void set_pixel(uint32_t pixel, uint32_t x_dest, uint32_t y_dest, uint32_t step, uint32_t frame);

void Read_message(void);


void putchart(char c);
void print(const char *p);
void print_dec(uint32_t v);


//utilidades
void delay(uint32_t k);
void wait_step(void);



////testes

//func para a recepcao da imagem do ARM
void distribute_image_from_zynq(void);
int check_last_pixel(uint32_t x,uint32_t y);
int check_local_pixel(uint32_t x,uint32_t y);
void wait_for_checkmsg(void);

//func para a recepcao da imagem do ARM
void get_first_image_recv(void);
void send_transfer_finished_notice_recv(void);
void get_and_set_local_image_from_zynq_recv(void);
void get_and_set_local_pixel_from_zynq_recv(void);


void back2arm(void);
void convolution(void);
void convolution_z(void);

void core_init(void);
uint32_t get_pixel(uint32_t x,uint32_t y,uint32_t s,uint32_t f);
void wait_central(void);
void wait_perifericos(void);



volatile uint32_t kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1, 1,  1,  1,  1},
    {1, 1,  1,  1,  1},
    {1, 1,  1,  1,  1},
    {1, 1,  1,  1,  1},
    {1, 1,  1,  1,  1}
};


volatile uint32_t kernelz[KERNEL_SIZE_Z][KERNEL_SIZE_Z] = {
    {  1,  1,  1},
    {  1,  1,  1},
    {  1,  1,  1}
};


void main()
{

     reg_uart_clkdiv=868;


    core_init(); // se tu eh um tile periferico recebe, se for o caso do central envia/recebe

    back2arm();


    while(1);


}

void back2arm(void)
{
    for(volatile uint32_t i = 0; i < 240; i++)
    {
        for(volatile uint32_t j = 0; j < 240; j++)
        {

         AXI_RISCV_IMAGE_PIXEL=get_pixel(j,i,1,0);

         AXI_RISCV_IMAGE_REQ=1;

         // escrever no registrador o pixel
         //dar um REQ

         while(AXI_RISCV_IMAGE_ACK==0);

         AXI_RISCV_IMAGE_REQ=0;

         while(AXI_RISCV_IMAGE_ACK==1);

        }
    }
}

void core_init(void)
{
        if(X_LOCAL==1){
            if(Y_LOCAL==1){
                distribute_image_from_zynq();
                convolution_z();
                wait_perifericos();
            }
            else
            {
                get_first_image_recv();
                convolution_z();
                wait_central();
            }
        }
        else
        {
            get_first_image_recv();
            convolution_z();
            wait_central();
        }

}
void wait_central(void)
{
    SAIDA_INIT_PROG_FIM=1;

	while((ENTRADA_INIT_PROG_FIM & (~0x1EF) )!=0x10)
    {
		LEDS=2;

        __asm("nop");
    }
    SAIDA_INIT_PROG_FIM=0;
}
void wait_perifericos(void)
{
	while((ENTRADA_INIT_PROG_FIM)!=0x1EF)
    {
		LEDS=3;
        __asm("nop");
    }
    SAIDA_INIT_PROG_FIM=1;

	while((ENTRADA_INIT_PROG_FIM)!=0x10)
    {
        LEDS=1;
        __asm("nop");
    }
	SAIDA_INIT_PROG_FIM=0;
}
/****************************************************************************************
**************************FUNCOES PARA ESCREVER NA PM LOCAL*****************************
*****************************************************************************************/
void set_pixel(uint32_t pixel, uint32_t x_dest, uint32_t y_dest, uint32_t step, uint32_t frame)
{
// resetar o buff de envio
SET_PIXEL_BUFFER=0ULL;
SET_PIXEL_BUFFER_TOP=0;


    set_pm_pixel(pixel);
    set_pm_x_dest(x_dest);
    set_pm_y_dest(y_dest);
    set_pm_step(step);
    set_pm_frame(frame);

//setar o request
SET_PIXEL_BUFFER= SET_PIXEL_BUFFER | 0x2;
//esperar pelo acknowledge do recebimento
while((SET_PIXEL_BUFFER & 0x1)== 0);

//reseta o pino de request
SET_PIXEL_BUFFER= SET_PIXEL_BUFFER & (~0x00000002) ;
//esperar pelo reset do acknowledge
while((SET_PIXEL_BUFFER & 0x1)== 1);

SET_PIXEL_BUFFER=0ULL;
SET_PIXEL_BUFFER_TOP=0;
}

void set_pm_pixel(uint32_t pixel)
{
    SET_PIXEL_BUFFER_TOP  |=   (SET_PIXEL_BUFFER_TOP & ~SET_PM_PIXEL_MASK) | ((uint32_t)pixel);
}

void set_pm_x_dest(uint32_t x_dest)
{
	SET_PIXEL_BUFFER  |=   (SET_PIXEL_BUFFER & ~SET_PM_X_DEST_MASK) | ((uint32_t)x_dest<<23);
}
void set_pm_y_dest(uint32_t y_dest)
{
	SET_PIXEL_BUFFER  |=   (SET_PIXEL_BUFFER & ~SET_PM_Y_DEST_MASK) | ((uint32_t)y_dest<<15);
}
void set_pm_step(uint32_t step)
{
	SET_PIXEL_BUFFER  |=   (SET_PIXEL_BUFFER & ~SET_PM_STEP) | ((uint32_t)step<<10);
}
void set_pm_frame(uint32_t frame)
{
	SET_PIXEL_BUFFER  |=   (SET_PIXEL_BUFFER & ~SET_PM_FRAME) | ((uint32_t)frame<<2);
}


/****************************************************************************************
****************************FIM DAS FUNCOES PARA ESCREVER NA PM LOCAL********************
*****************************************************************************************/






/**********************************************************************************************************************************************
**************************************************************FUNCOES PARA ESCREVER NA REDE***************************************************
*****************************************************************************************/

void write_gpio(uint32_t pixel, uint32_t x_dest, uint32_t y_dest, uint32_t step, uint32_t frame, uint32_t x_orig, uint32_t y_orig, uint32_t fb)
{
// resetar o buff de envio
PE_WRITE_ADDRESS=0ULL;

    PE_WRITE_ADDRESS=0;
    PE_WRITE_ADDRESS_TOP=0;
    write_buffer_top=0;
    write_buffer=0;

    setPixel(pixel);
    setXdest(x_dest);
    setYdest(y_dest);
    setFrame(step);
    setStep (frame);
    setXorig(x_orig);
    setYorig(y_orig);
    setFb   (fb);
    PE_WRITE_ADDRESS_TOP= write_buffer_top;
    PE_WRITE_ADDRESS= write_buffer;


//setar o request
PE_WRITE_ADDRESS= PE_WRITE_ADDRESS | 0x2;

//esperar pelo acknowledge do recebimento
while((PE_WRITE_ADDRESS & 0x1)== 0);

//reseta o pino de request
PE_WRITE_ADDRESS= PE_WRITE_ADDRESS & (~0x00000002) ;

//esperar pelo reset do acknowledge
while((PE_WRITE_ADDRESS & 0x1)== 1);

}


void setPixel(uint32_t pixel_Value)
{
		write_buffer_top |=   (write_buffer_top & ~PE_WRITE_ADDRESS_MASK_PIXEL) | ((uint32_t)pixel_Value<<16);
}
void setXdest(uint32_t x_Dest)
{
		write_buffer_top |=   (write_buffer_top & ~PE_WRITE_ADDRESS_MASK_XDEST) | ((uint32_t)x_Dest<<8);
}
void setYdest(uint32_t y_Dest)
{
		write_buffer_top |=   (write_buffer_top & ~PE_WRITE_ADDRESS_MASK_YDEST) | ((uint32_t)y_Dest);
}

void setStep(uint32_t step)
{
	    write_buffer |=   (write_buffer & ~PE_WRITE_ADDRESS_MASK_STEP) | ((uint32_t)step<<27);
}
void setFrame(uint32_t frame)
{
	    write_buffer |=   (write_buffer & ~PE_WRITE_ADDRESS_MASK_FRAME) | ((uint32_t)frame<<19);
}

void setXorig(uint32_t x_Orig)
{
	    write_buffer |=   (write_buffer & ~PE_WRITE_ADDRESS_MASK_XORIG) | ((uint32_t)x_Orig<<11);
}
void setYorig(uint32_t y_Orig)
{
	    write_buffer |=   (write_buffer & ~PE_WRITE_ADDRESS_MASK_YORIG) | ((uint32_t)y_Orig<<3);
}
void setFb(uint32_t fb)
{
	    write_buffer |=   (write_buffer & ~PE_WRITE_ADDRESS_MASK_FB) | ((uint32_t)fb<<2);
}



/****************************************************************************************
****************************FIM DAS FUNCOES PARA ESCREVER NA REDE****************************
*****************************************************************************************/











/****************************************************************************************
****************************FUNCOES PARA LER NA REDE****************************
*****************************************************************************************/
void read_gpio(void)
{

    // espera algum sinal de request
    while((PE_READ_ADDRESS & 0x2)==0);

    // recebe a mensagem
    readPixel();
    readXdest();
    readYdest();
    readStep ();
    readFrame();
    readXorig();
    readYorig();
    readFb();    // seta o acknowledge
    PE_READ_ADDRESS= PE_READ_ADDRESS | 0x1;

    // espera o reset do request
    while((PE_READ_ADDRESS & 0x2)==1);

    // reseta o pino do acknowledge
    PE_READ_ADDRESS= PE_READ_ADDRESS & (~0x1);
}

/*
 "1 1111 1111"  "1111 1111"  "1111 1111"  "1111 1111"  "1111 1111"  "1111 1111"  "1111 1111"   "1"    "1"  |  "1"
    Pixel          Xdest        Ydest        Step         Frame        Xorig        Yorig       fb    Req  |  Ack

                            ESCRITA                                                                       | LEITURA
*/

void readPixel(void)
{

    read_message_pixelValue= ((PE_READ_ADDRESS_TOP) & (PE_WRITE_ADDRESS_MASK_PIXEL))>>16;

}
void readXdest(void)
{
    read_message_xDest= ((PE_READ_ADDRESS_TOP) & (PE_WRITE_ADDRESS_MASK_XDEST))>>8;
}
void readYdest(void)
{
   read_message_yDest= ((PE_READ_ADDRESS_TOP) & (PE_WRITE_ADDRESS_MASK_YDEST));
}

void readStep(void)
{
	 read_message_step= ((PE_READ_ADDRESS) & (PE_WRITE_ADDRESS_MASK_STEP))>>27;
}
void readFrame(void)
{
	read_message_frame=((PE_READ_ADDRESS) & (PE_WRITE_ADDRESS_MASK_FRAME))>>19;
}
void readXorig(void)
{
	 read_message_xOrig= ((PE_READ_ADDRESS) & (PE_WRITE_ADDRESS_MASK_XORIG))>>11;
}
void readYorig(void)
{
	read_message_yOrig =((PE_READ_ADDRESS) & (PE_WRITE_ADDRESS_MASK_YORIG))>>3;
}
void readFb(void)
{
	read_message_fb=((PE_READ_ADDRESS) & (PE_WRITE_ADDRESS_MASK_FB))>>2;
}
void readReq(void)
{
	read_message_req=((PE_READ_ADDRESS) & (PE_WRITE_ADDRESS_MASK_REQ))>>1;
}
void readAck(void)
{
	read_message_ack=((PE_READ_ADDRESS) & (PE_WRITE_ADDRESS_MASK_ACK))>>0;
}


/****************************************************************************************
****************************FUNCOES PARA LER NA REDE****************************
*****************************************************************************************/





/****************************************************************************************
****************************FUNCOES PARA LER ARM SET IMAGEM INICIAL**********************
*****************************************************************************************/

void putchart(char c)
{
	if (c == '\n')
		putchart('\r');
	reg_uart_data = c;
}

void print(const char *p)
{
	while (*p)
		putchart(*(p++));
}

void print_dec(uint32_t v)
{

	if      (v >= 900) { putchart('9'); v -= 900; }
	else if (v >= 800) { putchart('8'); v -= 800; }
	else if (v >= 700) { putchart('7'); v -= 700; }
	else if (v >= 600) { putchart('6'); v -= 600; }
	else if (v >= 500) { putchart('5'); v -= 500; }
	else if (v >= 400) { putchart('4'); v -= 400; }
	else if (v >= 300) { putchart('3'); v -= 300; }
	else if (v >= 200) { putchart('2'); v -= 200; }
	else if (v >= 100) { putchart('1'); v -= 100; }
	else putchart('0');

	if      (v >= 90) { putchart('9'); v -= 90; }
	else if (v >= 80) { putchart('8'); v -= 80; }
	else if (v >= 70) { putchart('7'); v -= 70; }
	else if (v >= 60) { putchart('6'); v -= 60; }
	else if (v >= 50) { putchart('5'); v -= 50; }
	else if (v >= 40) { putchart('4'); v -= 40; }
	else if (v >= 30) { putchart('3'); v -= 30; }
	else if (v >= 20) { putchart('2'); v -= 20; }
	else if (v >= 10) { putchart('1'); v -= 10; }
	else putchart('0');

	if      (v >= 9) { putchart('9'); v -= 9; }
	else if (v >= 8) { putchart('8'); v -= 8; }
	else if (v >= 7) { putchart('7'); v -= 7; }
	else if (v >= 6) { putchart('6'); v -= 6; }
	else if (v >= 5) { putchart('5'); v -= 5; }
	else if (v >= 4) { putchart('4'); v -= 4; }
	else if (v >= 3) { putchart('3'); v -= 3; }
	else if (v >= 2) { putchart('2'); v -= 2; }
	else if (v >= 1) { putchart('1'); v -= 1; }
	else putchart('0');
}



////////////////////////////////////////////////////
//TESTES E APLICACOES//////////////////////////////



void distribute_image_from_zynq(void)
{
    volatile uint32_t count=0;
    while(count!=IMAGE_WIDTH*IMAGE_HEIGHT)
    {
    // espera o req da imagem ser 1
    while(AXI_IMAGE_REQ==0);

    // verificar se o pixel é da memoria interna
    if(check_local_pixel(AXI_IMAGE_X,AXI_IMAGE_Y))
    {
        set_pixel(AXI_IMAGE_PIXEL,AXI_IMAGE_X,AXI_IMAGE_Y,0,0);
    }
    else
    {
         write_gpio(AXI_IMAGE_PIXEL,AXI_IMAGE_X,AXI_IMAGE_Y,0,0,AXI_IMAGE_X,AXI_IMAGE_Y,1);
    }
    AXI_IMAGE_ACK=1;
    // esperar reset do req
    while(AXI_IMAGE_REQ==1);
    AXI_IMAGE_ACK=0;
    count=count+1;
    }
    wait_perifericos();
}
int check_last_pixel(uint32_t x,uint32_t y)
{
    if( y== (Y_INIT+SUB_IMAGE_HEIGHT-1)){
        if( x== (X_INIT+SUB_IMAGE_WIDTH-1)){
            return 1;
        }
    }
    return 0;
}
int check_local_pixel(uint32_t x,uint32_t y)
{
    if ((x >= X_INIT) & (x< X_INIT+SUB_IMAGE_WIDTH)){
      if((y >= Y_INIT) & (y< (Y_INIT+SUB_IMAGE_HEIGHT))){
        return 1;
    }}}
//funcs referentes aos tiles de fora
void get_first_image_recv(void)
{
get_and_set_local_image_from_zynq_recv();
wait_central();
}

void wait_for_checkmsg(void)
{
        while((ENTRADA_INIT_PROG_FIM & 0x1FF)!=0x1FF)
        {
            __asm("nop");
        }
}

void send_transfer_finished_notice_recv(void)
{
    SAIDA_INIT_PROG_FIM = 0x1;
}

void wait_step(void)
{
    LEDS=3;

    while(ENTRADA_INIT_PROG_FIM!=0x1FF){
    }
    SAIDA_INIT_PROG_FIM=0;
}

void get_and_set_local_image_from_zynq_recv(void)
{
    for(volatile uint32_t i = 0; i < SUB_IMAGE_HEIGHT; i++)
    {
        for(volatile uint32_t j = 0; j < SUB_IMAGE_WIDTH; j++)
        {
        get_and_set_local_pixel_from_zynq_recv();
        }
    }
}

void get_and_set_local_pixel_from_zynq_recv(void)
{
    read_gpio();
    set_pixel(read_message_pixelValue,read_message_xDest,read_message_yDest,0,0);
}

void delay(uint32_t k)
{
    for (volatile uint16_t ko=0;ko<k;ko++){
        __asm("nop");
    }
}

uint32_t get_pixel(uint32_t x,uint32_t y,uint32_t s,uint32_t f)
{
        write_gpio(0,x,y,s,f,X_INIT,Y_INIT,0);
        read_gpio();
        return read_message_pixelValue;
}

void convolution(void) {



    for (int i = Y_LOCAL*SUB_IMAGE_HEIGHT; i < Y_LOCAL*SUB_IMAGE_HEIGHT+SUB_IMAGE_HEIGHT; i++) {
        for (int j = X_LOCAL*SUB_IMAGE_WIDTH; j < X_LOCAL*SUB_IMAGE_WIDTH+SUB_IMAGE_WIDTH; j++) {
            uint32_t sum = 0;
            uint8_t sum_npixels=0;
            // Loop over each element in the kernel
            for (int k = -KERNEL_SIZE/2; k <= KERNEL_SIZE/2; k++) {
                for (int l = -KERNEL_SIZE/2; l <= KERNEL_SIZE/2; l++) {

                    // Calculate the index of the pixel in the image
                    int x = j + l;
                    int y = i + k;

                    // Handle edge cases by setting out of bounds pixels to 0
                    if (x < 0 || x >= IMAGE_WIDTH || y < 0 || y >= IMAGE_HEIGHT) {
                        sum += 0;
                    } else {
                        sum += kernel[k+KERNEL_SIZE/2][l+KERNEL_SIZE/2] * get_pixel(x,y,0,0);
                        sum_npixels++;
                    }
                }
            }

            // Set the new pixel value in the image
            set_pixel((uint32_t)sum/sum_npixels,j,i,2,0);

        }
    }

}
void convolution_z(void) {



    for (int i = Y_LOCAL*SUB_IMAGE_HEIGHT; i < Y_LOCAL*SUB_IMAGE_HEIGHT+SUB_IMAGE_HEIGHT; i++) {
        for (int j = X_LOCAL*SUB_IMAGE_WIDTH; j < X_LOCAL*SUB_IMAGE_WIDTH+SUB_IMAGE_WIDTH; j++) {
            uint32_t sum = 0;
            uint8_t sum_npixels=0;
            // Loop over each element in the kernel
            for (int k = -KERNEL_SIZE_Z/2; k <= KERNEL_SIZE_Z/2; k++) {
                for (int l = -KERNEL_SIZE_Z/2; l <= KERNEL_SIZE_Z/2; l++) {

                    // Calculate the index of the pixel in the image
                    int x = j + l;
                    int y = i + k;

                    // Handle edge cases by setting out of bounds pixels to 0
                    if (x < 0 || x >= IMAGE_WIDTH || y < 0 || y >= IMAGE_HEIGHT) {
                        sum += 0;
                    } else {
                        sum += kernelz[k+KERNEL_SIZE_Z/2][l+KERNEL_SIZE_Z/2] * get_pixel(x,y,0,0);
                        sum_npixels++;
                    }
                }
            }

            // Set the new pixel value in the image
            set_pixel((uint32_t)sum/sum_npixels,j,i,1,0);

        }
    }

}
