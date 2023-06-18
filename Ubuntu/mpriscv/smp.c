/*
 * Copyright (c) 2012 Xilinx, Inc.  All rights reserved.
 *
 * Xilinx, Inc.
 * XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS" AS A
 * COURTESY TO YOU.  BY PROVIDING THIS DESIGN, CODE, OR INFORMATION AS
 * ONE POSSIBLE   IMPLEMENTATION OF THIS FEATURE, APPLICATION OR
 * STANDARD, XILINX IS MAKING NO REPRESENTATION THAT THIS IMPLEMENTATION
 * IS FREE FROM ANY CLAIMS OF INFRINGEMENT, AND YOU ARE RESPONSIBLE
 * FOR OBTAINING ANY RIGHTS YOU MAY REQUIRE FOR YOUR IMPLEMENTATION.
 * XILINX EXPRESSLY DISCLAIMS ANY WARRANTY WHATSOEVER WITH RESPECT TO
 * THE ADEQUACY OF THE IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO
 * ANY WARRANTIES OR REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE
 * FROM CLAIMS OF INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "program.h"
#include <time.h>

#define TILE_0_0 0x400000000
#define TILE_0_1 0x1000000000
#define TILE_0_2 0x1100000000
#define TILE_1_0 0x1200000000
#define TILE_1_1 0x1300000000
#define TILE_1_2 0x1400000000
#define TILE_2_0 0x1500000000
#define TILE_2_1 0x1600000000
#define TILE_2_2 0x1700000000


/* General Purpose Register - Control Bits */
#define VIS_AXI_EN			((uint32_t )0x00000001)
#define VIS_AXI_WEN			((uint32_t )0x0000001E) // HABILITA TUDO

/* General Purpose Register - Status Bits */
#define VIS_AXI_OK			((uint32_t )0x00100000)

/* Address Register */
#define VIS_AXI_ADR			((uint32_t)0x000FFFE0)
#define VIS_AXI_ADR_SHIFT	0UL
#define VIS_AXI_ADR_MASK	((uint32_t)0xFFFFFFFF)

/* Data In Register */
#define VIS_AXI_DTI			((uint32_t)0x00000001)
#define VIS_AXI_DTI_SHIFT	0UL
#define VIS_AXI_DTI_MASK	((uint32_t)0xFFFFFFFF)

/* Data Out Register */
#define VIS_AXI_DTO			((uint32_t)0x00000001)
#define VIS_AXI_DTO_SHIFT	0UL
#define VIS_AXI_DTO_MASK	((uint32_t)0xFFFFFFFF)
typedef struct
{
	volatile uint32_t ADDR;
	volatile uint32_t DIN;
	volatile uint32_t DOUT;
	volatile uint32_t RST;
} Vis_AxiStruct;
typedef struct
	{
	volatile uint32_t PIXEL;
	volatile uint32_t REQ;
	volatile uint32_t ACK;
	}ImageInitRecv_struct;
	typedef struct
	{
	volatile uint32_t PIXEL;
	volatile uint32_t X_DEST;
	volatile uint32_t Y_DEST;
	volatile uint32_t REQ;
	volatile uint32_t ACK;
	}ImageInitTransfer_struct;

void prog_riscv(Vis_AxiStruct* p);
void transfer_image(ImageInitTransfer_struct *p,uint32_t pixel,uint32_t x_dest,uint32_t y_dest, float *t_primeiro_ma, float *t_ultimo_ma);
uint8_t* recv_pixel(ImageInitRecv_struct *p,float *t_primeiro_am, float *t_ultimo_am);

uint8_t mlena1[240][240];
uint8_t mlena2[240][240];
uint8_t ml2[240*240];

uint8_t* mpriscv(int sel_img, float *t0, float *t1, float *t2, float *t3,float *t4,float *t5) {
  	
	*t0= 1.73;
	unsigned 	page_addr;
	unsigned	page_offset;
	unsigned 	page_size = sysconf(_SC_PAGESIZE);

	Vis_AxiStruct *tile00 ;
	Vis_AxiStruct *tile01 ;
	Vis_AxiStruct *tile02 ;
	Vis_AxiStruct *tile03 ;
	Vis_AxiStruct *tile04 ;
	Vis_AxiStruct *tile05 ;
	Vis_AxiStruct *tile06 ;
	Vis_AxiStruct *tile07 ;
	Vis_AxiStruct *tile08 ;
	ImageInitRecv_struct     *Receiv   ;
	ImageInitTransfer_struct *Transfer ;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);


	tile00 = (Vis_AxiStruct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(TILE_0_0));
	tile01 = (Vis_AxiStruct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(TILE_0_1));
	tile02 = (Vis_AxiStruct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(TILE_0_2));
	tile03 = (Vis_AxiStruct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(TILE_1_0));
	tile04 = (Vis_AxiStruct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(TILE_1_1));
	tile05 = (Vis_AxiStruct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(TILE_1_2));
	tile06 = (Vis_AxiStruct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(TILE_2_0));
	tile07 = (Vis_AxiStruct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(TILE_2_1));
	tile08 = (Vis_AxiStruct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(TILE_2_2));


	Transfer = (ImageInitTransfer_struct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(0xA0000000));
	Receiv = (ImageInitRecv_struct *)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(0xA0001000));

	prog_riscv(tile00);
	prog_riscv(tile01);
	prog_riscv(tile02);
	prog_riscv(tile03);
	prog_riscv(tile04);
	prog_riscv(tile05);
	prog_riscv(tile06);
	prog_riscv(tile07);
	prog_riscv(tile08);


	tile00->RST=0;
	tile01->RST=0;
	tile02->RST=0;
	tile03->RST=0;
	tile04->RST=0;
	tile05->RST=0;
	tile06->RST=0;
	tile07->RST=0;
	tile08->RST=0;
	usleep(50);
	tile00->RST=1;
	tile01->RST=1;
	tile02->RST=1;
	tile03->RST=1;
	tile04->RST=1;
	tile05->RST=1;
	tile06->RST=1;
	tile07->RST=1;
	tile08->RST=1;
	close(fd);



	transfer_image(Transfer,conf[sel_img][j+240*i],j,i);


      return recv_pixel(Receiv);
}

void transfer_image(ImageInitTransfer_struct *p, uint32_t pixel, uint32_t x_dest, uint32_t y_dest, float *t_primeiro_ma, float *t_ultimo_ma)
{
	  for (volatile uint32_t i = 0; i < 240; i++)
	  {
		  for (volatile uint32_t j = 0; j < 240; j++)
		  {
			  if (i = 240 &&j = 240)
			  {
				  *t_primeiro_ma = p->T_PRIMEIRO_MA;
			  }
			  else if (i = 0 &&j = 0)
			  {
				  *t_ultimo_ma = p->T_ULTIMO_MA;
			  }
			  // primeiro eu seto os valores
			  p->PIXEL = pixel;
			  p->X_DEST = x_dest;
			  p->Y_DEST = y_dest;
			  for (volatile int x = 0; x < 20000; x++)
				  ;

			  // setar o req
			  p->REQ = 1;
			  // espera o ack = '1'
			  while (p->ACK == 0)
				  ;

			  // reseta o req
			  p->REQ = 0;
			  // espera o ack = '0'
			  while (p->ACK == 1)
				  ;
		  }
	  }
}

uint8_t *recv_pixel(ImageInitRecv_struct *p, float *t_primeiro_am, float *t_ultimo_am)
{
	  for (volatile uint32_t i = 0; i < 240; i++)
	  {
		  for (volatile uint32_t j = 0; j < 240; j++)
		  {
			  if (i = 240 &&j = 240)
			  {
				  *t_primeiro_am = p->T_PRIMEIRO_MA;
			  }
			  else if (i = 0 &&j = 0)
			  {
				  *t_ultimo_am = p->T_ULTIMO_MA;
			  }
			  while (p->REQ == 0)
				  ;

			  // setar o req
			  p->ACK = 1;
			  ml2[j + 240 * i] = p->PIXEL;
			  // espera o ack = '1'
			  while (p->REQ == 1)
				  ;

			  // reseta o req
			  p->ACK = 0;
		  }
	  }
	  return ml2;
}

void prog_riscv(Vis_AxiStruct* p)
{
for(int i=0; i < sizeof(program)/sizeof(program[0]);i++)
{

	p->DIN =  (uint32_t) program[i];
	p->ADDR |= VIS_AXI_WEN ;
	//printf("%8.x\n",program[i]);
	p->ADDR |= ((uint32_t)addrs[i] << 5);
	p->ADDR |= VIS_AXI_EN ; // Gatilho do enable por ultimo

   while(!(p->ADDR & VIS_AXI_OK))
   {
    	__asm("nop");
   }
   p->ADDR =0;
}
}



