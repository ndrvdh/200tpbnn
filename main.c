/*
 *BNN Source code
 */
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include <math.h>

/* Network Declaration*/

#define DATASET_SIZE      4
#define INPUT_LAYER_SIZE  2
#define HIDDEN_LAYER_SIZE 2
#define OUTPUT_LAYER_SIZE 1
#define TOTAL_LAYERS      3

#define stub  1
#define Debug 1
static float ilayer[DATASET_SIZE][INPUT_LAYER_SIZE]={0};   //4x2
static float exp_output[DATASET_SIZE][OUTPUT_LAYER_SIZE]={0};  //4x1
static float hlayer[DATASET_SIZE][HIDDEN_LAYER_SIZE]={0};   //4x2
static float olayer[DATASET_SIZE][OUTPUT_LAYER_SIZE]={0};   //4x1
static float iweights[INPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE]={0};  //2x2
static float hweights[HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE]={0};  //2x1

float sigmoid_func(float x,int derive);

/* Populate with Training dataset*/
void Initialize_Network(void)
{
  printf("\n Initializing Network..... \n");
  #ifdef stub
  float dataset[][3]= {{0,0,0},{0,1,1},{1,0,1},{1,1,0}};

  int set,element;

  for (set=0;set < DATASET_SIZE ; set++)  //Data set for training
  {
    for(element=0 ;element < (INPUT_LAYER_SIZE + OUTPUT_LAYER_SIZE) ;element++)  //Input elements
    {
      if(element < INPUT_LAYER_SIZE )
      {
        ilayer[set][element]= dataset[set][element];
        printf("\n Input[%d][%d] = %f",set,element,ilayer[set][element]);
      }
      if(element == INPUT_LAYER_SIZE)
      {
        exp_output[set][0]=dataset[set][element];
        printf("\n ExpectedOutput[%d][%d] = %f\n",set,0,exp_output[set][0]);
      }
    }
  }
  //Initialize weights

  hweights[0][0]= 15;
  hweights[1][0]= -15;
  for(int i=0;i < INPUT_LAYER_SIZE ;i++)
  {
    for(int e=0 ; e < HIDDEN_LAYER_SIZE ; e++)
    {
      iweights[i][e] = i- 1.5;
      printf("\n InputWeight[%d][%d] = %f",i,e,iweights[i][e]);
    }
  }
  printf("\n");
  #else
  //To do copy dataset from file


  #endif
}

float sigmoid_func(float x,int derive)
{
  float s=0;
  if(derive ==0)
  {
        s = 1/ (1+ exp(-x));
  }
  else
  {
      s = x *(1 - x);
  }
  return s;
}

int Train_Network()
{
  int i,j,t;
  int status=0;
  float l1sum=0;
  float l2sum=0;
  float hdlayer[DATASET_SIZE][HIDDEN_LAYER_SIZE]={0};
  float odlayer[DATASET_SIZE][0];
  float l2error_delta[DATASET_SIZE][OUTPUT_LAYER_SIZE]={0};
  float l1error[DATASET_SIZE][HIDDEN_LAYER_SIZE]={0};
  float l1error_delta[DATASET_SIZE][HIDDEN_LAYER_SIZE]={0};
  float htranspose[HIDDEN_LAYER_SIZE][DATASET_SIZE]={0};
  float itranspose[INPUT_LAYER_SIZE][DATASET_SIZE]={0};


while(t < 8000)
{
  //Compute input & hidden layer weighted sum & its derivative
  for(i=0;i<DATASET_SIZE ; i++)
  {
    for(j=0;j< HIDDEN_LAYER_SIZE ; j++)
    {
      hlayer[i][j]  = (ilayer[i][0]*iweights[0][j]) + (ilayer[i][1]*iweights[1][j]);
      hlayer[i][j]  = sigmoid_func(hlayer[i][j],0);
      hdlayer[i][j] = sigmoid_func(hlayer[i][j],1);
    }
  }
  //Predict the output & compute its derivative
  for(i=0;i<DATASET_SIZE ; i++)
  {

    for(j=0;j<HIDDEN_LAYER_SIZE ;j++)
    {
      olayer[i][0]  += (hlayer[i][j])*(hweights[j][0]);
    }
    olayer[i][0]  = sigmoid_func(olayer[i][0],0);
    odlayer[i][0] = sigmoid_func(olayer[i][0],1);
  }

  //Calculate hidden to output  layer  error
  for(i=0;i<DATASET_SIZE ; i++)
  {
      l2error_delta[i][0] =(exp_output[i][0]-olayer[i][0])*odlayer[i][0];
  }

   /*Calculate input to hidden layer error by finding the contribution weighted error of hidden layer
   * Contributed weight errors tells us how much of each hidden layer node contributed to the final output error
   * This CWE is the input to hidden layer error
   * This is called back propagation
   */
  for(i=0;i< HIDDEN_LAYER_SIZE ; i++)
  {
    for(j=0;j<DATASET_SIZE;j++)
    {
      l1error[j][i]=l2error_delta[j][0]*hweights[i][0];
      l1error_delta[j][i] = l1error[j][i]*hdlayer[j][i];
    }
  }

  //Update weights
  for(i=0;i<DATASET_SIZE ; i++)
  {
    for(j=0;j<2;j++)
    {
      htranspose[j][i] = hlayer[i][j];
      itranspose[j][i] = ilayer[i][j];
    }
  }

  for(i=0; i<2 ; i++)
  {
    l1sum=0;
    l2sum=0;
    for(j=0;j<DATASET_SIZE ; j++)
    {
      l2sum += htranspose[i][j]*l2error_delta[j][0];
    }
    hweights[i][0] += l2sum;

    //update l1 weights

    for(int k=0;k < 2 ;k++)
    {
      for(j=0;j<DATASET_SIZE ; j++)
      {
        l1sum += itranspose[i][j]*l1error_delta[j][k];
      }
      //l1weight[i][k] = l1sum;
      iweights[i][k] += l1sum;
    }
 }

#ifdef Debug
    //Debug prints
    if(t % 1000 == 0)
    {
      printf("\n @Iteration:%d",t);
      for(i=0; i<4 ;i++)
      {
        printf("\n Output[%d]=%f \n L2_Delta=%f \n L2_Slope=%f \n L2 Weights = %f %f \n",i,olayer[i][0],l2error_delta[i][0],odlayer[i][0],hweights[0][0],hweights[1][0]);
      }
    }
#endif

  //Loop through
    t++;
}

  for(i=0;i<DATASET_SIZE ; i++)
  {
      if( 0.4 > (exp_output[i][0]-olayer[i][0]))
      {
        status=1;
      }

      else
      {
        status=0;
      }

      printf("\n Output for training set %d is : %f", i,olayer[i][0]);
      //printf("\n L2 Weight[%d][0] = %f",i,hweights[i][0]);

  }

  return status;

}

int main()
{
  printf("\n XOR Basic Network Training \n");
  Initialize_Network();
  if(Train_Network())
  {
    printf("\n Network Successfully trained");
  }
  else
  {
    printf("\n Network training failed");

  }
  return 0;
}
