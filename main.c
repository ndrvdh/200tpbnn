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

#define stub 1
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
  #ifdef stub
  float dataset[][3]= { {1,0,1},{0,1,1},{1,1,0},{0,0,1}};

  int set,element;

  for (set=0;set < DATASET_SIZE ; set++)
  {
    for(element=0;element <3 ;element++)
    {
      if(element < 2)
      {
        ilayer[set][element]= dataset[set][element];
      }
      if(element ==2)
      {
        exp_output[set][element]=dataset[set][element];

      }

    }
  }
  //Initialize weights
  for(int i=0;i < INPUT_LAYER_SIZE ;i++)
  {
    hweights[i][0]=1;
    for(int e=0 ; e < 2 ; e++)
    {
      iweights[i][e]=1;
    }
  }


  #else
  //To do copy from file


  #endif


}

float sigmoid_func(float x,int derive)
{
  float s=0;
  if(derive ==0)
  {
        s = 1/ (1+ exp(x));
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
  float hdlayer[DATASET_SIZE][HIDDEN_LAYER_SIZE]={0};
  float odlayer[DATASET_SIZE][0];
  float herror[DATASET_SIZE][1]={0};
  float herror_delta[DATASET_SIZE][HIDDEN_LAYER_SIZE]={0};
  float ierror[DATASET_SIZE][HIDDEN_LAYER_SIZE]={0};
  float ierror_delta[DATASET_SIZE][HIDDEN_LAYER_SIZE]={0};
  float htranspose[HIDDEN_LAYER_SIZE][DATASET_SIZE]={0};
  float itranspose[INPUT_LAYER_SIZE][DATASET_SIZE]={0};


while(t < 100)
{
  //Compute hidden layer weighted sum & its derivative
  for(i=0;i<DATASET_SIZE ; i++)
  {
    for(j=0;j<2 ; j++)
    {
      hlayer[i][j]  = (ilayer[i][0]*iweights[0][j]) + (ilayer[i][1]*iweights[1][j]);
      hlayer[i][j]  = sigmoid_func(hlayer[i][j],0);
      hdlayer[i][j] = sigmoid_func(hlayer[i][j],1);
    }
  }
  //Predict the output & compute its derivative
  for(i=0;i<DATASET_SIZE ; i++)
  {
    for(j=0;j<2 ;j++)
    {
      olayer[i][0]  = (hlayer[i][j])*(hweights[j][0]);
      olayer[i][0]  = sigmoid_func(olayer[i][0],0);
      odlayer[i][0] = sigmoid_func(olayer[i][0],1);
    }
  }

  //Calculate hidden to output  layer  error
  for(i=0;i<DATASET_SIZE ; i++)
  {
      herror[i][0] =(exp_output[i][0]-olayer[i][0])*odlayer[i][0];
  }

   /*Calculate input to hidden layer error by finding the contribution weighted error of hidden layer
   *Contributed weight errors tells us how much of each hidden layer node contributed to the final output error
   * This CWE is the input to hidden layer error
   * This is called back propagation
   */
  for(i=0;i<DATASET_SIZE ; i++)
  {
    for(j=0;j<2;j++)
    {
      herror_delta[i][j]=herror[i][0]*hweights[j][0];
      ierror[i][j] = herror_delta[i][j]*hlayer[i][j];
      ierror_delta[i][j] = ierror[i][j]* iweights[i][j];
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

  for(i=0;i<2; i++)
  {
    for(j=0;j<DATASET_SIZE ; j++)
    {
      hweights[i][0] += htranspose[i][j]*herror_delta[j][i];
      iweights[i][j] += itranspose[i][j]*ierror_delta[j][i];
    }
  }
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

      printf("\n Output for training set i is : %f", olayer[i][0]);
  }

  return status;

}

int main()
{
  printf("\nTraining set is { {1,0,1},\n {0,1,1},\n  {1,1,0},\n {0,0,1} }\n");
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
