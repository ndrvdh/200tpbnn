/*
 *BNN Source code
 */
#include <stdio.h>
#include <stdin.h>
#include <math.h>

#define INPUT_LAYER_SIZE  (2)
#define HIDDEN_LAYER_SIZE (2)
#define OUTPUT_LAYER_SIZE (1)

typedef struct connection  //connection id with its corresponding weight
{
  uint8_t id;
  uint8_t weight;
}connection_t;

typedef struct node{
  connection_t* xn;  //pointer to connection array of a node
  uint8_t bias;
  uint8_t value;
  uint8_t node_type;   //input or hidden or output node
}node_t;

typedef struct BNN
{
  node_t input[INPUT_LAYER_SIZE];
  node_t hidden[HIDDEN_LAYER_SIZE];
  node_t output[OUTPUT_LAYER_SIZE];
}BNN_t;

int main()
{
  return 0;
}
