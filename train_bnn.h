/*
 *BNN Header file
 */

#ifndef BNN_HEADER_FILE
#define BNN_HEADER_FILE
/* Populate with Training dataset*/
void Initialize_Network(void);
float sigmoid_func(float x,int derive);
int Train_Network(void);
extern int bnn_train_network(char *arg);

#endif
