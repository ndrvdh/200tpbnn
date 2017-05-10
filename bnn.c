/*
* BNN - BINARY NETWORK SIMULATOR
*
* The Binary Network Simulator allows the setup, training and execution of a neural network
* with 3 to 16 layers and 1 to 16 nodes per layer.*
*
* This program was written as part of the SJSU CmpE200 semester project by Team .
*
* Team Members:
*
* Note: Compile with gcc -o bnn bnn.c -lm
* Run : ./bnn xor init verbose full 2 1 2 1 1
*
* ToDo:	Add functionality to visualize network [bnn name show] (?)
*	Expand SET for non-full networks (?)
*	Enfore Upper dimension for values (32) (?)
*	Check if filename has extension and, if not, append .cfg extension (?)
*	Check malloc for fail (return NULL) (!)
*	Free malloc's code (!)
*/

#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include "train_bnn.h"

#define FALSE 0
#define TRUE 1

#define BNN_INIT 0
#define BNN_TRAIN 1
#define BNN_RUN 2

#define DIM_INPUTS 0
#define DIM_OUTPUTS 1
#define DIM_NODES 2
#define DIM_LAYERS 3
#define DIM_SIGMOID 4

#define DEFAULT_INPUTS 2	// Input Nodes
#define DEFAULT_OUTPUTS 1	// Output Nodes
#define DEFAULT_NODES 2		// Nodes per hidden layer
#define DEFAULT_LAYERS 1	// Number of hidden layers
#define DEFAULT_SIGMOID 1
#define DEFAULT_BIAS 1
#define DEFAULT_WEIGHT 1
#define DELIM ","

struct Node {
	float value;
	int bias;
	int inputs;
	int weights;
	int sigmoid;
};

int show_help(char *, char *);
int bnn_init(char *, int, char **, int, int *, int);
int bnn_train(char *);
int bnn_run(int, char **, int, struct Node **, int, int *, int);
int bnn_load(char *, struct Node **, int *, int);
void bnn_show(int, struct Node **, int);

int main(int argc, char *argv[]) {

	//int counter;
	int ret_val = 0;
	int argpos = 1;
	int mode;
	int isVerbose = FALSE;
	int dim[5] = {0, 0, 0, 0, DEFAULT_SIGMOID};
	char *filename = NULL;
  printf("\n***** Binary Network Simulator ******\n\n");
	if (argc < 3)
	{
		ret_val = show_help("help","-a");
		return ret_val;
	}
	else if(argc < 4)
	{
		ret_val = show_help("help",argv[2]);
		return ret_val;
	}
	else
	{
		filename = argv[argpos];
		argpos++;

		if (strcmp(argv[argpos], "init") == 0)
		{
			mode = BNN_INIT;
		}
		else if (strcmp(argv[argpos], "train") == 0)
		{
			mode = BNN_TRAIN;
		}
		else if (strcmp(argv[argpos], "run") == 0)
		{
			mode = BNN_RUN;
		}
		else
		{
			ret_val = show_help(argv[1], argv[2]);
			return ret_val;
		}
		argpos++;

		if (strcmp(argv[argpos], "verbose") == 0) {
			isVerbose = TRUE;
			argpos++;
		}
	}

	if (BNN_INIT == mode) {
		ret_val = bnn_init(filename, argc, argv, argpos, dim, isVerbose);
		printf("\n BNN Initialized");
		return ret_val;
	}
	else if (BNN_TRAIN == mode) {
		ret_val = bnn_train(filename);
		return ret_val;
	}
	else {
		struct Node *bnn;
		int bnn_size = bnn_load(filename, &bnn, dim, isVerbose);

		if (isVerbose) {
			printf("\n[%s] Inputs: %d Outputs: %d Nodes: %d Layers: %d Sigmoid: %d\n\n", filename, dim[DIM_INPUTS], dim[DIM_OUTPUTS], dim[DIM_NODES], dim[DIM_LAYERS], dim[DIM_SIGMOID]);
		}

		ret_val = bnn_run(argc, argv, argpos, &bnn, bnn_size, dim, isVerbose);
		return ret_val;
	}

	return 0;
}

int bnn_init(char *target, int argc, char *argv[], int argpos, int *dim, int isVerbose) {

	FILE *fp;
	char *source = argv[argpos];
	int isFull = (strcmp(source, "full") == 0);
	int counter = 0;
	//int total_nodes = 0;
	//int non_outpout_nodes = 0;
	//char type;
	int number;
	//int max_nodes = 4 * (int)sizeof(int);

	dim[DIM_INPUTS] = DEFAULT_INPUTS;
	dim[DIM_OUTPUTS] = DEFAULT_OUTPUTS;
	dim[DIM_NODES] = DEFAULT_NODES;
	dim[DIM_LAYERS] = DEFAULT_LAYERS;
	dim[DIM_SIGMOID] = DEFAULT_SIGMOID;

	argpos++;

	if (isFull) {
		for (counter = argpos; counter < argc; counter++) {
			dim[counter - argpos] = atoi(argv[counter]);
			if (0 == dim[counter - argpos]) {
				printf("Bad input value %s.\n\n", argv[counter]);
				return EXIT_FAILURE;
			}
		}
	}
	else if (strcmp(source, "set") == 0) {
		//char *strings[5] = {"input nodes", "output nodes", "nodes per hidden layer", "hidden layers", "sigmoid function"};
		printf("\n[%s] Inputs: %d Outputs: %d Nodes: %d Layers: %d Sigmoid: %d\n\n", target, dim[DIM_INPUTS], dim[DIM_OUTPUTS], dim[DIM_NODES], dim[DIM_LAYERS], dim[DIM_SIGMOID]);
		}

	fp = fopen(target, "w+");
	if (!fp) {
		printf("Unable to write definitions file: %s\n\n", target);
		return EXIT_FAILURE;
	}

	fprintf(fp, "%d\n", dim[DIM_INPUTS] + dim[DIM_LAYERS] * dim[DIM_NODES]+ dim[DIM_OUTPUTS]);

	for (counter = 0; counter < dim[DIM_INPUTS]; counter++) {
		fprintf(fp, "i,0,0,0,0\n");
	}

	for (counter = 0; counter < (dim[DIM_LAYERS] * dim[DIM_NODES]); counter++) {
		number = (int)pow(2,(counter<dim[DIM_NODES]?dim[DIM_INPUTS]:dim[DIM_NODES])) - 1;
		fprintf(fp, "%d,%d,%d,%d,%d\n",(int)(counter/dim[DIM_NODES]),DEFAULT_BIAS,number,(DEFAULT_WEIGHT?number:0),dim[DIM_SIGMOID]);
	}

	for (counter = 0; counter < dim[DIM_OUTPUTS]; counter++) {
		number = (int)pow(2,dim[DIM_NODES]) - 1;
		fprintf(fp, "o,%d,%d,%d,%d\n",DEFAULT_BIAS,number,(DEFAULT_WEIGHT?number:0),dim[DIM_SIGMOID]);
	}
	fclose(fp);

	return 0;
}

int bnn_train(char *source)
{
	int status=0;

	status = bnn_train_network(source);

	return status;
}

int bnn_run(int argc, char *argv[], int argpos, struct Node **bnn, int bnn_size, int *dim, int isVerbose) {

	FILE *fp = NULL;
	char *source = NULL;
	char buffer[255];
	char *token;
	int args = 0;
	int counter = 0;
	int *values = (int *)malloc( dim[DIM_INPUTS] * sizeof(int) );

	if (!values) {
		printf("Insufficient memory.\n\n");
		return EXIT_FAILURE;
	}

	if ( strcmp(argv[argpos], "file") == 0 && argc > argpos) {
		source = argv[argpos + 1];
		if (source) {
			fp = fopen(source, "r");
		}

		if (!fp) {
			printf("Missing or invalid filename.\n\n");
			return EXIT_FAILURE;
		}

		fscanf(fp, "%s", buffer);
		fclose(fp);

		token = strtok(buffer, DELIM);
		while (token && args < dim[DIM_INPUTS]) {
			values[args] = atoi(token);
			args++;

			token = strtok(NULL, DELIM);
		}
	}
	else {
		while (argpos < argc && args< dim[DIM_INPUTS]) {
			values[args] = atoi(argv[argpos]);
			argpos++;
			args++;
		}
	}

	if (args < dim[DIM_INPUTS]) {
		printf("Insufficient arguments [%d] in %s (require %d).\n\n", args, (source?"file":"command line"), dim[DIM_INPUTS]);
		return EXIT_FAILURE;
	}

	// Set input node values
	for (args = 0; args < dim[DIM_INPUTS]; args++) {
		(*bnn)[args].value = values[args];
	}

// Move to separate STEP function that is also used in training:
	int currLayer = 0;
	int currNodes = 0;
	int currNode;
	int prevInputs;
	int inputs;
	//int input;
	float currValue;
	int currWeight;
	float sum;
	int counter2;

	if (isVerbose) bnn_show(0, bnn, bnn_size);
	for (currLayer = 0; currLayer <= dim[DIM_LAYERS]; currLayer++) {

		currNodes = (currLayer < dim[DIM_LAYERS])?dim[DIM_NODES]:dim[DIM_OUTPUTS];
		for (counter = 0; counter < currNodes; counter++) {
			currNode = dim[DIM_INPUTS] + currLayer * dim[DIM_NODES] + counter;
			inputs = (*bnn)[currNode].inputs;
			sum = 0;
			prevInputs = (0 == currLayer)?dim[DIM_INPUTS]:dim[DIM_NODES];
			// Shift through bits in input array
			for (counter2 = 0; counter2 < prevInputs; counter2++) {
				// and read values for all connected nodes from prev layer
				if ( inputs & (1 << counter2) ) {
					if (0 == currLayer) {
						currValue = (*bnn)[counter2].value;
					}
					else {
						currValue = (*bnn)[currLayer * dim[DIM_NODES] + counter2].value;
					}
					// Use '1:0' for binray, '1:-1' for bipolar network
					currWeight = (*bnn)[currNode].weights & (1 << counter2)?1:-1;
					sum += currValue * currWeight;
				}

			}
			sum += (*bnn)[currNode].bias;

			switch ((*bnn)[currNode].sigmoid) {
				case 1:
					currValue = (1 - exp(-sum)) / (1 + exp(-sum));
					break;
				default:
					printf("Invalid sigmoid function code [%d].\n\n", (*bnn)[currNode].sigmoid);
					return EXIT_FAILURE;
			}
			(*bnn)[currNode].value = currValue;
		}
		if (isVerbose) bnn_show(currLayer + 1, bnn, bnn_size);
	}

	printf("Result: ");
	for (counter = bnn_size- dim[DIM_OUTPUTS]; counter < bnn_size; counter++) {
		printf("%1.2f ", (*bnn)[counter].value);
	}
	printf("\n\n");

	free (values);
	return 0;
}

int bnn_load(char *source, struct Node **bnn, int *dim, int isVerbose) {

	FILE *fp;
	int bnn_size;


		fp = fopen(source, "r");
		char buffer[255];
		char *token;

		int currLayer;
		int prevLayer = -1;

		if (!fp) {
			printf("Invalid filename: %s\n\n", source);
			return EXIT_FAILURE;
		}

		fscanf(fp, "%s", buffer);
		bnn_size = atoi(buffer);
		if (0 == bnn_size) {
			printf("Invalid input file (%s).\n\n", source);
			return EXIT_FAILURE;
		}
		*bnn = (struct Node *)malloc (bnn_size * sizeof (struct Node));

		int row = 0;
		while ( fscanf(fp, "%s", buffer) != EOF ) {
			token = strtok(buffer, DELIM);

			int col = 0;
			while (token) {

				if (0 == col) {

					(*bnn)[row].value = 0;

					if ('i' == token[0]) {
						dim[DIM_INPUTS]++;
					}
					else if ('o' == token[0]) {
						dim[DIM_OUTPUTS]++;
					}
					else {
						currLayer = atoi(token);
						if (0 == currLayer) {
							dim[DIM_NODES]++;
						}
						if (prevLayer < currLayer) {
							dim[DIM_LAYERS]++;
							prevLayer = currLayer;
						}
					}
				}
				else {
					if (1 == col) {
						(*bnn)[row].bias = atoi(token);
					}
					else if (2 == col) {
						(*bnn)[row].inputs = atoi(token);
					}
					else if (3 == col) {
						(*bnn)[row].weights = atoi(token);
					}
					else if (4 == col) {
						(*bnn)[row].sigmoid = atoi(token);
					}
				}
				token = strtok(NULL, DELIM);
				col++;
			}
			row++;
		}
		fclose(fp);

	return bnn_size;
}

void bnn_show(int layer, struct Node **bnn, int bnn_size) {

	int counter;

	if (layer)
		printf("Calculating layer %d:\n", layer);
    else
        printf("\n\n");

	for (counter = 0; counter < bnn_size; counter++) {
		printf("%1.2f,%d,%d,%d,%d\t", (*bnn)[counter].value, (*bnn)[counter].bias, (*bnn)[counter].inputs, (*bnn)[counter].weights, (*bnn)[counter].sigmoid);
		if ( 4 == (counter % 5) )
			printf("\n");
	}
}


int show_help(char *arg1, char *arg2)
{

	int isHelp = ( arg1 && strcmp(arg1, "help") == 0 );

	if (isHelp)
	{

	  printf("Usage: ./bnn [help|xor] [-a|train|run|full|sources|set|values|sigmoid|example]\n\n");

    if(strcmp(arg2, "-a") == 0)
    {
	  	printf("         full [inputs] [output] [nodes] [layers] [sigmoid]\n\n");
			printf("         full   = flag to create a fully connected network\n");
			printf("         inputs = number of input nodes (Default: 2)\n");
			printf("         outputs = number of output nodes (Default: 1)\n");
			printf("         nodes   = number of nodes in each hidden layer (Default: 2)\n");
			printf("         layers  = number of layers including input and output (Default: 3)\n");
			printf("         sigmoid = function to calculate node value (Default: 0)\n");
			printf("         (Type 'bnn help sigmoid' for a list of available functions.)\n");
			printf("         (Type 'bnn help full' for 'full' file format.)\n\n");
			printf("         name of a file containing network definitions in format\n");
			printf("         NODE_COUNT\\n[i|number|o],bias,inputs,weights,sigmoid\\n(...)\n\n");
			printf("         inputs and weights are binary encoded, so '1,5,15,3,0\\n1,-3,15,8,0'\n");
			printf("         sets node 1 and 2 in hidden layer 1 connected to four input nodes\n");
			printf("         (1111=15) and uses weights -1,-1,-1,1 (0001=8, 0=-1, 1=1)\n\n");
		}
		else if ( strcmp(arg2, "train") == 0 )
		{
			printf("        action 'train': Set up the network using a file of known inputs and\n");
			printf("                        outputs.\n\n");
			printf("        File format: i(1,1),...,i(1,max_input):o(1,1),...,o(1,max_output\n");
			printf("                     (...)\n");
			printf("                     i(n,1),...,i(n,max_output):o(n,1),...,o(n,max_output)\n\n");
		}
		else if ( strcmp(arg2, "run") == 0 )
		{
			printf("        action 'run': Evaluate a set of input values using a trained network.\n\n");
			printf("        Input values can be supplied via a file or on the command line. To\n");
			printf("        use a file, use arguments 'file FILENAME'.\n\n");

		}
		else if ( strcmp(arg2, "full") == 0 )
		{
			printf("        arg 'full': Initialize a fully connected neural network. In a\n");
			printf("                    fully connected neural network, each layer n (n>1) node\n");
			printf("                    is connected to each layer n-1 node.\n\n");
			printf("        Input values can be supplied via a file or on the command line. If a\n");
			printf("        file is used, supply the filename instead of 'full' and use the file\n");
			printf("        format: full,inputs,outputs,nodes,layers,sigmoid\n\n");
			printf("        Node: If some values are omitted, defaults are used.\n\n");
			printf("        For command line option or default values, type 'bnn help init'.\n\n");
		}
		else if ( strcmp(arg2, "source") == 0 )
		{
			printf("       arg 'source': Use a file to provide input values. File format differ\n");
			printf("                     based on the action type. In some cases, a file can be\n");
			printf("                     used in conjunction with values. In this case, values\n");
			printf("                     overwrite the parameters provided in the file.\n\n");
			printf("       For file formats, type 'bnn help action'.\n\n");
		}
		else if ( strcmp(arg2, "set") == 0 )
		{
			printf("       arg 'set': Prompt for input values during action\n\n");
		}
		else if ( strcmp(arg2, "values") == 0 )
		{
			printf("       arg 'values': Provide input values on the command line. In some cases,\n");
			printf("                     values can be used in conjunction with a file. In this\n");
			printf("                     case, values overwrite the parameters provided in the file.\n\n");
			printf("       For arguments, type 'bnn help action'.\n\n");
		}
		else if ( strcmp(arg2, "sigmoid") == 0 )
		{
			printf("       A sigmoid is a function used in calculating the value of a neural network\n");
			printf("       node. Different sigmoid functions can be selected. Valid choices are:\n\n");
			printf("        1 = 1/x where x=\n\n");
		}
		else if ( strcmp(arg2, "example") == 0 )
		{
			printf("       Example Commands:bnn xor [action] [verbose] arg\n\n");
			printf("       Setup XOR network: ./bnn xor init verbose full 2 1 2 1 1\n");
			printf("       Run XOR network:   ./bnn xor run verbose 1 1\n\n");
		}

  }
	return ((isHelp)? EXIT_SUCCESS : EXIT_FAILURE);
}
