Class Network
{
	Data Members:
		double eta;
        int batch_size;
        int epochs;
        int layer_sizes[];          //array of layer sizes
        int L;                      //number of layers i.e. sizeof(layer_sizes)
        vector<matrix> weights;
        vector<matrix> biases;

        vector<matrix> activations;
        vector<matrix> weighted_inputs;
  		
		vector<matrix> delta; 
        vector<matrix> sum_nabla_b;
        vector<matrix> sum_nabla_w;
		
		vector<int> test data indices;
		vector<int>	mini batch indices;
		
		int number correct;
		
		string training_data_filename;		//consider truncating "filename". unnecessary and unwieldy
        string expected values_filename;
		
	Methods:
		Constructor(requires eta, batch size, layer sizes, file names, and epochs provided by the ReadIn static member function)
        {
            //Generate L, which is the number of layers. This is given by sizeof(layer sizes). 
            //Weights, biases, activations, weighted inputs, nabla_w, nabla_b resized to size L.
            //Everything but the activation vector will have an effective size of L-1, as their first element will be left unused.
            Looping an index i from 1 to L
                resize the ith element of the weights vector 		to be layer sizes at i by layer sizes at i-1, fill w/ rand #s
                resize the ith element of the sum_nabla_w vector 	to be layer sizes at i by layer sizes at i-1, fill with zeros
				
                resize the ith element of the biases vector 		to be layer sizes at i by 1, fill with random numbers (normal dist)
                resize the ith element of the sum_nabla_b vector 	to be layer sizes at i by 1, fill with zeros
				resize the ith element of the delta vector 			to be layer sizes at i by 1, fill with zeros
				
                resize the ith element of the activations vector 	to be layer sizes at i by 1, fill with zeros
                resize the ith element of the weightedinputs vector to be layer sizes at i by 1, fill with zeros	
			
            resize the 0th member of the activations vector to be layer sizes at 0 by 1, fill with zeros
			
			resize mini batch indices to be mini batch size
			
            //Populate ith element in the array of weights vector with pseudorandom numbers, mean 0, st. dev. 1/Sqrt[layer sizes at i-1]
            Set object's hyperparameters to values passed to constructor
        }
		
		static Network ReadIn()
         {
             Prompt the user to choose between an existing network file or to make a new network
             If the user opts to choose a file
                 Read in file name
                 Input validation (TBD)
                 Extract Network architecture and hyperparameters from file to temporary variables
             Else
                 Prompt the user to enter details through the command prompt, storing them in temporary variables
             Call class constructor, passing temporary variables as arguments
         }
	
		//forwardPass() is a function that sets all activation values for a single test data input. 
        //It needs the layer of activations at 0 to be assigned values from the test data.
        void forwardPass()
        {
            Looping an index i from 1 to L
                activations at i is equal to the sigmoid function of the weighted inputs at i,
                    where the weighted inputs at i is set to be equal to (weights at i * activations at i-1) + biases at i
        }
		
		//backProp() is a function that calculates the nabla_b and nabla_w vectors.
        //backProp requires sigmoid_prime function, cost derivative function, activations and weighted inputs already been set
        void backProp(matrix of expected output for a given training input (expected val) )
        {
           	compare expected val to activations[L]
			if they are the same, increment number correct 
			
           	delta at L = cost_prime(expected output for the given training sample, activations at L) hadamarded with (sigmoid_prime of weighted inputs at L)
           	sum_nabla_b at L += delta at L
 			sum_nabla_w at L += (delta at L) * (activations at L-1 transposed)
        	Looping an index i from L-1 to 1 (the remaining layers)
                delta at i = ((weights at i + 1 transposed)*(delta at i + 1)) hadamarded with (sigmoid_prime of weighted inputs of at i)

                sum_nabla_b at i += delta at i 
              	sum_nabla_w at i += (delta at i) * (activations at i-1 transposed)
        }
		
		//SGD's function is to complete a forward pass and backward pass on a mini batch and compute the average nabla_b and 
        //  nabla_w vectors over the whole batch. it then updates the weights and biases using the nabla vectors and the learning
        //  rate eta
        
        //SGD requires an int batch size to iterate over the vector mini batch indices which hold indices of the test data
        //  with respect to the mini batch to be iterated
        //  it is assumed that these indices are randomly generated and won't repeat a test element for the same epoch but that
        //  is outside SGD's scope and it won't need to worry about that.
		//SGD also requires test data indices to have been resized appropriately
	
        void SGD(int batch size) // may not be the same as mini batch size due to leftover data at end of test data
        {
            //compute average errors for the batch
            Looping index i through mini batch indices (0 to batch size)
            // its important to use batch size and not mini batch size here
                assign activations[0] with test data (test data[mini batch indices[i]])
                //haven't yet thought through how test data will be accessed. not familiar with the ifstream operations
                perform forward pass 
                perform backward pass // sums of nablas computed here
            
            //apply error information to weights and biases
            update()
                
            zero out avg_nabla vectors
        }
		
		void update()
		{
			Looping index i from 1 to L
                weights[i] -= eta * (sum_nabla_w[i])
                biases[i]  -= eta * (sum_nabla_b[i])
		}
		
		//Train brings all the other methods together and trains the network. It loops through the epochs separating out 
        //  mini batches from the test data and running sgd on all mini batches for each epoch. Train doesnt actually access the files
        void Train()
        {
            open test data files
            int test data size = size of test data
            test data indices resized to test data size and initialized:
                such that test data indices at i = i    // these values but will be randomized later
				
			Loop index i through test data indices
				test data indices[i] = i
			
            int sgd calls = (epochs)/(mini batch size)  // number of times sgd is called to finish one epoch 
            int leftover = epochs remainder(%) mini batch size     // to see if there's a small mini batch left in the data
            
            Looping index i from 0 to epochs
                Fisher Yates shuffle (test data indices)
				int batch size = mini batch size
				
                Looping index j from 0 to sgd calls
                    Looping through index k from 0 to (mini batch size - 1)
                        mini batch indices[k] = test data indices [(j*mini batch size) + k)]
                    SGD(batch size)
                if (leftover > 0) //one more sgd call is there was some leftover data that cant fit into a regualr mini batch
                    Looping through index k from 0 to (leftover - 1)
                        mini batch[k]  = test data indices [test data size-k-1]
                    SGD(leftover)
				cout << "Efficiency at epoch: " << i << " = " << (number correct) / (test data size)
        }
		
		void Classify (verification data)
		{
			Loop index i from 0 to size of (verification data)
				set activation[0] = verification data[i]
				feedforward()
				int biggest = 0 // index of the biggest activation value
				Looping index j from 0 to size of (activations[L])
					if activations[L][j] > activations[L][biggest] then biggest = j
				print out classification of biggest // 
		}
}
