

//the basic building block of a neural network
//opts can be used to specify if this is an input,output neuron (vs regular)
function BasicNeuron(opts) { 

  //we receive input from the parents 
  this.parentEdges = [];

  //we send our output to the children
  this.childEdges = [];

  //this is the entire neuron state 
  this.input = 0;
  this.output = 0;

  //the 2 values below represent proportionally how much the loss changes when 
  //the corresponding neuronal state changes by a tiny amount
  this.dLoss_dOutput = 0;//<- we don't need to store this...
  this.dLoss_dInput = 0;

  //input and output nodes get special treatment
  if(opts && opts.is_output) { 
    //use "training_output" during backpropogation 
    this.is_output_node = true;
    //user needs to set the training_output value externally 
    this.training_output = 0;
  }
  if(opts && opts.is_input) { 
    this.is_input_node = true;
    //the user needs to externally set the input for input nodes
  }


  //use this during forward prop 
  this.activationFunction = function(x) { 
    return Math.atan(x);
  };
  
  
  //use this for backprop
  this.derivativeActivationFunction = function(x) { 
    return 1/(1 + x*x);
  };

  
  //use this to connect nodes together  
  this.addChild = function(child_node, initial_weight) { 
    if(this.is_output_node) 
      throw "cannot add child node to output node"      
  
    var edge = { 
        source:this, 
        dest:child_node,
        weight : initial_weight,
        dLoss_dWeight : 0,
      };

    this.childEdges.push( edge );
    child_node.parentEdges.push( edge ) ;
  };


  //goal here is to calculate the output
  this.forwardPropogate = function() { 

    if(this.is_input_node) { 
      //if we are an input node, we simply use whatever was presented to us as our input 
    } else { 
      //we are not the input node, so...
      //gather input from parents, and apply our activation function to it
      this.input = 0;
      for(var i = 0; i < this.parentEdges.length; i++) { 
        this.input += this.parentEdges[i].source.output * this.parentEdges[i].weight;
      }
    }
      

    //apply the nonlinearity!!!
    this.output = this.activationFunction(this.input);    

  };
  

  //goal here is to calculate how loss is affected by input,output, and weights
  this.backPropogate = function() { 
    this.dLoss_dOutput = 0;

    if(this.is_output_node) {

      //if we're an output node, we're responsible for the initial loss value 
      //that will be back-propogated.
      this.dLoss_dOutput =  2.0*(this.output - this.training_output);
    
    } else { 
      
      //we're not an output node, so accumulate dLoss_dOutput using children
      for(var i = 0; i < this.childEdges.length; i++) { 
        this.dLoss_dOutput += this.childEdges[i].dest.dLoss_dInput * this.childEdges[i].weight;
      }
      //also we can calculate how much each child edge affects the loss
      for(var i = 0; i < this.childEdges.length; i++) {
        this.childEdges[i].dLoss_dWeight = this.output * this.childEdges[i].dest.dLoss_dInput;
      } 
    }

    //use the chain rul to convert from dLoss_dOutput to dLoss_dInput
    this.dLoss_dInput = this.dLoss_dOutput * this.derivativeActivationFunction(this.input);
  };


};


function NeuralNetwork() { 

  //the order of nodes is important, you cannot place 
  //children nodes before their parents in here! 
  //externally populate this array (yes, hacky)
  this.nodes = [];
  
  this.forwardPropogate = function() { 
    for(var i = 0; i < this.nodes.length; i++)
      this.nodes[i].forwardPropogate();
  };
  

  this.backPropogate = function() { 
    //NOTE: backprop order is REVERSE of forwardprop order!
    for(var i = this.nodes.length-1; i >= 0; i--)
      this.nodes[i].backPropogate();
  };


  this.trainWeights = function(rate) { 
    for(var i = 0; i < this.nodes.length; i++) { 
      var edges = this.nodes[i].childEdges;
      for(var j = 0; j < edges.length; j++)
        edges[j].weight -= edges[j].dLoss_dWeight * rate;
    }
  };

};





















