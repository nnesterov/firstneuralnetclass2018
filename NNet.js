
//neural net architecture:
//
//
//
//A neural network is a directed graph of computation.
//Nodes represent functions performed on data.
//
//
//Node types:
//
//  Input node
//    - for this node, input and output are identical 
//
//  Bias node
//    - an input node whose output is always a fixed value
//
//  Function node
//    - this node performs a differentiable function over data
//    - this is the workhorse node.
//    - it can tell us how to compute the output from the input via a function
//    - it can tell us the derivative of the output with respect to the input when using that function
//
//  Loss (output) node
//    - this node receives as input the final output of the neural network
//    - the user sets the data_output of this node before backprop.
//    - this node describes how far away the final output of the neural network
//      is from the user set data_output 
//    
//



var NNetNodeTypes = { 
  BIAS : 0,
  FXN : 1,
  INPUT: 2,
  LOSS : 3,    
};






function NNetInput(sz) {
  this.type = NNetNodeTypes.INPUT;

  this.output = [];//this is actually the input to the network  
  this.dErr_dOutput = [];
  
  for(var i = 0; i < sz; i++) {
    this.output[i] = 0;
    this.dErr_dOutput[i] = 0;
  }

  this.clone = function() { 
    return new NNetInput(sz);
  }
};




function NNetBias(v) { 
  this.type = NNetNodeTypes.BIAS;
  this.output = [v];
  //this makes the code a bit easier, since we don't care about edge cases in backprop
  this.dErr_dOutput = [0];

  this.clone = function() { 
    return new NNetBias(this.output[0]);
  };
};




function NNetFxnBasic(type) { 
  this.type = NNetNodeTypes.FXN;

  this.input = [0];
  this.output = [0];
  this.dErr_dInput = [0];
  this.dErr_dOutput = [0];


  //setup the type of activation function
  if(type == 0) { 
    //sigmoid
    
    this.fxn = function(x) { 
      return 2.0 / (1.0 + Math.exp(-x)) - 1.0;
    };
    this.dfxn = function(x) { 
      var z = 1.0 / (1.0 + Math.exp(-x));
      return 2*z*(1.0-z);
    };


  } else if(type == 1)  {
    //logit 

    this.fxn = function(x) { 
      if(x > 300) {
        //NOTE: this is very important, otherwise we get infinity very quickly since:
        //  Math.exp(800) = inf
        //  Math.log(inf) = inf
        //  even though Math.log(Math.exp(800)) = 800
        return x;
      }
      return Math.log(1 + Math.exp(x));
    };

    this.dfxn = function(x) { 
      return 1.0 / (1.0 + Math.exp(-x));
    };


  } else { 
    this.fxn = function(x) { 
      return x;
    }
    this.dfxn = function(x) { 
      return 1;
    };
  }

  this.forw = function() { 
    //console.log("input:", this.input[0]);
    this.output[0] = this.fxn(this.input[0]);
    //console.log("output:", this.output[0]);
    //if(isNaN(this.output[0]))
    //  console.log("got nan from ",this.input[0]);
  
  };

  this.back = function() { 
    this.dErr_dInput[0] = this.dErr_dOutput[0] * this.dfxn(this.input[0]);
  };
  
  this.clone = function() { 
    return new NNetFxnBasic(type);
  };

};




function NNetFxnNetWrapper(net, share_sub_weights) { 
  //TODO: used for wrapping multiple networks
  
  this.type = NNetNodeTypes.FXN;
  
  this.input = [];
  this.output = [];
  this.dErr_dInput = [];
  this.dErr_dOutput = [];

  //we can use this to control wether or not this subnetwork
  //accumulates backprop values for it's edges 
  this.bp_rate = 1;

  for(var i = 0; i < net.inputCt(); i++) { 
    this.input[i] = 0;
    this.dErr_dInput[i] = 0;
  }
  for(var i = 0; i < net.outputCt(); i++) { 
    this.output[i] = 0;
    this.dErr_dOutput[i] = 0;
  }

  this.forw = function() { 
    net.distributeInput(this.input);
    net.forwProp();
    net.gatherOutput(this.output);
  };

  this.back = function() { 
    net.distribute_dErr_dOutput(this.dErr_dOutput);
    net.backProp(this.bp_rate);
    net.gather_dErr_dInput(this.dErr_dInput);
  };
  
  this.clone = function() { 
    return new NNetFxnNetWrapper(net.clone(share_sub_weights), share_sub_weights);
  };
};





function NNetSquaredLoss(sz) { 
  this.type = NNetNodeTypes.LOSS;

  //the network's final output goes here 
  this.model_output = [];
  for(var i = 0; i < sz; i++)
    this.model_output[i] = 0;

  //this makes things easier, basically the input to this layer is the same as the model output
  //NOTE: this is a reference
  this.input = this.model_output;
    
  this.data_output = [];
  for(var i = 0; i < sz; i++)
    this.data_output[i] = 0;
  
  //what the derivative of the error with respect to the network's final output goes here 
  //this tells us changing which parameters improves the fit the most 
  this.dErr_dModelOutput = [];
  for(var i = 0; i < sz; i++)
    this.dErr_dModelOutput[i] = 0;

  //this is a reference, makes writing backpropogation easier
  this.dErr_dInput = this.dErr_dModelOutput;

  //how good/bad is the current fit is
  this.loss = function() {     
    var err = 0;
    for(var i = 0; i < sz; i++) { 
      //console.log("i", i);
      //console.log("model out", this.model_output[i], "data", this.data_output[i]);
      err += (this.model_output[i]-this.data_output[i])*(this.model_output[i]-this.data_output[i]);
    }
    return err;
  };

  this.forw = function() { 
    //no final transformation
  };

  //this initializes the derivative of the error wrt the network's final output so we 
  //can start backprop
  this.back = function() { 
    for(var i = 0; i < sz; i++) {
      //console.log("Model out", this.model_output[i], "data", this.data_output[i]);
      this.dErr_dModelOutput[i] = 2*(this.model_output[i] - this.data_output[i]);
    }
  };

  this.clone = function() { 
    return new NNetSquaredLoss(sz);
  }
};



function NNetSoftMaxCrossEntropyLoss(sz) { 
  this.type = NNetNodeTypes.LOSS;
  
  //the network's final output goes here 
  this.model_output = [];
  for(var i = 0; i < sz; i++)
    this.model_output[i] = 0;
  
  this.input = [];
  for(var i = 0; i < sz; i++)
    this.input[i] = 0;

  //what we want the output to be  
  this.data_output = [];
  for(var i = 0; i < sz; i++)
    this.data_output[i] = 0;
  
  //what the derivative of the error with respect to the network's final output goes here 
  //this tells us changing which parameters improves the fit the most 
  this.dErr_dModelOutput = [];
  for(var i = 0; i < sz; i++)
    this.dErr_dModelOutput[i] = 0;

  this.dErr_dInput = this.dErr_dModelOutput;
  //this.dErr_dInput = [];
  //for(var i = 0; i < sz; i++)
  //  this.dErr_dInput[i] = 0;

  //how good/bad is the current fit is
  this.loss = function() {     
    //here each input is transformed:
    //
    // x_i -> y_i
    // y_i =  exp(x_i) / [sum_over_all_k]{ exp(x_k)  }
    //
    //  of the bat:
    //    this is really easy to overflow, but if we multiply top and 
    //    bottom by exp(-q) we can see that we can adjust the x's
    //
    // this forces each of our "n" inputs into a range of [0,1] and they
    // also sum to 1. this is a natural way to represent the probability 
    // of choosing a value from 0 to (n-1). The probability of choosing 
    // category i is simply y_i. 
    //
    // what if the observed value was not a category, but a pdf over categories?
    // now instead of simply maximizing y_i, we would be maximizing p_0*y_0 +  p_1*y_1 + ...
    //
    // we will want to maximize the -log of this value, but in order to do so it's good to 
    // take a look at what d/x_j (y_k)  
    //
    // if j==k then it's
    //
    //  exp(x_j) / sum - exp(x_j) / sum^2 * d/dx_j(sum) 
    //  exp(x_j) / sum - exp(x_j) / sum^2 * exp(x_j) 
    //  y_j - y_j^2 = y_j*(1-y_j) 
    //  -(y_j-1)*y_j
    //
    //
    // if j!=k then it's:
    //
    //  0 / sum  - y_k * y_j
    //  -y_k*y_j
    //
    // now if we take d/dx_j of -log(prob) it's the same as 
    // sum_over_q dy_q/dx_j * (d/dy_q of -log(prob)) 
    //  
    //
    // d/d_yj [ -log(p0 * y0 + p1*y1 + ...) ]
    //
    // is -pj/(p0*y0 + p1*y1 ... )
    //  
    // now we do the product and sum over q
    //
    // (y0yjp0 + y1yjp1 + ... yj(yj-1)pj + ...) /(p0*y0 + p1*y1 + ... ) 
    //
    // yj*(y0p0 + y1p1 + ... + yjpj + ...  - pj) /(p0*y0 + p1*y1 + ... ) 
    //
    // yj*(1 - pj/(p0*y0 + p1*y1 + ...))      [EQ0]
    //
    //  NOTE: if pj is 1 and pw w!=j is 0 then this simplifies to 
    // 
    // yj*(1 - 1/yj) = yj - 1         [EQ1]
    //  
    //  NOTE: if pj is 0 then this simplifies to 
    //
    //  yj*(1 - 0/") = yj            [EQ2]
    //
    //  So in the case of an exact category we use EQ1 and EQ2, but if we have uncertain categories we can use EQ0
    //
   

    var prob = 0;
    for(var i = 0; i < sz; i++) { 
      prob += this.model_output[i] * this.data_output[i];
    }
    return -Math.log(prob);
  };

  this.forw = function() { 
    
    //we will scale down the max values to prevent exp(x) from turning into infinity
    var max = -Number.MAX_VALUE;    
    for(var i = 0; i < sz; i++) { 
      if(this.input[i] > max)
        max = this.input[i];
    }

    var sum = 0;   
    //console.log("-----------");
    for(var i = 0; i < sz; i++) { 
      this.model_output[i] = Math.exp(this.input[i] - max);
      sum += this.model_output[i];
      //console.log("E", this.input[i], this.model_output[i], sum);
    }
    //console.log("SUM:",sum);
    for(var i = 0; i < sz; i++) {
      this.model_output[i] /= sum;
      //console.log("R", i,this.model_output[i]);
    } 
  };

  //this initializes the derivative of the error wrt the network's final output so we 
  //can start backprop
  this.back = function() { 
    
    if(true) { 

      //when data output is only 0 or 1 
      for(var i = 0; i < sz; i++) { 
        //console.log("DERR", i, this.model_output[i], this.data_output[i]);
        if(this.data_output[i] == 1.0) { 
          this.dErr_dModelOutput[i] = this.model_output[i] - 1;
          //console.log("SM", this.dErr_dModelOutput[i]);
        } else {  
          this.dErr_dModelOutput[i] = this.model_output[i];
          //console.log("DIFF", this.dErr_dModelOutput[i]);
        }
      }

    } else { 
    
      //TODO: test
      //when we try to match pdf of output, note this could be numerically sketchy 
      var wsum = 0;
      for(var i = 0; i < sz; i++) 
        wsum += this.model_output[i] * this.data_output[i];
      //NOTE: model_output will rarely ever be 0, but data_output is likely to be 0 or 1
      for(var i = 0; i < sz; i++) 
        this.dErr_dModelOutput[i] = this.model_output[i]*(1 - this.data_output[i] / wsum );

    }
  };

  this.clone = function() { 
    return new NNetSoftMaxCrossEntropyLoss(sz);
  };

};


function NNetGassianLoss(mixtures, data_sz) { 
  
  //here we have mixtures (count) independant gaussians that 
  //are mixed together to produce a multidimensional output 
  //for data_sz. this is useful for modelling data where there 
  //is more than one possible output. 
  //
  //
  //there are three ways to do this:
  //simplest:
  //  each output is completely independant mixture of 1d gaussian
  //
  //
  //simple:
  //  there's only 1 mixture, each mixture component is a data_sz dimensional gassian, but the variance is diagonal (or identity scaled) 
  //  this seems good when we have a LOT of seperate modes. could we pass the generated value through another network to get arbitrary
  //  distributions effectively? in essence, could we use this as a function layer? seems related to VAO.
  //
  //complex:
  //  the input is the the cov as the cholseky decomposition of the inverse of the cov (see research papers)
  //  so we have a mixture of a few N-dimensional gaussians with off diagonal elements in their covariance matrixes
  //
  //TODO: this is broken for now



  this.input = [];
  this.model_output = input;
  
  for(var i = 0; i < mixtures*data_sz*3; i++) { 
    //perc, mean, var for each mixture
    this.input[i] = 0;
  }

  this.data_output = [];
  for(var i = 0; i < data_sz; i++) 
    this.data_output[i] = 0;

  this.dErr_dModelOutput = [];
  for(var i = 0; i < sz; i++)
    this.dErr_dModelOutput[i] = 0;
  
  this.dErr_dInput = this.dErr_dModelOutput;

  this.loss = function() { 
    //return the negative gaussian log probability for this mixture
    var ret = 0;
    for(var i = 0; i < data_sz; i++) { 

      var prob0 = 0;
      for(var k = 0; k < mixtures; k++) { 
        var perc = this.input[3*k];
        var mean = this.input[3*k+1];
        var variance = Math.exp(this.input[3*k+2]);
        
        //NOTE: e^-800 = 0, we would prefer to be working in log probability mode the whole time 
        //see C++ HMM code for dealing with adding log probabilities. 
        var x = mean - this.data_output[i];
        prob0 += Math.exp(-x*x / 2.0 / variance) / Math.sqrt(2*Math.PI*variance);
        //log_prob = -0.5 * Math.log(2.0 * variance * Math.PI) - x*x/(2.0 * variance);
      }  

      ret += -Math.log(prob0);
    }
    return ret;
  };

  this.forw = function() { 
    //nothing here really
  };

  this.back = function() { 
    //TODO
    //
    //  if we have 
    //    
    //    p(x) = [sum over q] p_q e^(-x^2/2v_q)  / sqrt(2v_qPi)
    //
    //
    //  we want to maximize -log(p(x)) wrt x 
    //
    //    
    //    this ends up being 
    //
    //      -x/v_q
    //
    //    easy!
    //
    //  now we want to maximize wrt p_q
    //
    //    [e^(-x^2/2v_q)/sqrt(2v_qPi)] /p(x)
    //  
    //
    //  now we want to maximize wrt v_q 
    //
    //    ...this is a mess... TODO
    //
    //
  };

};



//this is used by WGAN and when concatenating several networks
function NNetExternalLoss(sz) { 
  this.type = NNetNodeTypes.LOSS;
  
  this.input = [];
  for(var i = 0; i < sz; i++)
    this.input[i] = 0;
  this.model_output = this.input;
  
  //set this externally 
  this.dErr_dModelOutput = [0];
  for(var i = 0; i < sz; i++)
    this.dErr_dModelOutput[i] = 0;
  this.dErr_dInput = this.dErr_dModelOutput;

  this.loss = function() {return undefined; };
  this.forw = function() { };
  this.back = function() { };

};

function NNetExternalSigmoidLoss(sz) { 
  this.type = NNetNodeTypes.LOSS;
  
  this.input = [];
  for(var i = 0; i < sz; i++)
    this.input[i] = 0;
  this.model_output = [];
  for(var i = 0; i < sz; i++)
    this.model_output[i] = 0;
  
  //set this externally 
  this.dErr_dModelOutput = [0];
  for(var i = 0; i < sz; i++)
    this.dErr_dModelOutput[i] = 0;
  this.dErr_dInput = [];
  for(var i = 0; i < sz; i++)
    this.dErr_dInput[i] = 0;

  this.loss = function() {return undefined; };

  this.forw = function() { 
    for(var i = 0; i < sz; i++) 
      this.model_output[i] = 1 / (Math.exp(-this.input[i]) + 1);
  };

  this.back = function() { 
    for(var i = 0; i < sz; i++) 
      this.dErr_dInput[i] = this.dErr_dOutput[i] * this.model_output[i] * (1 - this.model_output[i]);
  };


};






function NNet() {

  var graph = new DGraph.Graph();

  var loss_nodes = [];
  
  var input_nodes = []; 

  //keep track of the order of evaluation of nodes in the network 
  var fford = undefined;

  this.rawLossNodes = function() { 
    return loss_nodes;
  };

  this.rawInputNodes = function() { 
    return input_nodes;
  };

  this.rawGraph = function() { 
    return graph;
  };

  this.nodeCt = function(){ 
    return graph.nodeCt();
  };

  this.edgeCt = function() { 
    return graph.edgeCt();
  };

  this.addNode = function(node) { 
    fford = undefined;
    usernode = graph.addNode();
    usernode.data = node;
    if(node.type == NNetNodeTypes.LOSS)
      loss_nodes.push(node);   
    else if(node.type == NNetNodeTypes.INPUT)
      input_nodes.push(node);   
    node.id = usernode.id; //just to help with logging for now...
    return usernode;
  };

  this.inputCt = function() { 
    var ret = 0;
    //NOTE: the output of the nodes marked as input is where the input goes 
    for(var i = 0; i < input_nodes.length; i++)
      ret += input_nodes[i].output.length;
    return ret;
  };

  this.distributeInput = function(data_vec) { 
    var idx = 0;
    var offs = 0;
    for(var i = 0; i < data_vec.length; i++) { 
      if(offs >= input_nodes[idx].output.length) {
        idx++;
        offs = 0;
      }
      //NOTE: the output of the nodes marked as input is where the input goes 
      input_nodes[idx].output[offs] = data_vec[i];
      offs++;
    }
  };
  
  this.gather_dErr_dInput = function(data_vec) { 
    var idx = 0;
    var offs = 0;
    for(var i = 0; i < data_vec.length; i++) { 
      //NOTE: input nodes only have output values, so we look at their dErr_dOutput
      if(offs >= input_nodes[idx].dErr_dOutput.length) {
        idx++;
        offs = 0;
      }
      data_vec[i] = input_nodes[idx].dErr_dOutput[offs];
      offs++;
    }
  };


  this.outputCt = function() { 
    var ret = 0;
    for(var i = 0; i < loss_nodes.length; i++) 
      ret += loss_nodes[i].model_output.length;
    return ret;
  };

  this.gatherOutput = function(data_vec) { 
    var idx = 0;
    var offs = 0;
    for(var i = 0; i < data_vec.length; i++) { 
      if(offs >= loss_nodes[idx].model_output.length) {
        idx++;
        offs = 0;
      }
      data_vec[i] = loss_nodes[idx].model_output[offs];
      offs++;
    }
  };

  this.distribute_dErr_dOutput = function(data_vec) { 
    var idx = 0;
    var offs = 0;
    for(var i = 0; i < data_vec.length; i++) { 
      if(offs >= loss_nodes[idx].dErr_dModelOutput.length) {
        idx++;
        offs = 0;
      }
      loss_nodes[idx].dErr_dInput[offs] = data_vec[i];
      offs++;
    }
  };


  this.clone = function(share_edges) { 
    var ret = new NNet(); 
    graph.eachNode(function(node) {
      ret.addNode(node.clone());
    });
    graph.eachEdge(function(edge) { 
      
      if(edge.data.direct) {
        ret.connectByIdsDirect(edge.s.id, edge.d.id);
      } else { 
        var e = ret.connectByIds(edge.s.id, edge.d.id);
        e.data.weights = undefined;
        e.data.dErr_dWeights = undefined;
        if(share_edges) { 
          e.data.weights = edge.data.weights;
          e.data.dErr_dWeights = edge.data.dErr_dWeights;
        }
      }

    });
    return ret;
  }

  //id0 -> id1
  this.connectByIds = function(id0, id1) { 
    fford = undefined;
    var n0 = graph.findNodeById(id0);
    var n1 = graph.findNodeById(id1);
    if(!n0)
      console.log("ERROR, no node with id", id0);
    if(!n1)
      console.log("ERROR, no node with id", id1);

    var from_ct = n0.data.output.length;
    var to_ct = n1.data.input.length;

    var edge = graph.connectByNodes(n0,n1);
  
    //initialize the edge data
    edge.data = {}

    //the actual weights
    edge.data.weights = [];

    //we will also track the dErr_dWeight just to make life easies
    edge.data.dErr_dWeights = [];

    for(var i = 0; i < from_ct*to_ct; i++) { 
      edge.data.weights.push(0);
      edge.data.dErr_dWeights.push(0);
    }

    //save these values for easier use later
    edge.data.from_ct = from_ct;
    edge.data.to_ct = to_ct;

    //NOTE: we define the weight for going "from output i" from node n0 -> "to input j" in node n1
    //as the weights at index i*to_ct + j ... we could choose an alternate definition but all that
    //matters is we _stick_ to the definition.

    return edge;
  };

  this.connectByIdsDirect = function(id0, id1) { 
    fford = undefined;
    var n0 = graph.findNodeById(id0);
    var n1 = graph.findNodeById(id1);
    
    var from_ct = n0.data.output.length;
    var to_ct = n1.data.input.length;

    var edge = graph.connectByNodes(n0,n1);
  
    //initialize the edge data
    edge.data = {}
    edge.data.direct = true;

    return edge;

  };

  


  this.addNormalNoiseToWeights = function(rand, sdev) { 
    if(sdev === undefined)
      sdev = 1;
    graph.eachEdge(function(edge) { 
      if(!edge.data.direct) { 
        for(var i = 0; i < edge.data.weights.length; i++) 
          edge.data.weights[i] += rand.normal() * sdev;
      }
    });  
  };


  this.maxNormIncomingWeights = function(r) { 

    //we want to sum over all the weights going into a given input idx at some node
    //this means we need to sum over all the parent nodes since the inputs from multiple
    //parents are summed together into the input slots. 

    graph.eachNode(function(node) { 
      var nnet_node = node.data;  
      if(nnet_node.input) { 

        var to_ct = nnet_node.input.length;
        
        var mags = [];
        for(var j = 0; j < to_ct; j++)
          mags[j] = 0;
        
        //first calculate total magnitude going into each location j
        
        node.eachParentNodeDataEdgeData(function(pdata, edata) { 
       
          if(!edata.direct) {    
            var from_ct = pdata.output.length;
            for(var j = 0; j < to_ct; j++) { 
              for(var i = 0; i < from_ct; i++) { 
                mags[j] += edata.weights[i*to_ct + j] * edata.weights[i*to_ct + j];
              }
            }
          }

        });

        //convert mag^2 to mag
        for(var j = 0; j < mags.length; j++)
          mags[j] = Math.sqrt(mags[j]);

        //for any mags that are over the limit r, normalize and scale them to distance r
        node.eachParentNodeDataEdgeData(function(pdata, edata) { 
          
          if(!edata.direct) { 
            var from_ct = pdata.output.length;
            for(var j = 0; j < to_ct; j++) { 
              if(mags[j] > r) { 
                for(var i = 0; i < from_ct; i++) { 
                  edata.weights[i*to_ct + j] *= r/mags[j];
                }
              }
            }
          }

        });


      }
    });
  };  
  
  this.clampGradient = function(sc) { 
    graph.eachEdge(function(edge) { 
      if(!edge.data.direct) { 
        for(var i = 0; i < edge.data.weights.length; i++) {
          if(Math.abs(edge.data.dErr_dWeights[i]) > sc) 
            edge.data.dErr_dWeights[i] = sc*Math.sign(edge.data.dErr_dWeights[i]);
        }
      }
    });
  };
  
  this.clampWeights = function(r) { 
    graph.eachEdge(function(edge) { 
      if(!edge.data.direct) { 
        for(var i = 0; i < edge.data.weights.length; i++) {
          if(edge.data.weights[i] > r)
            edge.data.weights[i] = r;
          else if(edge.data.weights[i] < -r)
            edge.data.weights[i] = -r;
        }
      }
    });
  };

  this.decayWeights = function(sc) { 
    graph.eachEdge(function(edge) { 
      if(!edge.data.direct) { 
        for(var i = 0; i < edge.data.weights.length; i++) 
          edge.data.weights[i] *= sc;
      }
    });
  };

  //this.exctr = 0;

  this.forwProp = function() { 
    if(!fford)
      fford = graph.FFOrder();

    for(var f = 0; f < fford.length; f++) { 
      //console.log("F", f);
      var graph_node = fford[f];
      var nnet_node = graph_node.data;
      //console.log("f---------------", f, "out0", nnet_node.output ? nnet_node.output[0] : "none", "out1", nnet_node.output ? nnet_node.output[1] : "none" );
      if(nnet_node.type == NNetNodeTypes.FXN || nnet_node.type == NNetNodeTypes.LOSS) { 
        
        var to_ct = nnet_node.input.length;

        if(graph_node.hasParents()) { 
          for(var j = 0; j < nnet_node.input.length; j++)
            nnet_node.input[j] = 0;
        }

        //begin accumulating input from parent nodes
        graph_node.eachParentNodeDataEdgeData(function(pdata, edata) { 
          
            
          var from_ct = pdata.output.length;
          
          if(edata.direct) {
            for(var i = 0; i < from_ct && i < to_ct; i++) {
              nnet_node.input[i] += pdata.output[i];
            }
          } else { 

            
            for(var i = 0; i < from_ct; i++) { 
              for(var j = 0; j < to_ct; j++) { 
                nnet_node.input[j] += edata.weights[i*to_ct + j] * pdata.output[i];
                //console.log("node input", j, "accum to", nnet_node.input[j], "from", pdata.output[i], "parentidx", i);
                //console.log("ijv", i,j,nnet_node.input[j], pdata.output[i], edata.weights[i*to_ct + j]);
              }
            }

          }
        });
        
        nnet_node.forw();
        //console.log("forw for", nnet_node);
        //console.log("node output 0 is ", nnet_node.output ? nnet_node.output[0] : "none");
      }
    }
    //this.exctr++;
    //if(this.exctr == 3)
    //  exit(0);

  };

  this.loss = function() { 
    var ret = 0;
    for(var i = 0; i < loss_nodes.length; i++) { 
      //console.log("loss node", i, ret);
      ret += loss_nodes[i].loss();
      //console.log("loss became", ret);
    }
    return ret;  
  };

  this.analyticBackProp = function(r, save_seperately) {
    if(!r)
      r = 0.000001;
    
    var self = this;
    graph.eachEdge(function(edge) { 
      
      if(!edge.data.direct) {   
        if(save_seperately && !edge.data.dErr_dWeights_analytic) {
          edge.data.dErr_dWeights_analytic = [];
          for(var i = 0; i < edge.data.weights.length; i++)
            edge.data.dErr_dWeights_analytic[i] = 0;
        }

        for(var i = 0; i < edge.data.weights.length; i++) {
          
          var savew = edge.data.weights[i];
          edge.data.weights[i] =  savew - r;
          self.forwProp();
          var loss0 = self.loss();
          edge.data.weights[i] = savew + r;
          self.forwProp();
          var loss1 = self.loss();
          edge.data.weights[i] = savew;


          if(save_seperately)
            edge.data.dErr_dWeights_analytic[i] += (loss1 - loss0)/(2*r);
          else
            edge.data.dErr_dWeights[i] += (loss1 - loss0)/(2*r);
        }
      }
    
    });


  };


  this.backProp = function(rate) { 
    if(rate === undefined)
      rate = 1;

    if(!fford)
      fford = graph.FFOrder();

    for(var f= fford.length-1; f >= 0; f--) { 
      var graph_node = fford[f];
      var nnet_node = graph_node.data;
      
      if(nnet_node.type == NNetNodeTypes.LOSS) {
        nnet_node.back();
      } 
      
      if(graph_node.hasChildren()) { // && nnet_node.type != NNetNodeTypes.LOSS) { 
        for(var i = 0; i < nnet_node.dErr_dOutput.length; i++) 
          nnet_node.dErr_dOutput[i] = 0;
      }
      //console.log("self", graph_node.id, 
      //    "vin", nnet_node.input ? nnet_node.input[0] : "", 
      //    "vout", nnet_node.output ? nnet_node.output[0] : "", 
      //    "dErr_dIn", nnet_node.dErr_dInput ? nnet_node.dErr_dInput[0] : "");

      if(nnet_node.type == NNetNodeTypes.FXN || nnet_node.type == NNetNodeTypes.BIAS || nnet_node.type == NNetNodeTypes.INPUT) { 
        
        var from_ct = nnet_node.output.length;
        graph_node.eachChildNodeDataEdgeData(function(cdata, edata) { 
          //now at each node we know the dErr_dInput
          //the dErr_dW for edge going from "output i" to "input j" is output[i] * dErr_dInput[j]

          //the dErr_dOutput for the parent node's output i the sum over all input j's of w*dErr_dInput[j]
          //this is simply because we can also write the output as function of the inputs. 
          //this is because the the network output can be written as a function of the parents input's and
          //then derived wrt the inputs using "total derivative".
          //NOTE: if partial derivatives of two parent's could cause an error in the derivative if say
          //the the two parents are multiplied later on. this requires being near an inflection point, 
          //of which there are usually a finite quantity.
          
          var to_ct = cdata.input.length;
    
          if(edata.direct) { 
            for(var i = 0; i < from_ct && i < to_ct; i++) { 
              nnet_node.dErr_dOutput[i] += cdata.dErr_dInput[i];              
            }
          } else { 

            //console.log("edge", edata.id);
            for(var i = 0; i < from_ct; i++) {
              for(var j = 0; j < to_ct; j++) { 
                //console.log("pre", edata.dErr_dWeights[i*to_ct + j], "w", edata.weights[i*to_ct + j]);
                edata.dErr_dWeights[i*to_ct + j] += nnet_node.output[i] * cdata.dErr_dInput[j] * rate;
                nnet_node.dErr_dOutput[i] += edata.weights[i*to_ct + j] * cdata.dErr_dInput[j];
                //console.log("ij _ par",i,j, cdata.dErr_dInput[j], nnet_node.dErr_dOutput[i]);
              }
            }

          }
        });
        
        if(nnet_node.type == NNetNodeTypes.FXN) {
          nnet_node.back();
        } 
        //console.log("self", graph_node.id, "==========dErr_dIn", nnet_node.dErr_dInput ? nnet_node.dErr_dInput[0] : "", "dErr_dOut", nnet_node.dErr_dOutput ? nnet_node.dErr_dOutput[0] : "");

      }//end looking at node tha might have children 

    }

  };
  
 
  //once we've done one pass of forwProp and 
  //on pass of backProp we have accumulated 
  //the desired change in weights. apply at any 
  //time. decay should be less than 1 and sc should
  //be greater than 0.   
  //this.v = 0;
  this.learnWithMomentum = function(sc, decay) { 
    if(!decay)
      decay = 0;
          
    //if(this.v == 2)
    //  console.log("v2", sc, decay);
    //var self= this;

    graph.eachEdge(function(edge) { 

      if(!edge.data.direct) { 
        for(var i = 0; i < edge.data.weights.length; i++) {
          //if(self.v == 2)
          //  console.log("DW",i,edge.data.dErr_dWeights[i]);
          //console.log(edge.data.dErr_dWeights[i]);
          edge.data.weights[i] -= edge.data.dErr_dWeights[i] * sc;
          edge.data.dErr_dWeights[i] *= decay;
        }
      }
    
    });
  };
  
  this.log = function() { 
    if(!fford)
      fford = graph.FFOrder();
    for(var i = 0; i < fford.length; i++) { 
      console.log("NODE "+i+ " type "+ fford[i].data.type + " id "+ fford[i].data.id);
      console.log("input0 " + (fford[i].data.input ? fford[i].data.input[0] : " ")); 
      fford[i].eachParentNodeDataEdgeData(function(pdata, edata) { 
        console.log("parent id ",  pdata.id, "edge0", edata.weights[0], edata.dErr_dWeights[0]);
      });
      
    }
  }
  
  ////do this before forwProp+backProp training 
  //this.dropoutStage0 = function(perc_keep, rand) { 
  //  if(!rand)
  //    rand = Math;
  //  this.perc_keep = perc_keep;

  //  graph.eachEdge(function(edge) { 

  //    //if we had saved weights before load them
  //    if(edge.data.dropped_weights)
  //      edge.data.weights = edge.data.dropped_weights;
  //    
  //    //clear the saved weight array 
  //    edge.data.dropped_weights = [];

  //    //drop weights and save them with some probability
  //    for(var i = 0; i < edge.data.weights.length; i++) {
  //      if(rand.random() > perc_keep) { 
  //        edge.data.dropped_weights[i] = edge.data.weights[i];
  //        edge.data.weights[i] = 0;
  //        edge.data.dErr_dWeights[i] = 0;
  //      }

  //    }
  //  
  //  });
  //};

  ////do this right before updating the weights
  ////this forces the 0-value weights to stay zero
  //this.dropoutStage1 = function() { 
  //  graph.eachEdge(function(edge) { 

  //    for(var i = 0; i < edge.data.weights.length; i++) {
  //      if(edge.data.dropped_weights[i] !== undefined) { 
  //        edge.data.weights[i] = 0;
  //        edge.data.dErr_dWeights[i] = 0;
  //      }
  //    }
  //  });
  //};

  ////do this after training, to see the regularized output 
  //this.dropoutStage2 = function() {
  //  var self = this; 
  //  graph.eachEdge(function(edge) { 

  //    for(var i = 0; i < edge.data.weights.length; i++) {
  //      if(edge.data.dropped_weights[i] !== undefined) 
  //        edge.data.weights[i] = edge.data.dropped_weights[i];
  //      else
  //        edge.data.dropped_weights[i] = edge.data.weights[i];
  //      edge.data.weights[i] *= self.perc_keep;
  //    }
  //  });
  //};

    
};



//TODO: make a training class
//
//function NNetTrainerV0(net, data) {
//
//
//};


//some GAN based networks
//GAN works by having one network generate a data sample
//from randomly distributed latent vectors. the discriminator
//then assigns a probability to the generated sample being 
//fake vs real. 
//
//
//the discriminator is trained on multiple samples, some are generated and some are real
//at this point the discriminator basically wants to learn to tell real apart from fake.
//
//
//the generator is trained to maximize the discrimator's probability
//of saying the current generated sample is real.

function UnrolledGAN(net_disc, net_gen) { 


};



//WGAN:
//
//in theory let's say we have some samples from the distribution 
//and some samples we generated. here the difference is defined by
//applying a fxn f to each sample from the dist and summing, then
//subtracting that same sum from the gen'd data. f is limited to 
//having slope 1. if the gen'd data is far away from the dist, 
//the difference will be huge. if the gen'd data and dist are the
//same the diff will be 0. We simultaneosly train a net to calc the
//the function f by maximizing f(data) - f(gen) while also bounding it
//by enforcing that (d/dInput[f(...)])^2 is less than 1 along random points
//between the sampled points. 

function WGAN(net_critic, net_gen) {
  
  //original paper: https://arxiv.org/pdf/1701.07875.pdf
  //improved weight penalty: https://arxiv.org/pdf/1704.00028.pdf

  this.net_critic = net_critic;
  this.net_gen = net_gen;
  
  //net_gen takes L inputs and procuces K outputs 
  //net_critic take K inputs and produces 1 output

  this.latent_ct = this.net_gen.inputCt();
  this.latent_row = [];
  for(var i = 0; i < this.latent_ct; i++)
    this.latent_row[i] = 0;

  this.gen_ct = this.net_gen.outputCt();
  console.log("OC", this.gen_ct);
  this.gen_row = [];
  for(var i = 0; i < this.gen_ct; i++)
    this.gen_row[i] = 0;   

  if(this.gen_ct != this.net_critic.inputCt()) {
    console.log("ERROR! Critic takes", this.net_critic.inputCt(), "inputs, but generator produces", this.gen_ct);
  }
  if(this.net_critic.outputCt() != 1) { 
    console.log("ERROR!, Critic must gen 1 output, but gens", this.net_critic.outputCt());
  }

  this.combo_net = new NNet()
  
  var n0 = new NNetFxnNetWrapper(this.net_gen);
  n0.bp_rate = 1.0;

  var n1 = new NNetFxnNetWrapper(this.net_critic);
  n1.bp_rate = 0.0;
  
  this.combo_net.addNode(new NNetInput(this.latent_ct));
  this.combo_net.addNode(n0);
  this.combo_net.addNode(n1);
  this.combo_net.addNode(new NNetExternalLoss(1));
  this.combo_net.connectByIdsDirect(0,1);
  this.combo_net.connectByIdsDirect(1,2);
  this.combo_net.connectByIdsDirect(2,3);
}

var exc = 0;
WGAN.prototype.batchTrain = function(data_src, batch_size, n_critic, rate, rand) { 
  //n_critic = 1;
  //batch_size = 1;
  
  //var max_w = 0.2;
  var max_w = 0.2;

  this.net_critic.clampWeights(max_w);
  //console.log("DDD"); 
  for(var nc = 0; nc < n_critic+35; nc++) {
    var delta = 0;
    for(var b = 0; b < batch_size; b++) { 

      var data_row = data_src.sampleRow();  
      
      for(var j = 0; j < this.latent_ct; j++)
        this.latent_row[j] = rand.normal();
      //console.log("samp", this.latent_row[0], this.latent_row[1]);

      this.net_gen.distributeInput(this.latent_row);
      this.net_gen.forwProp();
      this.net_gen.gatherOutput(this.gen_row);
      //console.log("gen", this.gen_row[0], this.gen_row[1]);

    
      //we want to maximize this.net_critic(data_row) - this.net_critic(this.net_gen(latent_row))       

      this.net_critic.distributeInput(data_row);
      this.net_critic.forwProp();
      this.net_critic.rawLossNodes()[0].dErr_dInput[0] = -1; //causes maximizing 
      this.net_critic.backProp();
      var f0 = this.net_critic.rawLossNodes()[0].input[0];
      
      //console.log(this.gen_row);  
      this.net_critic.distributeInput(this.gen_row);
      this.net_critic.forwProp();
      this.net_critic.rawLossNodes()[0].dErr_dInput[0] = 1; //causes minimizng for this input 
      this.net_critic.backProp();
      var f1 = this.net_critic.rawLossNodes()[0].input[0];
      
      delta += f0-f1;
   
    }
    //console.log("delta", delta);;

    //console.log("NC");
    this.net_critic.learnWithMomentum(rate/batch_size/200, 0.0);  
    this.net_critic.clampWeights(max_w);
  }
  exc++;
  //if(exc == 2)
  //  exit(0);

  //console.log("++++++++++++++++++++++++++++++++++++++++++++++++++");
  for(var k = 0; k < 1; k++){
    for(var b = 0; b < batch_size; b++) { 
      for(var j = 0; j < this.latent_ct; j++)
        this.latent_row[j] = rand.normal();

      this.combo_net.distributeInput(this.latent_row);
      this.combo_net.forwProp();
      this.combo_net.rawLossNodes()[0].dErr_dInput[0] = -1;
      this.combo_net.backProp();
      
      console.log("GEN_NET");
      this.net_gen.log();
      console.log("CRITIC_NET");
      this.net_critic.log();
      
      //console.log(this.combo_net.rawLossNodes()[0].input[0]);
      //console.log(this.net_critic.rawLossNodes()[0].input[0]);
    }
    //console.log("NG");    
    this.net_gen.learnWithMomentum(rate/batch_size, 0.0);  
    //this.net_critic.learnWithMomentum(0.00000000001, 0.0);  
  }
  //exit(0);

};

WGAN.prototype.sample = function(rand) { 
  for(var j = 0; j < this.latent_ct; j++)
    this.latent_row[j] = rand.normal();
  this.net_gen.distributeInput(this.latent_row);
  this.net_gen.forwProp();
  this.net_gen.gatherOutput(this.gen_row);
  //console.log(this.gen_row);
  return this.gen_row;
};

WGAN.prototype.transform = function(row) { 
  this.net_gen.distributeInput(row);
  this.net_gen.forwProp();
  this.net_gen.gatherOutput(this.gen_row);
  //console.log(this.gen_row);
  return this.gen_row;
}













