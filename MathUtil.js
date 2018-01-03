

var MathUtil = {
 
  //get the log probability of a gaussian 
  //the pdf of a gaussian is 1/sqrt(2*pi*var) * e^(  -(x-avg)^2 / (2 var )   )
  //taking the log of that gives -0.5*log(2*pi*var) - (x-avg)^2/ (2 var )
  logGaussianPdf : function(x, variance) { 
    return -0.5 * Math.log(2.0 * variance * Math.PI) - x*x/(2.0 * variance);  
  },

  gaussianPdf : function(x, variance) { 
    return Math.exp(-x*x / 2.0 / variance) / Math.sqrt(2*Math.PI*variance);
  },

  //TODO: when multiple independant gaussians are summed the means and variances sum together
  //when we take the product of multiple gaussian pdfs, the pdfs change in a predictable way
  //too: we want that formula here.

  //laplacian distribution is:
  //1/(2scale) * exp( - |x-avg| / scale )
  //note that the variance is 2*scale^2
  logLaplacianPdf : function(x, scale) { 
    return -Math.log(2*scale) - Math.abs(x)/scale;
  },

  laplacianPdf : function(x, scale) { 
    return Math.exp(-Math.abs(x)/scale) / 2 / scale;
  },

  //cdf is 1/2b * exp(x/scale)  if x < 0
  //  x = Math.log(2b*y) *scale
  //cdf is 1 - 1/2b * exp(-x/scale) if x >= 0
  //  x = -Math.log(2b*(1-y)) *scale
  //TODO: test this
  sampleLaplacian : function(rand, scale) { 
    var y = rand.random();
    if(y < 0.5)
      return Math.log(2*scale*y);
    return -Math.log(2*scale*(1-y));
  },

  //classic super fat tailed distribution
  //pdf is 1/(PI*scale) * scale^2 / ( (x-median)^2 + scale^2 )
  //pdf is 1/PI * scale / ( (x-median)^2 + scale^2 )
  //luckily samplying it directly is pretty easy (todo)
  logCauchyPdf : function(x, scale) { 
    return -Math.log(Math.PI) + Math.log(scale) - Math.log(x*x + scale*scale);
  },

  cauchyPdf : function(x, scale) { 
    return scale / Math.PI / (x*x + scale*scale);
  },

  //cdf is 1/pi * atan(x / scale)  + 0.5
  //inverting ew get  tan(pi*(y-0.5))*scale
  sampleCauchy : function(rand, scale) {
    if(!scale)
      scale = 1.0; 
    return Math.tan(Math.PI*(rand.random()-0.5)) * scale;
  },

  //shape must be greater than 0, usually keep scale around 1 or so
  //starts at scale (which also must be greater than 0)
  //pdf is shape*scale^shape / x^(shape+1)
  logParetoPdf : function(x, scale, shape) { 
    if(x < scale)
      return Number.NEGATIVE_INFINITY
    return Math.log(shape) + shape*Math.log(scale) - Math.log(x)*(shape+1);
  },

  paretoPdf : function(x, scale, shape) { 
    if(x < scale)
      return 0.0;
    return shape*Math.pow(scale,shape)/Math.pow(x,shape+1);
  },

  //TODO: test, what if rand returns 0 or Number.MIN_VALUE? 
  //cdf is 1 - (scale/x)^shape
  //inverting we get x = scale/Math.pow(1-y, 1/shape)
  //when sampling 1-y or y doesn't make a difference
  samplePareto(rand, scale, shape) { 
    return scale/Math.pow(y, 1/shape);
  },

  gamma : function(x) { 
    var A = [
      0.99999999999980993, 
      676.5203681218851, 
      -1259.1392167224028,
      771.32342877765313, 
      -176.61502916214059, 
      12.507343278686905,
      -0.13857109526572012, 
      9.9843695780195716e-6, 
      1.5056327351493116e-7
    ];

    if(x<0.5)
      return Math.PI/(Math.sin(x*Math.PI) * MathUtil.gamma(1.0 - x));

    x -= 1.0;
    var a = A[0];
    var t = x + 7.5;
    for (var i = 1; i < A.length; i++)
      a += A[i]/(x + i); 
    return Math.sqrt(2*Math.PI)*Math.pow(t, x+0.5)*Math.exp(-t)*a;
  },

  logGamma : function(x) { 
    var A = [
      76.18009172947146,    
      -86.50532032941677,
      24.01409824083091,    
      -1.231739572450155,
      0.1208650973866179e-2,
      -0.5395239384953e-5
    ];
   
    x -= 1.0;
    var tmp = x + 5.5;
    tmp -= (x+0.5)*Math.log(tmp);
    var ser=1.000000000190015;
    for(var j=0; j<6; j++) {
      x += 1.0;
      ser += A[j]/x;
    }
    return -tmp+Math.log(2.5066282746310005*ser);
  },

  //log of the pdf of the gamma distribution
  logGammaPdf : function(x, shape, scale) { 
    if(x <= 0.0)
      return Number.NEGATIVE_INFINITY;
    return Math.log(x)*(shape-1) - x/scale - MathUtil.logGamma(shape) - Math.log(scale)*shape;
  },

  //pdf of the gamma distribution
  gammaPdf : function(x, shape, scale) { 
    if(x <= 0.0)
      return 0;
    return Math.pow(x,shape-1)*Math.exp(-x/scale)/MathUtil.gamma(shape)/Math.pow(scale, shape);
  }, 

  //TODO: sampleGamma
  
  //TODO: betaPdf
  //TODO: logBetaPdf
  //TODO: sampleBeta

  //clamps a single value into a range [-maxv,maxv]
  clampV1 : function(arg, maxv) {
    if(arg > maxv)
      return maxv;
    if(arg < -maxv)
      return -maxv;
    return arg;
  },
  
  clampMinMaxV1 : function(arg, minv, maxv) {
    if(arg > maxv)
      return maxv;
    if(arg < minv)
      return minv;
    return arg;
  },

  sigmoidV1 : function(arg) { 
    return 1.0/(1.0 + Math.exp(-arg));
  },
  
  //this treats u64[1] as bits 63:32 and u64[0] as bits 31:0 of two uint32's 
  //and performs a zero-filling rightshift on them. 
  rshiftZF64 : function( u, amt) { 
    if(amt > 0) {
      if(amt >= 32) {
        u[0] = u[1] >>> (amt-32);
        u[1] = 0;
      } else { 
        u[0] = (u[0] >>> amt) | (u[1] << (32-amt));
        u[1] >>>= amt;
      }
    } else if(amt < 0) { 
      amt *= -1;
      if(amt >= 32) { 
        u[1] = u[0] << (amt - 32);
        u[0] = 0;
      } else { 
        u[1] = (u[1] << amt) | (u[0] >>> (32-amt));
        u[0] <<= amt;
      }
    }
  },
  xorAndrshiftZF64 : function( u, amt) { 
    if(amt > 0) {
      if(amt >= 32) {
        u[0] ^= u[1] >>> (amt-32);
      } else { 
        u[0] ^= (u[0] >>> amt) | (u[1] << (32-amt));
        u[1] ^= u[1] >>> amt;
      }
    } else if(amt < 0) { 
      amt *= -1;
      if(amt >= 32) { 
        u[1] ^= u[0] << (amt - 32);
      } else { 
        u[1] ^= (u[1] << amt) | (u[0] >>> (32-amt));
        u[0] ^= u[0] << amt;
      }
    }
  },
  multU32 : function(u32a, u32b) { 
    var hia = u32a >>> 16;
    var loa = (u32a << 16) >>> 16; 
    var hib = u32b >>> 16;
    var lob = (u32b << 16) >>> 16; 
    //the product can overflow by at most 1 bit and the sum can also overflow 1 bit
    //performing the shifts prevents overflow from happening 
    return (((hia * lob + loa * hib) << 16) + (loa * lob)) >>> 0; 
  },
  multU64 : function(u64a, u64b, res) { 
    var a0 = (u64a[0] << 16) >>> 16;
    var a1 = u64a[0] >>> 16;
    var a2 = (u64a[1] << 16) >>> 16;
    var a3 = u64a[1] >>> 16;
    var b0 = (u64b[0] << 16) >>> 16;
    var b1 = u64b[0] >>> 16;
    var b2 = (u64b[1] << 16) >>> 16;
    var b3 = u64b[1] >>> 16;
    
      
    var shift16 = 1<<16;
    var shift32 = shift16*shift16;//2^32 cannot be made using shifts in js
    //this sum can overflow 32 bits, but not 48
    var tmp = (((a1 * b0 + a0 * b1) * shift16) + (a0 * b0)); 
    //cast it back down to 32 bits
    res[0] = tmp >>> 0;
    //measure the overflow in res[0], shift it down 32 bits
    //we have about 16 bits of overflow due to float64's being able to 
    //store up to exact 48 bit integers, but this is getting quite hacky 
    res[1] = (((tmp - res[0])/shift32)>>>0) + a1*b1 + a2*b0 + a0*b2 + ((a1*b2)<<16) + ((a2*b1)<<16) + ((a3*b0)<<16) + ((a0*b3)<<16); 
    res[1] >>>= 0; 
  },



  //TODO: move this somewhere else, random number math library 
  //this implements a random number generator that has a significantly 
  //smaller "state" than the mersenne twister, so it's good for sending 
  //accross the network. 
  extXORShift32 : function(tmp) { 
    tmp ^= tmp << 13;
    tmp ^= tmp >>> 17;
    tmp ^= tmp << 5;
    tmp >>> 0;
    return tmp;
  },

  //solves c0, c1 in 
  //x00 * c0 + x01 * c1 = y0
  //x10 * c0 + x11 * c1 = y1
  solve2x2 : function(x00, x01, x10, x11, y0, y1) { 
    //if x00 and x10 are both zero, then det() is 0.0!
    //however, we can still find a least squares solution, save that for another time.
    c1 = (y0*x10 - y1*x00) / (x01*x10 - x11*x00); 
    if(Math.abs(x00) >= Math.abs(x10))
      c0 = (y0 - x01*c1)/x00;
    else
      c0 = (y1 - x11*c1)/x10;
    return [c0,c1];
  },


  //very basic matrix stuff, not optimized at all
  //treates an array of arrays as an MxN matrix:
  // [[a11, a12, ... a1N],
  //  [a21, ...
  //   ...
  //  [aM1, ...      aMN]]
  //TODO: dont use .w and .h properties of arrays
  Mtx : {
    
    Zeros : function(h,w) { 
      if(h === undefined) 
        console.log("ERROR!, unspecified matrix height");
      if(w === undefined)
        w = h;
      var ret = new Array(h);
      for(var i = 0; i < h; i++) { 
        ret[i] = new Array(w);
        for(var j = 0; j < w; j++)
          ret[i][j] = 0;
      }
      ret.h = h;
      ret.w = w;
      return ret;
    },

    clone : function(a) { 
      var ret = MathUtil.Mtx.Zeros(a.h, a.w);
      for(var i = 0; i < a.h; i++) { 
        for(var j = 0; j < a.w; j++) { 
          ret[i][j] = a[i][j];
        }
      }
      return ret;
    },

    padSquareZeros : function(a) { 
      if(a.w > a.h) { 
        for(var i = a.h; i < a.w; i++) { 
          var add = [];
          for(var j = 0; j < a.w; j++)
            add.push(0);
          a.push(add);
        }
        a.h = a.w;
      } else if(a.h > a.w) { 
        for(var i = 0; i < a.h; i++) { 
          for(var j = a.w; j < a.h; j++) { 
            a[i].push(0.0);
          }
        }
        a.w = a.h
      }      
      return a;
    },
    
    padSquareIdentity : function(a) { 
      if(a.w > a.h) { 
        for(var i = a.h; i < a.w; i++) { 
          var add = [];
          for(var j = 0; j < a.w; j++)
            add.push(i==j ? 1.0 : 0.0);
          a.push(add);
        }
        a.h = a.w;
      } else if(a.h > a.w) { 
        for(var i = 0; i < a.h; i++) { 
          for(var j = a.w; j < a.h; j++) { 
            a[i].push(i == j ? 1.0 : 0.0);
          }
        }
        a.w = a.h
      }      
      return a;
    },
    
    Identity : function(h, w) { 
      var ret = MathUtil.Mtx.Zeros(h,w);
      for(var i = 0; i < ret.h && i < ret.w; i++)
        ret[i][i] = 1.0;
      return ret;
    },
    
    Identity : function(h, w) { 
      var ret = MathUtil.Mtx.Zeros(h,w);
      for(var i = 0; i < ret.h && i < ret.w; i++)
        ret[i][i] = 1.0;
      return ret;
    },

    AllOnes : function(h,w) { 
      var ret = MathUtil.Mtx.Zeros(h,w);
      for(var i = 0; i < ret.h; i++) { 
        for(var j = 0; j < ret.w; j++) { 
          ret[i][j] = 1;
        }
      }
      return ret; 
    },

    SampleWishart : function(rand, dof, dim, scale) { 
      if(!scale)
        scale = 1.0;
      if(typeof(scale) == 'number')
        scale = Math.sqrt(scale);
      else {
        console.log("ERROR, matrix scale paramter not supported yet");
        scale = 1;  
      }

      var samples = [];
      for(var i = 0; i < dim; i++) { 
        var addee = [];
        for(var j = 0; j < dof; j++) {
          addee.push(rand.normal() * scale);
        }
        samples.push(addee);
      }

      var scatter = MathUtil.Mtx.Zeros(dim,dim);
      for(var i = 0; i < dim; i++) { 
        for(var j = i; j < dim; j++) { 
          var prod = 0;
          for(var k = 0; k < dof; k++) 
            prod += samples[i][k]*samples[j][k];
          scatter[i][j] = prod;
          scatter[j][i] = prod;
        }
      }
      return scatter;
    },
    
    //this is one way to generate a sample for a covariance matrix,
    //however it has a few flaws. people like to use this because 
    //it's conjugacy (it's posterior has a similar form as it's prior
    //after observing data points and incorporating their covariance estimates). 
    SampleInvWishart : function(rand, dof, dim, scale) { 
      var r = MathUtil.Mtx.SampleWishart(rand, dof, dim, scale);
      return MathUtil.Mtx.jacobiInv(r);
    },

    //create a matrix filled with Math.random() values 
    JSRandom : function(h,w) { 
      var ret = MathUtil.Mtx.Zeros(h,w);
      for(var i = 0; i < ret.h; i++) { 
        for(var j = 0; j < ret.w; j++) { 
          ret[i][j] = Math.random();
        }
      }
      return ret; 
    },

    //scale the matrix in place, return it for convenience
    scaleEq : function(a,sc) { 
      for(var i = 0; i < a.h; i++) { 
        for(var j = 0; j < a.w; j++) { 
          a[i][j] *= sc;
        }
      }
      return a;
    },
    
    product : function(a,b) { 
      if(a.w != b.h) { 
        console.log("ERROR! matrix product dims mismatch!!!");
        console.log(" ...", a.h,"x",a.w, "  *  ", b.h,"x",b.w);
      }
      var ret = MathUtil.Mtx.Zeros(a.h, b.w);
      for(var i = 0; i < a.h; i++) { 
        for(var j = 0; j < b.w; j++) { 
          var prod = 0.0;
          for(var k = 0; k < a.w; k++) {
            prod += a[i][k] * b[k][j];
          }
          ret[i][j] = prod;
        }
      }
      return ret;
    },

    //multiply A*B*C and return the result, saves on keystrokes 
    product3 : function(a,b,c) {
      return MathUtil.Mtx.product(MathUtil.Mtx.product(a,b),c);
    },

    //does not modify "a", return a transposed copy of "a"
    transpose : function(a) { 
      var ret = MathUtil.Mtx.Zeros(a.w, a.h);
      for(var i = 0; i < a.h; i++) { 
        for(var j = 0; j < a.w; j++) { 
          ret[j][i] = a[i][j];
        }
      }
      return ret;
    },

    //average a[i][j] and a[j][i] off-diagonal entries to force symmetry
    symmetrize : function(a) { 
      for(var i = 0; i < a.h && i < a.w; i++) { 
        for(var j = i+1; j < a.h && j < a.w; j++) { 
          var tmp = (a[i][j] + a[j][i])/2.0;
          a[i][j] = tmp;
          a[j][i] = tmp;
        }
      }
      return a;
    },

    //return the sum of a[i][j]^2 for all i,j
    norm2 : function(a) { 
      var r = 0;
      for(var i = 0; i < a.h; i++) { 
        for(var j = 0; j < a.w; j++) { 
          r += a[i][j] * a[i][j];
        }
      }
      return r;
    },

    //return the square root of (the sum of a[i][j]^2 for all i,j)
    norm : function(a) { 
      return Math.sqrt(MathUtil.Mtx.norm2(a));
    },

    //normalize the the j-th column of matrix a in place, return the norm of the column
    colNormalize : function(a,j) { 
      var sumsq = 0;
      for(var i = 0; i < a.h; i++) 
        sumsq += a[i][j] * a[i][j];
      var ret = Math.sqrt(sumsq);
      if(ret <= 0)
        return 0;
      for(var i = 0; i < a.h; i++) 
        a[i][j] /= ret;
      return ret;
    },

    sumScaled : function(a, a_sc, b, b_sc) { 
      if(a.w != b.w || a.h != b.h) { 
        console.log("ERROR! matrix sumScaled dims mismatch!!!");
      }
      var ret = MathUtil.Mtx.Zeros(a.h, a.w);
      for(var i = 0; i < a.h; i++) { 
        for(var j = 0; j < a.w; j++) { 
          ret[i][j] = a[i][j] * a_sc + b[i][j] * b_sc;
        }
      }
      return ret;
    },

    //compute singular value decomposition of a matrix A
    //don't expect it to be fast. return an object {U:mtx,S:mtx,V:mtx}
    //such that U*S*(V^T) = a, U^T*U = I, V^T*V = I, S diagonal    
    //  
    //for rectangular matrix with height A.h and width A.w,
    //U should have dims of A.h*A.h 
    //V should have dims of A.w*A.w
    //S should have dims of A.h*A.w
    //One way to solve for the SVD is to find the eigen vectors of A^T*A and A*A^T
    //Eigen vectors of A*A^T make up the columsn of U, Eigen vecs of A^T*A make up columns of V
    //S is diagonal, containing the square root of the eigen values from U or S
    //
    //Here we use the jacobi method as described here:
    //thanks to: http://www.math.pitt.edu/~sussmanm/2071Spring08/lab09/index.html
    //This is the one-sided jacobi, needs square matrixes to operate
    //NOTE: "a" must be square!!!
    jacobiSVD(a) { 
      if(a.w != a.h)
        console.log("ERROR! SVD doesn't work with square matrixes");
      var u = MathUtil.Mtx.clone(a);
      var n = a.h;
      var v = MathUtil.Mtx.Identity(n);
      var conv = Number.MAX_VALUE;
      tol = 0.000001;
      //var min_iters = 20;
      //while(1) { 
      //  if(conv >= tol) { 
      //    min_iters--;
      //  }
      //  if(min_iters <= 0)
      //    break;
      while (conv >= tol) { 

        var conv = 0.0;
        for(var j = 1; j < n; j++) {
          for(var i = 0; i < j; i++) { 

            var alpha = 0;
            var beta = 0;
            var gamma = 0;
            for(var k = 0; k < n; k++) {
              alpha += u[k][i]*u[k][i];
              beta += u[k][j]*u[k][j];
              gamma += u[k][i]*u[k][j];
            }
            var vconv = Math.abs(gamma)/Math.sqrt(alpha*beta);
            if(vconv > conv)
              conv = vconv;

            var zeta = (beta - alpha)/(2.0*gamma);
            var t = Math.sign(zeta)/(Math.abs(zeta) + Math.sqrt(1 + zeta*zeta));
            var c = 1.0/Math.sqrt(1.0 + t*t);
            var s = c*t;

            for(var k = 0; k < n; k++ ) {
              var tmp = u[k][i];
              u[k][i] = c*tmp - s*u[k][j];
              u[k][j] = s*tmp + c*u[k][j];
            }

            for(var k = 0; k < n; k++) { 
              var tmp = v[k][i];
              v[k][i] = c*tmp - s*v[k][j];
              v[k][j] = s*tmp + c*v[k][j];
            }
          
          }
        }
      }
      var s = MathUtil.Mtx.Zeros(n);
      for(var j = 0; j < n; j++ ) {
        s[j][j] = MathUtil.Mtx.colNormalize(u, j);
      }      
      return {U:u, S:s, V:v}; 
    },

    //compute the inverse using jacobi svd
    jacobiInv : function(a) { 
      var svd = MathUtil.Mtx.jacobiSVD(a);
      for(var i = 0; i < svd.S.h; i++) {
        svd.S[i][i] = 1 / svd.S[i][i];
        if(isNaN(svd.S[i][i]))
          svd.S[i][i] = 0.0;
        else if(svd.S[i][i] == Number.POSITIVE_INFINITY)
          svd.S[i][i] = 0.0;
          //svd.S[i][i] = 10.0;
        else if(svd.S[i][i] == Number.NEGATIVE_INFINITY)
          svd.S[i][i] = 0.0;
          //svd.S[i][i] = -10.0;
      }
      return MathUtil.Mtx.product3(svd.V, svd.S, MathUtil.Mtx.transpose(svd.U));  
    },


    //Ok, so some stuff about householder transforms...
    //(HN*...*H3*H2*H1)^-1 = H1*H2*H3*...*HN
    //H = I - 2vv^T where v corresponds to the unit normal of the the vectors.
    //plane we are reflecting through.
    //Note, since it's a reflection we have that H^-1 = H^T
    //Householder transforms are hermition, unitary, involutory, determinant is -1, ev's are + and -1
    //Basically, we treat a column as a vector, 
    //and we are gonna reflect <x0,x1,x2,x3, ... xk, xk+1,..., xN> via plane
    //so that it becomes <y0, y1 ... yk, 0, 0, 0, 0....>
    //Now, reflecting means setting v = (x-y)/|x-y|
    //
    //If we want to introduce zeros below i, we can use only the elements below i to do that,
    //so if that all elements above are unaffected. This fact can be used to easily make an upper
    //triangular matrix using HHT's and is the QR algo.
    //
    //NOTE: this is super inneficient to use, first of, we don't need to multiply the WHOLE matrix
    houseHolderZeroBelow : function(a, i0, j0) { 
      var ret = MathUtil.Mtx.Identity(a.h, a.h);
      
      var norm =  0.0;
      for(var i = i0; i < a.h; i++) 
        norm += a[i][j0] * a[i][j0];
      norm = Math.sqrt(norm);

      //soo... we will reflect that vector into one that is simply <norm,0,0...> or <-norm,0,0,...>
      //we will pic the sign so that the first element of (X-Y) is larger. that would be achieved
      //by making the sign of the resulting vector be the opposite of the sign of X[0]      
        
      var s = -Math.sign(a[i0][j0]);
     
      var xminusy = [];
      for(var i = i0; i < a.h; i++)
        xminusy.push(a[i][j0]); 
      xminusy[0] -= s*norm;

      //that means P = I - 2*WW^T
      //W = (X-Y)/|X-Y|
      var normxy = 0;
      for(var i = 0; i < xminusy.length; i++)
        normxy += xminusy[i]*xminusy[i];
      normxy = Math.sqrt(normxy);
      //console.log(norm, normxy);

      for(var i = 0; i < xminusy.length; i++)
        xminusy[i] /= normxy;
      
      for(var i = 0; i < xminusy.length; i++) { 
        for(var i2 = 0; i2 < xminusy.length; i2++) { 
          ret[i+i0][i2+i0] -= xminusy[i]*xminusy[i2]*2.0;
        }
      }

      return ret;
    },


    ////decompose the matrix using householder transforms 
    //We want to decompose A into orthogonal Q and upper triangular R
    //Householder transforms are orthogonal transforms that mean
    //reflection through a hyperplane passing though the origin.
    //
    qrDecompositionHH : function(a) { 
      var m = a.h;
      var n = a.w;
            
      var q = MathUtil.Mtx.Identity(m);
      var r = MathUtil.Mtx.clone(a);
      
      //we will make r upper triangular via HHT, and accumulate the inverse HHT's in q
      //which will stay orthogonal
    
      for(var j = 0; j < n; j++) { 

        //horribly inneficient, but illustrates what's happening much better
        var hh = MathUtil.Mtx.houseHolderZeroBelow(r, j, j);

        //note:  hh*hh = I (they're orthogonal/symmetric/ reflections), so after this update q*r = the same thign
        q = MathUtil.Mtx.product(q, hh);
        r = MathUtil.Mtx.product(hh, r); 
        //when multiply q_new * r_new = (q * hh) * (hh * r) = q*r
      }
    
      return {Q:q, R:r};
    },

    ////find the real eigen values via QR decomposition
    qrEigen : function(a) { 
      //super dupe slow, see what wiki has to say

      var a_tmp = MathUtil.Mtx.clone(a);
      var p_q = MathUtil.Mtx.Identity(a.h);
      //while(1) { 
      //TODO ... how many iterations?
      for(var iter = 0; iter < 900; iter++) { 
        var qr = MathUtil.Mtx.qrDecompositionHH(a_tmp);
        p_q = MathUtil.Mtx.product(p_q, qr.Q);
        a_tmp = MathUtil.Mtx.product(qr.R, qr.Q);
      }
      //console.log("----------------");
      //console.log(a_tmp);
      //console.log("vecs");
      //console.log(p_q);
      var v2 = MathUtil.Mtx.Zeros(a.h, 1);
      for(var i = 0; i < a.h; i++)
        v2[i][0] = a_tmp[i][i];
      return {vals:v2, vecs:p_q};
    },

    //factors "a" into L*L^T 
    //
    //This method is very useful for sampling N-dimensional gaussians with non-identity
    //covariance matrixes. 
    //
    //If we have a N-d gaussian with a known covariance matrix that we want to sample, 
    //use the cholesky decomposition on the covariance matrix, and sampling N independant
    //one-dimensional gaussians and multiply them by L to obtain samples that have the 
    //required covariance.
    cholesky : function(a) { 
      
      var n = a.h;
      var L = MathUtil.Mtx.Zeros(n);

      for(var i = 0; i < n; i++) {
        for(var j = 0; j < i+1; j++) {
          var s = 0;
          for(var k = 0; k < j; k++)
            s += L[i][k] * L[j][k];
          if(i == j) 
            L[i][j] = Math.sqrt(a[i][i] - s);
          else
            L[i][j] = (1.0 / L[j][j] * (a[i][j] - s));
        }
      }

      return L;
    },

    //solve Ax = Y via iterative methods
    gaussSeidelRaw : function(A, Y)  {
      var X = MathUtil.Mtx.Zeros(A.w, 1);
      for(var iter = 0; iter < 30000; iter++) { 
        for(var i = 0; i < A.h; i++) { 
          var r2 = 0;
          for(var j = 0; j < i; j++)
            r2 += A[i][j] * X[j][0];
          for(var j = i+1; j < A.w; j++)
            r2 += A[i][j] * X[j][0];
          X[i][0] = 1.0/A[i][i] * (Y[i][0] - r2);
        }
      }
      return X;
    },

  },
};



var XORShift32 = function(seed) {   
  if(!seed)
    seed = 0x2589ede2; 
  this.state = seed;
};

XORShift32.prototype.gen = function() { 
  this.state = MathUtil.extXORShift32(this.state);//not sure if this speeds anything up
  return this.state;
};

var RandSeeds = [
  [0x12311afc, 0x5fecaa34, 0x41b28a7f, 0x238c9c72, 0x6662f962, 0x2a7dd33a, 0x20e100d3, 0x7cca4f92, 0x013a3eff, 0xdb70622b, 0x28db77a0], //11 
  [0x9b8c1aff, 0xcd481748, 0xe0b2e99d, 0x7225c2db, 0x5f6096d5, 0x815fb56c, 0x71a87f32],//7
  [0x5c5e7be4, 0x23784080, 0x1a9f7921, 0x124c9fda, 0x131dee90],//5
 ];




var XORShift64Star = function(seed0, seed1) {  
  if(!seed0)
    seed0 = 0x3232532;
  if(!seed1) {
    seed1 = MathUtil.extXORShift32(seed0);
  } 
  this.state =[seed1 >>> 0,seed0 >>> 0];
  this.ret = [];
  this.needs_gen = true;
};

XORShift64Star.prototype.writeToBuffer = function(buf) { 
  if(!buf.offs) {
    console.log("creating buf offs");
    buf.offs = 0;
  }    
  console.log("write to", buf.offs);
  buf.writeUInt8((this.needs_gen ? 1 : 0) | (this.lp_normal ? 2 : 0), buf.offs);
  buf.offs++;
  console.log("write to", buf.offs);
  buf.writeUInt32LE(this.state[0], buf.offs);
  buf.offs += 4;
  console.log("write to", buf.offs, "buf.length", buf.length, "state", this.state[1]);
  buf.writeUInt32LE(this.state[1], buf.offs);
  buf.offs += 4;
  if(this.lp_normal) {
    console.log("write to", buf.offs);
    buf.writeFloatLE(this.lp_normal, buf.offs);
    buf.offs += 8;
  }
};


//how does sync work?
//first of all when generating values, we have to generate one U64 at a time, but since we 
//typically make use of only U32's sometimes we don't advance the generator. That's what needs_gen
//signifies. Another tricky thing is when we generate normal random values we generate two at once,
//so one value is sometimes held in a cache. Based on the state of needs_gen and lp_normal we might
//have a variable serialization size. For now simplify things with a fixed size 
XORShift64Star.prototype.writeToModBuf = function(buf) { 
  buf.pushU8((this.needs_gen ? 1 : 0) | (this.lp_normal ? 2 : 0), buf.offs);
  buf.pushU32(this.state[0]);
  buf.pushU32(this.state[1]);
  if(this.lp_normal) 
    buf.pushF32(this.lp_normal);
  else
    buf.pushF32(0);
};

XORShift64Star.prototype.readFromModBuf = function(buf) { 
  var flags = buf.popU8();
  if(flags & 1)
    this.needs_gen = true;
  this.state[0] = buf.popU32();
  this.state[1] = buf.popU32();
  if(flags & 2) { 
    this.lp_normal = buf.popF32();
  } else { 
    this.lp_normal = undefined;
    buf.popF32();
  }
};



XORShift64Star.prototype.readFromDataView = function(inview, offs) { 
  if(!offs)
    offs = 0;
  var offs0 = offs;
  var flags = inview.getUint8(offs);
  offs++;
  this.state[0] = inview.getUint32(offs, true);
  offs += 4;
  this.state[1] = inview.getUint32(offs, true);
  offs += 4;
  if(flags & 1)
    this.needs_gen = true;
  else  
    MathUtil.multU64(this.state, [0x4F6CDD1D, 0x2545F491], this.ret);
  if(flags & 2) {
    this.lp_normal = inview.getFloat64(offs, true);
    offs += 8;
  }    
  return offs - offs0;
};


XORShift64Star.prototype.gen = function() {
  if(this.needs_gen) {       
    MathUtil.xorAndrshiftZF64(this.state, 12);
    MathUtil.xorAndrshiftZF64(this.state, -25);
    MathUtil.xorAndrshiftZF64(this.state, 27);
    this.state[0] >>>= 0;
    this.state[1] >>>= 0;
    MathUtil.multU64(this.state, [0x4F6CDD1D, 0x2545F491], this.ret);
    this.needs_gen = false;
    return this.ret[1];
  } else { 
    this.needs_gen = true;
    return this.ret[0];
  }
  //this works for basic [0,1]
  //return this.gen() / (0xFFFFFFFF);
};


//result will be in [0,1) using 53 bits effectively (supposedly)
//taken from mersenne-twister.js
XORShift64Star.prototype.random = function() { 
  var a=this.gen()>>>5, b=this.gen()>>>6; 
  return(a*67108864.0+b)*(1.0/9007199254740992.0); 
};

//scale this by a the desired standard deviation
//and add the desired mean to the result.
XORShift64Star.prototype.normal = function() {
  if(this.lp_normal === undefined) { 
    var u1 = this.random(); 
    var u2 = this.random(); 
    var tmp = Math.sqrt( -2.0 * Math.log(u1) );
    this.lp_normal = tmp * Math.sin( 2.0 * Math.PI * u2);
    return tmp * Math.cos( 2.0 * Math.PI * u2);
  } else { 
    var tmp = this.lp_normal;
    this.lp_normal = undefined;
    return tmp;
  }
};



function BinDensitySampler(min, max, bins) { 
  this.bincts = [];
  this.gtect = 0;
  this.ltct = 0;
  this.ct = 0;
  this.min = min;
  this.max = max;
  this.sum = 0;
  this.bins = bins;
  this.spacing = (max - min)/bins;
  for(var i = 0; i < bins; i++)
    this.bincts[i] = 0;
};
BinDensitySampler.prototype.avg = function() {
  return this.sum / this.ct;
}; 
BinDensitySampler.prototype.add = function(v) { 
  this.sum += v;
  this.ct++;
  var idx = Math.floor(this.bins * (v - this.min)/(this.max - this.min)) 
  if(idx < 0)
    this.ltct++;
  else if(idx >= this.bins)
    this.gtect++;
  else
    this.bincts[idx]++;
};
BinDensitySampler.prototype.approxPdf = function() { 
  var ret = [];
  for(var i = 0; i < this.bins; i++) {   
    ret.push({p: this.bincts[i]/this.ct/this.spacing, v: (i+0.5)*this.spacing+this.min, left:i*this.spacing+this.min, right:(i+1)*this.spacing+this.min});
  }
  return ret;
}



//can use this to sample arbitrary densities
function DensitySampler() { 
  this.samples = {};
  this.ct =0;
  this.sum = 0;
  this.sumsq = 0;
};

DensitySampler.prototype.avg = function() {
  return this.sum / this.ct;
}; 

DensitySampler.prototype.avgOfSq = function() {
  return this.sumsq / this.ct;
};

DensitySampler.prototype.variance = function() {
  //E[(E(X) - X)^2] = E[X^2] - E[X]^2
  //scale up to get a more correct estimate
  //return this.ct/(this.ct-1) * (this.sumsq/this.ct  - (this.sum/this.ct)*(this.sum/this.ct));
  return this.sumsq/(this.ct-1)  - (this.sum/this.ct)*(this.sum/(this.ct-1));
};
DensitySampler.prototype.push = function(x) {  
  if(!this.samples[x])
    this.samples[x] = 1;
  else
    this.samples[x]++;
  this.ct++;
  this.sum += x;
  this.sumsq += x*x;
};
DensitySampler.prototype.add = function(x) { 
  this.push(x);
}; 

DensitySampler.prototype.approxPdf = function(min_ct, min_w) { 
  
  if(min_ct === undefined) {
    min_ct = Math.round(this.ct / 1000);
    if(min_ct < 4)
      min_ct = 4;
  }

  var ret = [];

  var keys = Object.keys(this.samples);
  for(var i = 0; i < keys.length; i++)
    keys[i] = Number(keys[i]);
  var keys = keys.sort(function(a,b) { return a-b; } );
  
  if(min_w === undefined) 
    min_w = (keys[keys.length-1] - keys[0]) / 1000.0; 

  var vL = keys[0];
  var ctb = this.samples[keys[0]];
  var sumb = this.samples[keys[0]] * keys[0];

  for(var i = 1; i < keys.length; i++) { 
    

    var vR = keys[i];
    ctb += this.samples[keys[i]] / 2;
    sumb += this.samples[keys[i]] * keys[i] /2;
  
    if(ctb > min_ct && (vR - vL) > min_w) { 
      ret.push({p: ctb / (vR - vL) / this.ct, v: sumb/ctb, left:vL, right:vR});

      vL = vR;
      ctb = 0;
      sumb = 0;
    } 

    ctb += this.samples[keys[i]] / 2;
    sumb += this.samples[keys[i]] * keys[i] /2;

  }
  
  return ret;

};




if(typeof(module) == 'object' && typeof(module.exports) == 'object') { 
  module.exports.MathUtil = MathUtil;
  module.exports.XORShift64Star = XORShift64Star;
  module.exports.BinDensitySampler = BinDensitySampler;
  module.exports.DensitySampler = DensitySampler;
}


