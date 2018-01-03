//This class represents 2-dimensional vectors


var V2 = function() {
  
  //construct by x,y value 
  this.x = arguments[0];
  this.y = arguments[1];

};

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////


var V2Util = {
  
  clone : function(vec) { 
    return new V2(vec.x, vec.y);
  },


  //compute the angle between B->A and B->C
  // A   ret  C 
  //  \   |  /
  //   \  v /
  //    \__/
  //     \/
  //     B
  //
  //If the shortest arc going from A to C relative to B is clockwise then the 
  //angle returned is positive, otherwise it will be negative. Returned in radians.  
  angleABC : function(ptA, ptB, ptC) { 
    //return (ptA.cloneMinus(ptB)).angleTo(ptC.cloneMinus(ptB));
    //premature optimization, test to see if this makes a difference:
    return Math.atan2(-(ptA.x-ptB.x)*(ptC.y-ptB.y)+(ptA.y-ptB.y)*(ptC.x-ptB.x), (ptA.x-ptB.x)*(ptC.x-ptB.x) + (ptA.y-ptB.y)*(ptC.y-ptB.y));
  },


  //The area of a triangle times a sign.
  //The sign is positive when the line loop a->b->c(->a) is clockwise 
  //and negative when it's counter-clockwise 
  signedAreaTriangle : function(ptA, ptB, ptC) { 
    return (ptB.cloneMinus(ptA)).cross(ptA.cloneMinus(ptC))/2.0;
  },


  //The centroid of a triangle is also it's center of mass
  centroidTriangle : function(ptA, ptB, ptC) { 
    return new V2((ptA.x + ptB.x + ptC.x)/3.0, (ptA.y + ptB.y + ptC.y)/3.0);
  },


  //calcultes moment of inertia about (0,0)
  momentOfInertiaTriangle : function(ptA, ptB, ptC) { 
    var area = V2Util.signedAreaTriangle(ptA, ptB, ptC);
    var d2 = V2Util.centroidTriangle(ptA, ptB, ptC).magSquared();
    //original more obvious method
    //var b = ptA.dist(ptC);
    //if(b == 0) { 
    //  return 0;
    //}
    //var h = ptA.cloneMinus(ptB).cross(ptC.cloneMinus(ptB)) / b;
    //var a = ptC.cloneMinus(ptA).dot(ptB.cloneMinus(ptA)) / b;
    //var ibase =  (b*b*b*h - b*b*h*a + b*h*a*a + b*h*h*h) / 36.0;
    //improved method for calcuialte ibase without divisions:
    var ibase = (ptA.x*(ptC.y-ptB.y)+ptB.x*(ptA.y-ptC.y)+ptC.x*(ptB.y-ptA.y))*(ptA.x*ptA.x+ptB.x*ptB.x+ptC.x*ptC.x-ptA.x*ptB.x-ptA.x*ptC.x-ptB.x*ptC.x+ptA.y*ptA.y+ptB.y*ptB.y+ptC.y*ptC.y-ptA.y*ptB.y-ptA.y*ptC.y-ptB.y*ptC.y)/36.0;
    return ibase + d2*area;//application of parallel axis theorem
  },


  average : function() { 
    if(arguments.length == 2) 
      return new V2((arguments[0].x + arguments[1].x)/2, (arguments[0].y + arguments[1].y)/2);
    else if(arguments.length == 3) 
      return new V2((arguments[0].x + arguments[1].x + arguments[2])/3, (arguments[0].y + arguments[1].y + arguments[2].y)/3);
    else { 
      var ret = new V2(0,0);
      for(var i = 0; i < arguments.length; i++) 
        ret.plusEq(arguments[i]);
      ret.divide(arguments.length);
      return ret;
    }
  },

  //TODO: test me
  rayHit : function(ray0, raydir, ptA, ptB) {

    //find t such that ray0 + raydir*t = ptA*u+ptB*(1-u)
    //subject to u being in [0,1]
    //
    //ray0.x + raydir.x *t = ptA.x * u + ptB.x * (1-u)
    //ray0.y + raydir.y *t = ptA.y * u + ptB.y * (1-u)
    //
    //ray0.x - ptB.x = (ptA.x - ptB.x) * u + (-raydir.x) * t;
    //ray0.y - ptB.y = (ptA.y - ptB.y) * u + (-raydir.y) * t;
    var res = MathUtil.solve2x2(ptA.x-ptB.x, -raydir.x, ptA.y - ptB.y, -raydir.y, ray0.x - ptB.x, ray0.y - ptB.y);
    if(res[0] >= 0 && res[0] <= 1.0 && res[1] >= 0.0) { 
      return res[1];
    }
    return undefined;
  },
 
  //assume pt0 goes from pt0A at time 0 to pt0B at time 1 and 
  //pt1 goes from pt1A at time 0 to pt1B at time 1, calculate
  //time t when pt0 and pt1 are nearest to each other
  //IMPORTANT: this function can return a value <0 or >1 for 
  //when the min distance occurs and undefined when the points
  //are travelling on parrallel lines at the same "velocity"
  pointPointNearestTime : function(pt0A, pt0B, pt1A, pt1B) { 
    
    //minimize t: | pt0A * (1-t) + pt0B * t - pt1A * (1-t)  - pt1B * t | 
    //
    //minimize t: | (pt0A - pt1A ) + (pt0B - pt0A + pt1A - pt1B) * t | 
    //
    //the minimizer for ax^2 + bx + c is at x = -b/2a
    //
    //for us b = (pt0B.x - pt0A.x + pt1A.x - pt1B.x)*(pt0A.x - pt1A.x)*2.0 + "same for y"
    //and a = (pt0B.x - pt0A.x + pt1A.x - pt1B.x)^2 + "same for y"
    //(and c is = (pt0A.x - pt1A.x)^2 + "same for y" 

    var wx = (pt0B.x - pt0A.x + pt1A.x - pt1B.x);
    var wy = (pt0B.y - pt0A.y + pt1A.y - pt1B.y);
    
    //Note: if wx is zero and wy is zero then t will be undefined, however this implies that:
    //  pt0B.x - pt0A.x = pt1B.x - pt1A.x 
    //  pt0B.y - pt0A.y = pt1B.y - pt1A.y
    //which implies that both points are travelling at the same speed on parallel lines 
    
    var t = -(wx*(pt0A.x - pt1A.x) + wy*(pt0A.y - pt1A.y)) / ( wx*wx + wy*wy);
    return t;
  },

  //Assume pt0 goes from pt0A at time 0 to pt0B at time 1 and 
  //pt1 goes from pt1A at time 0 to pt1B at time 1.
  //Return the first time t when pt0 and pt1 are exactly r appart
  //but only for t in the range of [0,1]. if there is no such
  //t return undefined.
  pointPointEqTime : function(pt0A, pt0B, pt1A, pt1B, r) { 
    var wx = (pt0B.x - pt0A.x + pt1A.x - pt1B.x);
    var wy = (pt0B.y - pt0A.y + pt1A.y - pt1B.y);
    var ux = (pt0A.x - pt1A.x);
    var uy = (pt0A.y - pt1A.y);
    var a = wx*wx+wy*wy;
    var b = (wx*ux+wy*uy)*2.0;
    var c = ux*ux + uy*uy - r;
    var d = b*b - 4.0*a*c;
    if(d < 0.0 || a == 0)
      return undefined;
    d = Math.sqrt(d);
    var t = (-b - d)/a/2.0;
    if(t < 0)
      t = (-b + d)/a/2.0;
    if(t <0 || t > 1)
      return undefined;
    return t;
    //return -(wx*(pt0A.x - pt1A.x) + wy*(pt0A.y - pt1A.y)) / ( wx*wx + wy*wy);
  },

  minOnSegment : function(a, b) { 
    //TODO: optimize this a bit
    var t = this.minOnSegmentTime(a,b);
    return new V2(a.x*(1.0-t) + b.x*t, a.y*(1.0-t) + b.y*t);
  },

  //return the time t at which the minimum occured.
  //NOTE: if we use this as a weighting, remember to use a*(1-t) + b*t to get the actual point
  minOnSegmentTime : function(a,b) { 
    //get the point with the smallest magnitude on the line going from pt0 to pt1
    //min[ (a_x + (b_x - a_x) * t)^2 + (a_y + (b_y - a_y) * t)^2 ]
    //occurs when the deriviative WRT to t is zero
    // 
    // (a_x + (b_x - a_x) * t) * (b_x - a_x)   +   (a_y + (b_y - a_y) * t) * (b_y - a_y) = 0
    // -(a_x * (b_x - a_x) + a_y * (b_y - a_y)) / ((b_x - a_x)*(b_x - a_x) + (b_y - a_y)*(b_y - a_y))
    var bax = b.x - a.x;
    var bay = b.y - a.y;
    var denom = bax*bax + bay*bay;
    if(Math.abs(denom) <= Number.MIN_VALUE*4) //this is true if a==b more or less
      return 0.5;
    var t = -(a.x * bax + a.y * bay) / denom;
    //enforce the bounds...
    if(t > 1.0)
      return 1.0;
    if(t < 0.0)
      return 0.0;
    return t;
  },

  //sum up two vectors scaled by two constants
  sumScale2 : function(a, asc, b, bsc) { 
    return new V2(a.x * asc + b.x * bsc, a.y * asc + b.y * bsc);
  },
  //sum up three vectors scaled by three constances
  sumScale3 : function(a, asc, b, bsc, c, csc) { 
    return new V2(a.x * asc + b.x * bsc + c.x * csc, a.y * asc + b.y * bsc + c.y * csc);
  },  
  
  //this is a strange function in that it doesn't take V2's, used externally
  rectPointDist2 : function(x0, xL, y0, yL, xP, yP) { 
    var sx;
    var sy;

    if(xP < x0)
      sx = x0;
    else if(xP > xL)
      sx = xL;
    else
      sx = xP;

    if(yP < y0)
      sy = y0;
    else if(yP > yL)
      sy = yL;
    else
      sy = yP;

    return (sx - xP)*(sx - xP) + (sy - yP)*(sy - yP);
  },

  //this samples on the disk uniformly
  sampleOnDisk : function(r0, r1, rand_src) { 
    
    if(rand_src === undefined)
      rand_src = Math;

    //first sample the angle uniformly
    var angle = rand_src.random() * 2.0 * Math.PI;
    
    //now prepare to sample the radius 
    //d(area)/d(r) 2 = 2*pi*r 
    //to sample the disk uniformly we need to have p(r) ~ 2*pi*r 
    //
    //if we uniformly sample x from [0 to 1] and apply monotone f to it,
    //the pdf of of f(x) is d(f_inverse(x))/dx for example taking the pdf x^2
    //the pdf will be proportional to d/dx ( sqrt(x) )  = 1/sqrt(x)
    //
    //the function we want has pdf(f(x)) = C*(x + r0) and going from r0 to r1
    //so first we take the integral of it and get C*(x^2/2 + r0*x) = 2C * (x^2 + x*r0/2) 
    //2C*(r1*r1 - r0*r0 + (r1*r0 - r0*r0)/2) = 1.0
    //...let C absorb the extra 2...
    //C*(r1*r1 - r0*r0 + (r1*r0 - r0*r0)/2) = 1.0
    var C = 1.0/((r1-r0)*(r1-r0) + r0*(r1-r0)/2);
    //
    //then we need to invert that. 
    //we need to reformulate the quadratic as (x+c)^2 + d = x^2 + 2xc + c^2 + d 
    //then we can write sqrt(y-d)-c = x
    //2c = r0/2
    //c = r0/4
    //c^2 = -d 
    //d = -r0^2/8
    //
    //sqrt(y/C + r0^2/8)-r0/4 = x     
    //
    //to solve for y0 we do sqrt(y/C + r0^2/8)-r0/4 = r0
    var y0 = ((r0*5/4)*(r0*5/4) - r0*r0/8)*C;
    var yL = ((r1 + r0/4)*(r1 + r0/4) - r0*r0/8)*C;

    var y = rand_src.random()*(yL-y0) + y0;
    var r = Math.sqrt(y/C + r0*r0/8)-r0/4; 

    return new V2( Math.cos(angle) * r, Math.sin(angle) * r);
  }, 



  //this just makes a cool looking result, not sure of the pdf, but its probably 
  //not too hard to derive
  sampleOnDisk2 : function(r0, r1, invert, rand_src) { 
    
    if(rand_src === undefined)
      rand_src = Math;

    //first sample the angle uniformly
    var angle = rand_src.random() * 2.0 * Math.PI;
    
    //prevent overflow problems
    //var sc = Math.sqrt(r0+r1)*0.5;///2 * 0.5;
    var sc = (r0+r1)/8;
    r0 /= sc;
    r1 /= sc;

    //this is an incorrect way to do this, but it makes a cool effect
    var re0 = (Math.exp(r0*(r1-r0),5) - r1)/(r1-r0);
    var re1 = (Math.exp(r1*(r1-r0),5) - r1)/(r1-r0);
    var x = rand_src.random() * (re1-re0) + re0;
    var r = Math.log(x*(r1-r0) + r1) / (r1-r0);
    if(invert) { 
      //swap r0 and rL
      r = r1 - (r - r0);
    }

    r *= sc;

    return new V2( Math.cos(angle) * r, Math.sin(angle) * r);

  },


  sampleCustom0 : function(r) { 
    if(!r)
      r = 1.0;
    var ret = new V2(0,0);
    for(var i = 0; i < 5; i++) {
      ret.x += Math.random()-0.5;
      ret.y += Math.random()-0.5;
    }
    ret.scale(r/2.5);
    return ret;
  },

};



////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

V2.prototype.copy = function(other) {
  this.x = other.x;
  this.y = other.y;
};

V2.prototype.clone = function() { 
  return new V2(this.x, this.y);
};

V2.prototype.zero = function() { 
  this.x = 0.0;
  this.y = 0.0;
};

V2.prototype.round = function() { 
  this.x = Math.round(this.x);
  this.y = Math.round(this.y);
};

//return the magnitude-squared of the vector
V2.prototype.magSquared = function() { 
  return this.x*this.x + this.y*this.y;
};

//quick alias
V2.prototype.mag2 = function() { 
  return this.x*this.x + this.y*this.y;
};

V2.prototype.mag = function() { 
  return Math.sqrt(this.x*this.x + this.y*this.y);
};

//also returns the mag prior to normalization
V2.prototype.normalize = function() { 
  var m = this.mag();
  if(m < 0.0000000001) 
    m = 0.0000000001;
  this.x /= m;
  this.y /= m;
  return m;
};

//call this in optimized code that doesn't care about getting the magnitude
//and knows ahead of time that the vector isn't 0.0
V2.prototype.normalizeNonZeroNoRet = function() { 
  var m = this.mag();
  this.x /= m;
  this.y /= m;
};

V2.prototype.normalizedClone = function() { 
  var m = this.mag();
  if(m < 0.0000000001) 
    m =  0.0000000001;
  return new V2( this.x / m , this.y / m);
};


//A.angleTo(B) returns the smallest angle going from A to B 
//the result is positive if clockwise, negative if counterclockwise
//NOTE: normalization seems not necassary, but why? does atan2 take care of normalizing?
//     ________  A
//    /
//   /
//  /     ... result is positive +135 degees, but in radians
//B
//
//
//        B
//     /
//    / 
//   /_____ A    ...result is negative -45 degrees, but in radians
//
//
////actual code example:
//var up = new V2(0,1); var right = new V2(1,0);
//up.angleTo(right) returns 
//console.log((new V2(0,1)).angleTo(new V2(1,0)) / Math.PI); -> 0.5
V2.prototype.angleTo = function(other) { 
  return Math.atan2(-this.x*other.y+this.y*other.x, this.x*other.x + this.y*other.y);
};


//A.dot(B) This returns something similar to Math.cos(A.angleTo(this.B)) * A.mag() * B.mag();
V2.prototype.dot = function(other) { 
  return this.x * other.x + this.y * other.y;
}

V2.prototype.cos = function(other) { 
  var m0 = Math.sqrt((this.x*this.x + this.y*this.y)*(other.x*other.x + other.y*other.y));
  if(m0 == 0.0)
    return 0.0;
  return (this.x * other.x + this.y * other.y) / m0;
};

//new V2(0,1).cross(new V2(1,0));
//wolfram alpha agrees that {0,1} cross {1,0} is -1
V2.prototype.cross = function(other) { 
  return this.x*other.y - this.y*other.x;
}

V2.prototype.plusEq = function(other) { 
  this.x += other.x; 
  this.y += other.y;
};

V2.prototype.plusEqXY = function(otherx, othery) { 
  this.x += otherx; 
  this.y += othery;
};
V2.prototype.clonePlusXY = function(otherx, othery) { 
  return new V2(this.x + otherx, this.y + othery);
};

V2.prototype.plusEqScaled = function(other, sc) { 
  this.x += other.x*sc; 
  this.y += other.y*sc;
};

//alternative names for this might be:
//plusClone, plusRet, plus
V2.prototype.clonePlus = function(other) { 
  return new V2(this.x+other.x, this.y+other.y);
};

V2.prototype.clonePlusScaled = function(other, sc) { 
  return new V2(this.x+other.x*sc, this.y+other.y*sc);
};

V2.prototype.minusEq = function(other) { 
  this.x -= other.x;
  this.y -= other.y;
};

V2.prototype.cloneMinus = function(other) { 
  return new V2(this.x-other.x, this.y-other.y);
};

V2.prototype.scale = function(c) { 
  this.x *= c;
  this.y *= c;
};

V2.prototype.cloneScale = function(c) { 
  return new V2(this.x * c, this.y * c);
};

V2.prototype.avgWith = function(v2_other, perc_other) { 
  this.x = v2_other.x * perc_other + this.x * (1.0 - perc_other);
  this.y = v2_other.y * perc_other + this.y * (1.0 - perc_other);
};

V2.prototype.divide = function(c) { 
  this.x /= c;
  this.y /= c;
};

//this rotates the vector 90 degrees counter-clockwise is 90 degrees counter-clockwise
V2.prototype.rotate90CCW = function() { 
  var tmp = this.x;
  this.x = -this.y;
  this.y = tmp;
};

V2.prototype.rotate90CW = function() { 
  var tmp = this.x;
  this.x = this.y;
  this.y = -tmp;
};


//if we are given an x-axis to project along,
//get the new (x,y) coordinates. note: to get 
//from x-axis to y-axes rotate 90* CCW
V2.prototype.projected0 = function(x_axes0) { 
  return new V2( this.x * x_axes0.x + this.y * x_axes0.y , -this.x * x_axes0.y + this.y * x_axes0.x );
};

V2.prototype.cloneRotate90CW = function() { 
  return new V2(this.y, -this.x);
};

V2.prototype.cloneRotate90CCW = function() { 
  return new V2(-this.y, this.x);
};


V2.prototype.eq = function(other) { 
  return this.x == other.x && this.y == other.y;
};

V2.prototype.distSquared = function(other) { 
  var dx = this.x - other.x;
  var dy = this.y - other.y;
  return dx*dx + dy * dy;
};

//alias
V2.prototype.dist2 = function(other) { 
  var dx = this.x - other.x;
  var dy = this.y - other.y;
  return dx*dx + dy * dy;
};

V2.prototype.dist = function(other) { 
  return Math.sqrt(this.dist2(other));
};

V2.prototype.gravF = function(v2_other, scale, normalizer) { 
  var dx = this.x - v2_other.x;
  var dy = this.y - v2_other.y;
  var mag2m = (dx*dx + dy*dy + normalizer)/scale;
  return new V2(dx / mag2m, dy / mag2m);
};

//this.asteroids[i].lin_vel.minusEq( this.asteroids[i].lin_pos.gravF(this.blackholes[j].pos, this.blackholes[j].mass, 5.0) );

//Rotate clockwise by some number of radians .
//NOTE: calculating Math.cos/Math.sin for each point may be 
//expensive. If rotating many points at ones, we need some way
//to cache those values.
V2.prototype.rotateCW = function(radians) { 
  var cr = Math.cos(radians);
  var sr = Math.sin(radians);
  var x2 = cr * this.x + sr * this.y;
  this.y = -sr * this.x + cr * this.y;
  this.x = x2;
};

V2.prototype.rotateCWRet = function(radians) { 
  var cr = Math.cos(radians);
  var sr = Math.sin(radians);
  var x2 = cr * this.x + sr * this.y;
  this.y = -sr * this.x + cr * this.y;
  this.x = x2;
  return this;
};


V2.prototype.cloneRotateCW = function(radians) { 
  var cr = Math.cos(radians);
  var sr = Math.sin(radians);
  return new V2(cr * this.x + sr * this.y, -sr * this.x + cr * this.y);
};


//special code for caching rotations
V2.prototype.cacheRotation = function(radians) { 
  this.x = Math.cos(radians);
  this.y = Math.sin(radians);
};
//if we need to rotate a lot of stuff we can avoid repeatedly calculating sin/cos
V2.prototype.rotateCWUsingCached = function(cached) { 
  var x2 = cached.x * this.x + cached.y * this.y;
  this.y = -cached.y * this.x + cached.x * this.y;
  this.x = x2;
};
V2.prototype.cloneRotateCWUsingCached = function(cached) { 
  return new V2(cached.x * this.x + cached.y * this.y, -cached.y * this.x + cached.x * this.y);
};




V2.prototype.rotateCWPlus = function(radians, offset) { 
  var cr = Math.cos(radians);
  var sr = Math.sin(radians);
  var x2 = cr * this.x + sr * this.y;
  this.y = -sr * this.x + cr * this.y + offset.y;
  this.x = x2 + offset.x;
};

V2.prototype.cloneRotateCWPlus = function(radians, offset) { 
  var cr = Math.cos(radians);
  var sr = Math.sin(radians);
  return new V2( cr * this.x + sr * this.y + offset.x, -sr * this.x + cr * this.y + offset.y);
};



//Treat the two arguments as the two endpoints of a linear segment
V2.prototype.distToSegment = function(segA, segB) {
  var len_ab = segA.dist(segB); 

  //calculate the distance of "this" along the line "segA->segB"
  //this is the same as (segB-segA).dot(this - segA) / len_ab; 
  var dist_along_ab = 
    ((segB.x - segA.x) * (this.x - segA.x) +
    (segB.y - segA.y) * (this.y - segA.y)) / len_ab;
  
  if(dist_along_ab < 0.0)
    return this.dist(segA);
  if(dist_along_ab > len_ab) 
    return this.dist(segB);    
  
  //This is the same as abs((segB-segA).cross(this - segB)) / d_ab;
  return Math.abs(((segB.x-segA.x)*(this.y - segB.y) - (segB.y - segA.y)*(this.x - segB.x))/len_ab);
};


//Given the line segment <segA,segB>, return the point on <segA, segB> that's nearest to "this"
//assumes that length of AB is nonzero
V2.prototype.nearestPointOnSegment = function(segA, segB) { 
  
  var len_ab = segA.dist(segB);
  if(len_ab <= Number.MIN_VALUE * 4)
    return V2Util.clone(segA); 
  var dist_along_ab = 
    ((segB.x - segA.x) * (this.x - segA.x) +
    (segB.y - segA.y) * (this.y - segA.y)) / len_ab;
  if(dist_along_ab < 0.0)
    return V2Util.clone(segA);
  if(dist_along_ab > len_ab) 
    return V2Util.clone(segB);
  return new V2( segA.x + dist_along_ab * (segB.x - segA.x) / len_ab, 
                  segA.y + dist_along_ab * (segB.y - segA.y) / len_ab ); 

};


//Reflects this about a provided normal vector, which can be un-normalized. Numerical
//rounding errors will occur if normal_v2 is very near perpindicular to this. 
//
//this  normal_v2
//    \     ^   _ this (after call)
//     \    |   /|
//      \   |  / 
//       \  | /
//       _\||/
//
//a more efficient version might assume that normal_v2 is already normalized properly
//TODO: test
V2.prototype.reflectWithNormal = function(normal_v2) {   
  var div = normal_v2.magSquared();  
  if(div < 0.0000000001) 
    div =  0.0000000001;
  var dist_along_normal_times_mag = this.dot(normal_v2);
  this.plusEqScaled(normal_v2, -2.0 * dist_along_normal_times_mag / div);
};


//given a triangle A,B,C in find the barycentric 
//coordinates (u,v) such that u+v <= 1, A*u + B*v + C*(1-u-v) = *this
//a_x * u  + b_x * v + c_x * (1-u-v) = this_x
//a_y * u  + b_y * v + c_y * (1-u-v) = this_y
//rephraised as a system of linear equations
//(a_x - c_x) * u + (b_x - c_x) * v = this_x - c_x
//(a_y - c_y) * u + (b_y - c_y) * v = this_y - c_y
//one option would be to copy code for solving a system of linear equations from MiniLin.h
//NOTE: if any of a==b, b==c, a==c then den will be zero
V2.prototype.barycentric = function(a,b,c) { 
  var v0_x = a.x - c.x;
  var v0_y = a.y - c.y;
  var v1_x = b.x - c.x;
  var v1_y = b.y - c.y;
  var v2_x = this.x - c.x;
  var v2_y = this.y - c.y;
  var den = 1.0/(v0_x * v1_y - v1_x * v0_y);
  return new V2((v2_x * v1_y - v1_x * v2_y) * den, (v0_x * v2_y - v2_x * v0_y) * den);
};


//Provide positions for the two endpoints of a segment at t=0 and t=1. 
//Get "t" when segment first collides with this point (can collide 0,1,2 or infinitely many times)
//
//This assumes that the endpoints of the segment move independantly in a linear fashion as an approximation,
//when really a rotating segment's endpoints moving in an arc.
//
//If both "this" and the segment are moving, treat "this" as stationary and add the removed
//velocity to the velocity of the segment. 
//
//If there are 0 collisions of t is outside of [0,1] return undefined. 
//
//TODO: test this some more
V2.prototype.collisionWithSlidingSegment = function(seg0A, seg1A, seg0B, seg1B) { 
  
  //NOTE: the distance to a segment is:
  //   Math.abs((segB.x-segA.x)*(this.y - segB.y) - (segB.y - segA.y)*(this.x - segB.x)) / length_AB;
  //   We only care when the distance is 0, so we can ignore the abs and length_AB normalizer. 
  //   ((segB.x-segA.x)*(this.y - segB.y) - (segB.y - segA.y)*(this.x - segB.x)) = 0;
  //   Next we can replace segB with seg0B * t + seg1B * (1-t) and likewise for segA 
  //   Then we will have a quadratic equation in terms of t and can simply solve for t. 
  //    at^2 + bt + c = 0 when t = (-b +/- sqrt(b^2-4ac))/2a
  
  //  ( (b0x*t + b1x*(1-t))  - (a0x*t + a1x*(1-t))) * (this.y - ( b0y*t + b1y*(1-t)))
  //    - ( (b0y*t + b1y*(1-t))  - (a0y*t + a1y*(1-t))) * (this.x - ( b0x*t + b1x*(1-t)))
  
  //    ((b0x-b1x-a0x+a1x)*t + (b1x-a1x)) * ((b1y - b0y)*t + (this.y - b1y)) 
  //    -((b0y-b1y-a0y+a1y)*t + (b1y-a1y)) * ((b1x - b0x)*t + (this.x - b1x)) = 0
  //    
  //    (c0*t +c1)*(c2*t+c3) - (c4*t + c5)*(c6*t+c7) = 0
  //    (c0*c2-c4*c6)*t^2 + (c1*c2 + c0*c3 - c5*c6 - c4*c7)*t + (c1*c3- c5*c7) = 0

  var c0 = seg0B.x - seg1B.x - seg0A.x + seg1A.x; //*t
  var c1 = seg1B.x - seg1A.x; //*1
  var c2 = seg1B.y - seg0B.y;
  var c3 = this.y - seg1B.y; 
  
  var c4 = seg0B.y - seg1B.y - seg0A.y + seg1A.y; //*t
  var c5 = seg1B.y - seg1A.y; //*1  
  var c6 = seg1B.x - seg0B.x;
  var c7 = this.x - seg1B.x;
  
  var a = c0*c2-c4*c6;
  var b = c1*c2 + c0*c3 - c5*c6 - c4*c7;
  var c = c1*c3 - c5*c7;
   
  if(a == 0) { 
    if(c == 0) {
      return 0.0;
    }
    if(b == 0) { 
      return undefined;
    }
    var ret = -c/b;
    if(ret < 0 || ret > 1)
      return undefined;
    return 1-ret;
  }
  
  var d = b*b - 4*a*c;  

  if(d < 0.0)
    return undefined;
  
  var t0 = (-b + Math.sqrt(d))/(2.0*a);
  var t1 = (-b - Math.sqrt(d))/(2.0*a);
  
  if(t0 < 0 || t0 > 1) { 
    if(t1 < 0 || t1 > 1)
      return undefined;
    return 1 - t1;
  }
  if(t1 < 0 || t1 > 1)
    return 1 - t0;
  if(t0 > t1)
    return 1- t0;
  return 1-t1;

};


if(typeof(module) == 'object' && typeof(module.exports) == 'object') { 
  module.exports.V2 = V2;
  module.exports.V2Util = V2Util;
}




