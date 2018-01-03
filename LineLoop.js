//This class represents a loop of 2D lines using their shared endpoints


//This depends on V2.js
//To quickly simulate a browser load with nodejs, uncomment the 3 lines below
//var fs = require('fs');
//var f = fs.readFileSync('V2.js');
//eval(f+"");

if(typeof(require) == 'function') {
  if(typeof(V2) == 'undefined') {   
    var _v2 = require('./V2.js');
    var V2 = _v2.V2;
    var V2Util = _v2.V2Util;
  }
};



var LineLoopUtil = { 
  
  //This calculates the bounds of an array of lineloops, 
  //taking into account the .offset key if it's present. 
  //The returned format is an array containing the V2 elements
  //[min, max]
  boundsOfArrayWithOffsets : function(loop_arr) { 
  
    var bb_ret = undefined;
    for(var i = 0; i < loop_arr.length; i++) {
      var loop = loop_arr[i];
      var bb = loop.bounds();
      if(loop.offset !== undefined) { 
        bb[0].x -= loop.offset;
        bb[0].y -= loop.offset;
        bb[1].x += loop.offset;
        bb[1].y += loop.offset;
      }
      if(!bb_ret)
        bb_ret = bb;
      else { 
        if(bb[0].x < bb_ret[0].x)
          bb_ret[0].x = bb[0].x;
        if(bb[0].y < bb_ret[0].y)
          bb_ret[0].y = bb[0].y;
        if(bb[1].x > bb_ret[1].x)
          bb_ret[1].x = bb[1].x;
        if(bb[1].y > bb_ret[1].y)
          bb_ret[1].y = bb[1].y;
      }
    }
    return bb_ret;

  },
  
  
  radiusOfArrayWithOffsets : function(loop_arr) { 
  
    var r_ret = 0;
    for(var i = 0; i < loop_arr.length; i++) {
      var loop = loop_arr[i];
      var r = loop.radius();
      if(loop.offset !== undefined)
        r += loop.offset;
      if(r > r_ret)
        r_ret = r;
    }
    return r_ret;
  },



};


var LineLoop = function() { 
  this.pts = [];
};


//make an n-sided polygon of radius 1
LineLoop.prototype.ngon = function(sides) { 
  if(sides <= 0)
    this.pts = [new V2(0,0)];
  else {
    this.pts.length = 0;
    for(var i = 0; i < sides; i++) { 
      var f = i*Math.PI*2.0 / sides;
      this.pts.push(new V2(-Math.cos(f), Math.sin(f)));
    }
  }
};


//get the point which maximized "point.dot(dir)" return
//the point index along with the value 
LineLoop.prototype.maxAlongDir = function(dir, ang_offset) { 
  if(ang_offset)
    dir = dir.cloneRotateCW(-ang_offset);
  var maxd = this.pts[0].dot(dir);
  var maxiz = 0;
  //console.log("PL", this.pts.length);
  for(var i = 1; i < this.pts.length; i++) { 
    var d = this.pts[i].dot(dir);
    if(d > maxd) { 
      //console.log(maxd, maxiz,"->",d,i);
      maxd = d;
      maxiz = i;
    }    
  }
  //console.log("R",maxiz);
  return [maxiz, maxd];
};


//Use the data generated by genHashSupport to find the support point in type O(1) time
//ang_offset means give the result if *this was rotated by ang_offset clockwise, or alternativily:
//if dir was rotated by ang_offset counterclockwise. NOTE: that ang_offset's absolute value should 
//be less than Math.PI * 2.0. Try substituting this function with maxAlongDir to see which gives 
//a performance advantage. 
//OPTIMIZATION NOTES: it appears the shape should be at least 10-15 points long before this 
//function really has any serious advantage in javascript. that's probably because atan2 is 
//way more pricey than doing a bunch of dot products. 
//This also has problems when rotational velocity is very high, then 
//ang_offset ends up being something crazy, and that slows things down.
LineLoop.prototype.maxAlongDirHashSupport = function(dir, ang_offset, canv) { 
  
  //if(true) { //false) { //this.pts.length <= 4) {  
  if(this.pts.length <= 2) {  
    //When there are a few points, rotating the dir vector and doing a few ifs and 
    //dot products is cheaper than atan2 + bounds checks + rounding floats + table indirection
    //Also this means primitive shapes (1pt or 2pt bullets) can be added in great number
    //efficiently (or genHashSupport would need to be called)
    return this.maxAlongDir(dir, ang_offset);
  }

  //return this.maxAlongDir(dir, ang_offset);
  //var dirc = dir.clone();
  var ang = Math.atan2(dir.x, dir.y) - ang_offset;
  
  //console.log("ang", ang / Math.PI, "(offs)", ang_offset);
  if(ang < 0.0) {
    ang += Math.PI * 2.0;
    //console.log("ang2", ang / Math.PI);
    if(ang < 0.0)
      ang += Math.PI * 2.0;
    //a special extra case for ang == Math.PI*2.0 by adding another point to hs_map
    if(ang == Math.PI * 2.0)
      ang = 0.0;
  } else if(ang >= Math.PI * 2.0) { 
    ang -= Math.PI * 2.0;
    if(ang >= Math.PI * 2.0) 
      ang -= Math.PI * 2.0;
  }

  if(!this.hs_map) { 
    this.genHashSupport(this.pts.length*2, canv);
  }

  var aidx = Math.floor(ang * this.hs_map.length / 2.0 / Math.PI);
  //console.log(dir, Math.atan2(dir.x, dir.y), ang, aidx);
  var maxd = -Number.MAX_VALUE;
  var maxi = -1;
  
  //console.log(this.hs_map === undefined);
  //console.log(aidx, ang/Math.PI,this.hs_map.length);
  
  if(aidx < this.hs_map.length && aidx >= 0 && this.hs_map[aidx]) { 
    //ok that's good!
  } else { 
    console.log("ERROR, aidx OOB. angle/pi =", ang/Math.PI, " aidx =", aidx, " hs_map.length =", this.hs_map.length);
    aidx = 0;
  }
  if(this.hs_map[aidx].length == 1) { 
    //return [this.hs_map[aidx][0], this.pts(this.hs_map[aidx][0]).dot(dir)];
    maxi = this.hs_map[aidx][0];
    maxd = this.pts[maxi].dot(dir);
  } else { 
    //need to rotate dir since we'll be actually dotting it with these values to find the max on
    if(ang_offset)
      dir = dir.cloneRotateCW(-ang_offset);
    //iterate over the candidates to find the best one
    for(var i = 0; i < this.hs_map[aidx].length; i++) { 
      var j = this.hs_map[aidx][i];
      var d = this.pts[j].dot(dir);
      //console.log("maxc", j, d);
      if(d > maxd) { 
        maxd = d;
        maxi = j;
      }
      //debug rendering:
      //canv.line(new V2(0,0), dir, "white");
      //canv.ctx.strokeStyle = "red";
      //canv.drawCircle(this.pts[j], 0.1);
    }
  }
  
  //var bbtest = this.maxAlongDir(dirc, ang_offset);
  //if(bbtest[0] != maxi && dir.mag() > 0) { 
  //  console.log("hsmap", this.hs_map[aidx]);
  //  console.log("ERROR, diff res", bbtest[0], bbtest[1], "cur", maxi,maxd, "mag", dir.mag(), "ang", ang/Math.PI);
  //  //exit;
  //}

  //more debug rendeirng:
  //dir = dir.clone();
  //dir.rotateCW(-ang_offset);
  //var tt = this.maxAlongDir(dir);
  //canv.ctx.strokeStyle = "green";
  //canv.drawCircle(this.pts[tt[0]], 0.15);
  //canv.ctx.strokeStyle = "orange";
  //canv.drawCircle(this.pts[maxi], 0.18);
  
  return [maxi, maxd];
};

//This generates a hashmap using atan2(x,y) for the support function of a convex shape
//Basically it caches the farthest points along (x,y) splitting up the values based on
//atan2(x,y). This is used purely for optimization and the values are not copied upon
//cloning. 
//TODO: this only works for convex shapes ordered in clockwise fashion. There's no reason
//this can't work for a generic point clould, which would be much more ideal.
LineLoop.prototype.genHashSupport = function(segments, canv) { 
  if(segments < 4)
    segments = 4;
  this.hs_map = [];
  //TODO: is this right?!?
  for(var i = 0; i < segments; i++) { 
    var a0 = 2.0*Math.PI * (i) / segments;  
    var a1 = 2.0*Math.PI * (i+1.0) / segments;  
    var axis0 = new V2(Math.sin(a0), Math.cos(a0));
    var axis1 = new V2(Math.sin(a1), Math.cos(a1));
    //canv.offs.x += 7;
    //canv.line(new V2(0,0), axis0, 'red');
    //console.log(a0, a1, axis1, axis0, this.pts[0], "------------");
    var s0 = this.maxAlongDir(axis0);
    var s1 = this.maxAlongDir(axis1);

    var j = s0[0];
    this.hs_map[i] = [];
    while(1) { 
      this.hs_map[i].push(j);
      if(j == s1[0])
        break;
      j++;
      if(j == this.pts.length)  
        j = 0;
    }
    //console.log(s0[0], s1[0], "---", i, this.hs_map[i]);
    //canv.line(axis0, this.pts[s0[0]], 'red');
    //canv.line(this.pts[s1[0]], this.pts[s0[0]], 'red');
    //canv.offs.x -= 7;
  }
  //exit;
};


//Sometimes the shape being clockwise or counter-clockwise matters. This
//let's us quickly change that property. 
LineLoop.prototype.reverse = function() { 
  for(var i = 0; i < Math.floor(this.pts.length/2); i++) { 
    var tmp = this.pts[i];
    this.pts[i] = this.pts[this.pts.length - i - 1];
    this.pts[this.pts.length - i - 1] = tmp;
  }
};

//This function extends the shape with mirrored points.
//As an example: if the shape consists of points a,b,c,d it then 
//adds points so that the shape contains a,b,c,d,-d,-c,-b,-a where
//-PT represents the PT mirrored about v2_axis. note: v2_axis is 
//perpindicular to the mirror plane. 
LineLoop.prototype.addMirroredPoints = function(v2_axis) { 
  var v2_normal = v2_axis.clone();
  v2_normal.rotate90CW();

  for(var i = this.pts.length - 1; i >= 0; i--) { 
    var addee = this.pts[i].clone();
    addee.reflectWithNormal( v2_normal );
    this.push( addee );
  }

};

//As an example, if v2_axis is (0,1) the points will be 
//mirrored about the y-axis, effectively performing pt[i].x *= -1.0
//for each point. note: v2_axis is perpindicular to the mirror plane. 
LineLoop.prototype.mirror = function(v2_axis) { 
  v2_normal = v2_axis.clone();
  v2_normal.rotate90CW();

  for(var i = 0; i < this.pts.length; i++)
    this.pts[i].reflectWithNormal(v2_normal);  
};


LineLoop.prototype.push = function() { 
  if(arguments.length == 2)
    this.pts.push(new V2(arguments[0], arguments[1]));
  else
    this.pts.push(arguments[0]);
};

LineLoop.prototype.popLast = function() { 
  if(this.pts.length)
    this.pts.length--;
}

//sum the distance between all consecutive pairs of points,
//including the pt at the end of the array and the beginning (full loop)
LineLoop.prototype.totalLength = function() { 
  var r = 0;
  if(this.pts.length <= 1)
    return 0.0;
  for(var i = 0; i < this.pts.length-1; i++)
    r += this.pts[i].dist(this.pts[i+1]); //sum up all the distances 
  return r + this.pts[this.pts.length-1].dist(this.pts[0]);//close the loop
};

LineLoop.prototype.clone = function() { 
  var ret = new LineLoop();
  ret.pts.length = this.pts.length;
  for(var i = 0; i < this.pts.length; i++) {
    ret.pts[i] = this.pts[i].clone();
  }
  if(this.offset !== undefined)
    ret.offset = this.offset;
  return ret;
};

//return a two element array containing the min and max bounds as V2's
LineLoop.prototype.bounds = function() { 
  var min = this.pts[0].clone();
  var max = this.pts[0].clone();
  for(var i = 1; i < this.pts.length; i++) { 
    if(this.pts[i].x > max.x)
      max.x = this.pts[i].x;
    if(this.pts[i].x < min.x)
      min.x = this.pts[i].x;
    if(this.pts[i].y > max.y)
      max.y = this.pts[i].y;
    if(this.pts[i].y < min.y)
      min.y = this.pts[i].y;
  } 
  return [min,max];
};

LineLoop.prototype.projectedBounds = function(dir0) { 
  var min = this.pts[0].projected0(dir0);
  var max = this.pts[0].projected0(dir0);
  for(var i = 1; i < this.pts.length; i++) { 
    var pt = this.pts[i].projected0(dir0);
    if(pt.x > max.x)
      max.x = pt.x;
    if(pt.x < min.x)
      min.x = pt.x;
    if(pt.y > max.y)
      max.y = pt.y;
    if(pt.y < min.y)
      min.y = pt.y;
  } 
  return [min,max];
};

//note we could also add an optimized projectedBounds function that uses the hash-support
//which would work really well for shapes with lots of points

//This calculates the maximum distance between a point and (0,0) which is the radius of 
//this lineloop. Note: this ignored the .offset property if it's present
LineLoop.prototype.radius = function() { 
  var max = 0.0;
  for(var i = 0; i < this.pts.length; i++) {
    var r = this.pts[i].mag();
    if(r > max)
      max = r;
  }
  return max;    
};

//Find the minimum radius value for all points along the lineloop, including
//points along the connecting segments. Note: the minimum radius might be constrained by a segment
//so this function might take 
LineLoop.prototype.innerRadius = function() { 
  if(!this.pts.length)
    return Number.MAX_VALUE;
  var min_r = this.pts[0].mag();
  if(this.pts.length == 1)
    return min_r;
  for(var i = 0; i < this.pts.length-1; i++) { 
    var r = V2Util.minOnSegment(this.pts[i], this.pts[i+1]);
    if(r < min_r)
      min_r = r;
  }
  var r = V2Util.minOnSegment(this.pts[this.pts.length-1], this.pts[0]);
  if(r < min_r)
    min_r = r;
  return min_r;
};

//this calculates the minimum radius but only checks the endpoints, which is 
//not technically correct, but works in some cases and is much cheaper to calc.
LineLoop.prototype.innerRadiusAlt = function() { 
  var min = Number.MAX_VALUE;
  for(var i = 0; i < this.pts.length; i++) {
    var r = this.pts[i].mag();
    if(r < min)
      min = r;
  }
  return min;    
};

//scale along a direction only 
LineLoop.prototype.dirScale = function(dir, sc) { 
  //we need dir to be normalized 
  dir.normalize();
  var rdir = dir.clone();
  rdir.rotate90CW();
  for(var i = 0; i < this.pts.length; i++) { 
    var x0 = this.pts[i].dot(dir);
    var x1 = this.pts[i].dot(rdir);
    this.pts[i].x = dir.x * x0 * sc + rdir.x * x1;
    this.pts[i].y = dir.y * x0 * sc + rdir.y * x1;
  }
};

LineLoop.prototype.cloneScale = function(sc) { 
  var ret = this.clone();
  for(var i = 0; i < this.pts.length; i++)
    ret.pts[i].scale(sc);
  return ret;
};

LineLoop.prototype.scale = function(sc) { 
  for(var i = 0; i < this.pts.length; i++)
    this.pts[i].scale(sc);
};

LineLoop.prototype.plusEq = function(v2) { 
  for(var i = 0; i < this.pts.length; i++)
    this.pts[i].plusEq(v2);
};

LineLoop.prototype.minusEq = function(v2) { 
  for(var i = 0; i < this.pts.length; i++)
    this.pts[i].minusEq(v2);
};

LineLoop.prototype.plusEqScaled = function(v2, sc) { 
  for(var i = 0; i < this.pts.length; i++)
    this.pts[i].plusEqScaled(v2, sc)
}

LineLoop.prototype.clonePlus = function(v2) { 

  var ret = new LineLoop();
  ret.pts.length = this.pts.length;
  for(var i = 0; i < this.pts.length; i++) {
    ret.pts[i] = this.pts[i].clonePlus(v2);
  }
  return ret;

};

LineLoop.prototype.clonePlusXY = function(x,y) { 

  var ret = new LineLoop();
  ret.pts.length = this.pts.length;
  for(var i = 0; i < this.pts.length; i++) {
    ret.pts[i] = this.pts[i].clonePlusXY(x, y);
  }
  return ret;

};

LineLoop.prototype.cloneMinus = function(v2) { 
  var ret = new LineLoop();
  ret.pts.length = this.pts.length;
  for(var i = 0; i < this.pts.length; i++) {
    ret.pts[i] = this.pts[i].cloneMinus(v2);
  }
  return ret;
};

LineLoop.prototype.rotateCW = function(radians) { 
  
  for(var i = 0; i < this.pts.length; i++) { 
    this.pts[i].rotateCW(radians);
  }

};

LineLoop.prototype.rotateCWPlus = function(radians, offset) { 
  for(var i = 0; i < this.pts.length; i++) { 
    this.pts[i].rotateCWPlus(radians, offset);
  }
};

LineLoop.prototype.cloneRotateCWPlus = function(radians, offset) { 
  var ret = new LineLoop();
  ret.pts.length = this.pts.length;
  for(var i = 0; i < this.pts.length; i++) { 
    ret.pts[i] = this.pts[i].cloneRotateCWPlus(radians, offset);
  }
  return ret;
}

//returns true if v2 is inside the loop
LineLoop.prototype.inside = function(v2) { 
  
  var tot_angle = 0.0;
  for(var i = 0; i < this.pts.length - 1; i++) {     
    tot_angle += V2Util.angleABC(this.pts[i], v2, this.pts[i+1]);
  }
  tot_angle += V2Util.angleABC(this.pts[this.pts.length-1], v2, this.pts[0]);
  //Details: if v2 is inside the loop the sum of the angles should 2.0 * Math.PI,
  //but if v2 is outside the loop the sum should be 0.0
  if(Math.abs(tot_angle) < Math.PI)
    return false;
  return true;

};

//This is a partial intersection check. Actually two shapes can intersect without
//having points inside of each other, but that's quite rare if both shapes have a 
//lot of points. This also handles the nice cases of one shape completely containing
//the other shape. NOTE: this is slow.
LineLoop.prototype.pointsInsideTwoWay = function(other) { 
  //this O(N^2) is quite brutal because it calculates the angle to each edge
  //which takes one atan2 operation per vertex and a bunch of other math too
  for(var i = 0; i < other.pts.length; i++) { 
    if(this.inside(other.pts[i])) {
      return true;
    }
  } 
  for(var i = 0; i < this.pts.length; i++) { 
    if(other.inside(this.pts[i])) {
      return true;
    }
  }
  return false;
};


//Find the nearest segment defined by a pair of connected points on this lineloop 
//as measured by distance to the point v2. 
//Returns and array containing 4 values corresponding to the segment which is nearest to v2:
// [ cw_first_point_of_seg, closest_point_on_seg, cw_second_point_of_seg, idx_of_seg ]
//
LineLoop.prototype.nearestOnLoop = function(v2) {

  var min_dist2 = Number.MAX_VALUE;
  var near_pts = undefined;
  for(var i = 0; i < this.pts.length-1; i++) { 
    var pt = v2.nearestPointOnSegment(this.pts[i], this.pts[i+1]);
    var d = v2.dist2(pt);
    if(d < min_dist2) {
      min_dist2 = d;
      near_pts = [this.pts[i], pt, this.pts[i+1], i];
    }
  }
  var pt = v2.nearestPointOnSegment(this.pts[this.pts.length -1], this.pts[0]);
  var d = v2.dist2(pt);
  if(d < min_dist2) {
    min_dist2 = d;
    near_pts = [this.pts[this.pts.length-1], pt, this.pts[0], this.pts.length-1];
  }
  return near_pts;

};


//positive if the line loop is clockwise
LineLoop.prototype.signedArea = function() { 
  var sum_area = 0.0;
  var zero = new V2(0,0);
  for(var i = 0; i < this.pts.length -1 ; i++) 
    sum_area += V2Util.signedAreaTriangle(zero, this.pts[i], this.pts[i+1]);
  sum_area += V2Util.signedAreaTriangle(zero, this.pts[this.pts.length-1], this.pts[0]);
  return sum_area;
};

var LineLoop_ConvexOffsetCfg = { 
  tolerance : 0.01,
  type: 1, //1 = inside, 2 = outside (todo), 3 = min error (todo)
};

//this creates a positive offset shape for all convex clockwise shapes
//and for some non convex ones too (no garuntees!) and will do negative
//offsets for some CCW shapes as well (again no garuntees!) this is not
//even suitable for generating a shape than can be refined for either of
//those two extra cases. for those other cases a completely different 
//routine that uses pure offsetting and intersections is needed. 
//TODO: 
//we may want to make use of an efficient version of this for filling a convex shape's 
//background. to do that we first offset, but without arc interpolation, then fill. then fill
//a bunch of circles at the points, this gives us a really fast way to fill the offset shape
//accurately. 
LineLoop.prototype.offsetConvexCW = function( radius ) {
  //for inside type
  //  err = r*(1 - cos(angle_step/2))
  var angle_step = Math.acos(1 - LineLoop_ConvexOffsetCfg.tolerance /  radius )
  //console.log(angle_step);
  
  var ret = new LineLoop();
  if(this.pts.length == 1 || (this.pts.length == 2 && this.pts[0].dist2(this.pts[1]) == 0.0)) { 
    for(var i = 0; i < Math.PI * 2.0; i+= angle_step) { 
      ret.push(Math.sin(i) * radius + this.pts[0].x, Math.cos(i) * radius + this.pts[0].y);
    }
  } else if(this.pts.length == 2) { 
    var offsA = this.pts[1].cloneMinus(this.pts[0]);
    offsA.normalize();
    var offsB = offsA.clone();
    offsB.rotate90CW();
    ret.push(this.pts[0].clonePlusScaled(offsB, radius));
    for(var i = 0; i < Math.PI; i += angle_step) { 
      ret.push(V2Util.sumScale3( this.pts[0], 1.0,  offsB, Math.cos(i)*radius, offsA, -Math.sin(i)*radius ));
    }
    ret.push(this.pts[0].clonePlusScaled(offsB, -radius));
    ret.push(this.pts[1].clonePlusScaled(offsB, -radius));
    for(var i = 0; i < Math.PI; i += angle_step) { 
      ret.push(V2Util.sumScale3( this.pts[1], 1.0,  offsB, -Math.cos(i)*radius, offsA, Math.sin(i)*radius ));
    }
    ret.push(this.pts[1].clonePlusScaled(offsB, radius));
  } else { 
  
    for(var i = 0; i < this.pts.length; i++) { 
      var left =  ((i == 0) ? this.pts[this.pts.length-1] : this.pts[i-1]);
      var mid = this.pts[i];
      var right =  ((i == this.pts.length-1) ? this.pts[0] : this.pts[i+1]);
      
      var offs0A = mid.cloneMinus(left);
      offs0A.normalize();
      var offs0B = offs0A.clone();
      offs0B.rotate90CCW();

      var offs1B = right.cloneMinus(mid);
      offs1B.normalize();
      offs1B.rotate90CCW();
      
      var delta_angle = offs0B.angleTo(offs1B);

      if(delta_angle < 0.0) { 
        //not convex... oops
        //for some simple shapes this can still work.
        //another alternative is to offset both lines
        //and find the intersection point (doesn't work well for tiny angles)
        var h = -radius / Math.cos(delta_angle/2.0);
        var ox = Math.sin(delta_angle/2.0) * h;
        ret.push(V2Util.sumScale3(mid, 1.0, offs0B, radius, offs0A, -ox));
      } else if(Math.abs(delta_angle) < angle_step/4.0) {
        //if there's no angle, don't do arc interpolation
        var addee = offs0B.clonePlus(offs1B);
        addee.normalize();
        addee.scale(radius);
        addee.plusEq(mid);
        ret.push(addee);
      } else {
        //TODO: do we care about edge cases when the points are too close?  
        for(var j = 0; j < delta_angle; j += angle_step) {
          //if(j > delta_angle - angle_step / 2.0)
          //  j -= angle_step / 4.0;
          ret.push(V2Util.sumScale3(mid, 1.0, offs0B, Math.cos(j)*radius, offs0A, Math.sin(j)*radius));
        }
        ret.push(mid.clonePlusScaled(offs1B, radius));
      }
      
    }
  }
  return ret;
  
};


LineLoop.prototype.convex = function() { 
  if(this.pts.length < 2)
    return true;
  var sgn = 0;
  for(var i = 0; i < this.pts.length; i++) { 
    var left =  ((i == 0) ? this.pts[this.pts.length-1] : this.pts[i-1]);
    var mid = this.pts[i];
    var right =  ((i == this.pts.length-1) ? this.pts[0] : this.pts[i+1]);
    //right-this.dot(this.minus.left.rotate90CW) should always be positive
    var dr = (right.x - mid.x) * ( left.y - mid.y ) + (right.y - mid.y ) * (mid.x - left.x);
    if(sgn == 0.0)
      sgn = dr;
    else if(sgn * dr < 0.0)
      return false;
  }
  return true;
};

//just return the average of all the points
LineLoop.prototype.avg = function() { 
  var ret = new V2(0,0);
  for(var i = 0; i < this.pts.length; i++)
    ret.plusEq(this.pts[i]);
  if(this.pts.length)
    ret.scale(1.0/this.pts.length);
  return ret;
};

//assumes that mass is evenly distributed and proportional to surface area
LineLoop.prototype.centerOfMass = function() { 
  if(this.pts.length == 1)
    return this.pts[0].clone();
  if(this.pts.length == 2)
    return new V2(this.pts[0].x*0.5 + this.pts[1].x*0.5, this.pts[0].y*0.5 + this.pts[1].y*0.5);
  var sum_area = 0.0;
  var area_times_pos = new V2(0,0);
  var zero = new V2(0,0);
  var tri_area;
  for(var i = 0; i < this.pts.length -1 ; i++) {
    tri_area = V2Util.signedAreaTriangle(zero, this.pts[i], this.pts[i+1]);
    sum_area += tri_area;
    area_times_pos.plusEqScaled( V2Util.centroidTriangle(zero, this.pts[i], this.pts[i+1]), tri_area);
  }
  tri_area = V2Util.signedAreaTriangle(zero, this.pts[this.pts.length-1], this.pts[0]);
  sum_area += tri_area
  area_times_pos.plusEqScaled( V2Util.centroidTriangle(zero, this.pts[this.pts.length-1], this.pts[0]), tri_area);
  area_times_pos.divide( sum_area );
  return area_times_pos;
};



//calculate moment of inertia about V2(0,0)
LineLoop.prototype.momentOfInertia = function() { 
  var sum_inertia = 0.0;
  var zero = new V2(0,0);
  for(var i = 0; i < this.pts.length-1; i++) { 
    sum_inertia += V2Util.momentOfInertiaTriangle(this.pts[i], this.pts[i+1], zero); 
  }
  sum_inertia += V2Util.momentOfInertiaTriangle(this.pts[this.pts.length-1], this.pts[0], zero); 
  return sum_inertia;
};


LineLoop.prototype.rayHit = function(ray0, raydir) { 
  var min_d = Number.MAX_VALUE;
  var min_p0;
  var min_p1;
  for(var i = 0; i < this.pts.length; i++) { 
    var p0 = this.pts[i];
    var p1 = this.pts[(i+1)%this.pts.length];
    var t = V2Util.rayHit(ray0, raydir, p0, p1);
    if(t !== undefined && t < min_d) { 
      min_d = t;
      min_p0 = p0;
      min_p1 = p1;
    }
  }
  if(min_p0) { 
    return [min_d, min_p0, min_p1];
  }
  return undefined;
};

if(typeof(module) == 'object' && typeof(module.exports) == 'object') { 
  module.exports.LineLoop = LineLoop;
  module.exports.LineLoopUtil = LineLoopUtil;
};
