
var FullScreenCanvas = function(bg_canv, add_to_el) { 
  
  //hide any scrollbars that might appear due to canvas being slightly too big 
  if(!add_to_el)
    document.body.style.overflow = "hidden";

  //prepare and appent the canvas element
  var canv = document.createElement("canvas");
  //just pick some random size for now
  canv.width = 10;
  canv.height = 10;
  //add it to the body element (very top level)
  if(!bg_canv && !add_to_el)
    document.body.appendChild(canv);
  if(add_to_el) {
    console.log("adding to el");
    add_to_el.appendChild(canv);
  }
  //get the canvas context that we use for rendering
  var ctx = canv.getContext("2d");
    
  //fix the canvas to the top-left corner
  if(!add_to_el) { 
    canv.style.position = "fixed";
    canv.style.top = "0px";
    canv.style.right = "0px";
  }

  this.canv = canv;
  this.ctx = ctx;

  this.offs = new V2(0,0);
  this.up = new V2(0,1);
  this.right = new V2(1,0);
  this.rp = 0;
  //this is a way of converting from world distances to pixel distances
  //this means that a vec of length 1.0 in the world will be length 35px 
  //on the canvas. 
  this.sc = 35;

  this.alphaimg = undefined;
  this.clearOnResize = false;
  var self = this;
  

  //this is going to be the global resize handler
  var handleResize = function() { 
    if(add_to_el) {
      console.log("handling resize add to el");
      return;
    }
    if(bg_canv) { 
      canv.width = window.innerWidth / 2;
      canv.height = window.innerHeight / 2;
    } else { 
      canv.width = window.innerWidth;
      canv.height = window.innerHeight;
    }
    self.alphaimg = ctx.createImageData(canv.width, canv.height);
    for(var i = 0; i < canv.width * canv.height; i++) {
      self.alphaimg.data[4*i+3] = 0; 
    }
    if(self.clearOnResize)
      self.clear();
  };

  //call it ahead of time to properly get the canvas size 
  handleResize();
  
  if(!window.onresize)
    window.onresize = handleResize;
  else {
    var fxn = window.onresize;
    window.onresize = function() { 
      handleResize();
      fxn();
    }
  }
  
  this.onepx = ctx.createImageData(1,1);

   //div.style = "z-index:10";
  if(!add_to_el)
    this.canv.style.zIndex = "1";
  //this.canv.style.zIndex = -1;
    
};

FullScreenCanvas.prototype.forceSize = function(width, height) { 
  this.canv.width = width;
  this.canv.height = height;
}

FullScreenCanvas.prototype.drawImageBG = function(img, border) {
  if(border === undefined)
    border = 0; 
  this.ctx.globalAlpha = 0.8;
  this.ctx.drawImage(img, border, border, this.canv.width-border*2, this.canv.height-border*2);
  this.ctx.globalAlpha = 1.0;
}


FullScreenCanvas.prototype.center = function(v2) { 
  if(v2 !== undefined)
    this.offs = v2.clone();
  return this.offs;
}

//clockwise 
FullScreenCanvas.prototype.rotation = function(r) { 
  this.right = new V2(Math.cos(-r), Math.sin(-r));
  this.up = this.right.clone();
  this.up.rotate90CCW();
  this.rp = r;
}

FullScreenCanvas.prototype.relativeClickCoord = function(evt) { 
  var ret = this.clickCoord(evt);
  ret.minusEq(this.offs);
  return ret;
}

FullScreenCanvas.prototype.clickCoord = function(evt) { 
  var x;
  var y;
  if (evt.pageX || evt.pageY) { 
    x = evt.pageX;
    y = evt.pageY;
  }
  else { 
    x = evt.clientX + document.body.scrollLeft + document.documentElement.scrollLeft; 
    y = evt.clientY + document.body.scrollTop + document.documentElement.scrollTop; 
  } 
  x -= this.canv.offsetLeft;
  y -= this.canv.offsetTop;
  

  x = (x - this.canv.width/2)/this.sc;
  y = (y - this.canv.height/2)/-this.sc;
  var ret = this.right.cloneScale(x).clonePlusScaled(this.up, y).clonePlus(this.offs);
  return ret;
};

FullScreenCanvas.prototype.clear = function() { 
  this.ctx.fillStyle = "#000000";
  this.ctx.fillRect(0,0,this.canv.width, this.canv.height);
};

FullScreenCanvas.prototype.clear2 = function() { 
  this.ctx.clearRect(0,0,this.canv.width, this.canv.height);  
};
FullScreenCanvas.prototype.clear3 = function() { 
  this.ctx.putImageData(this.alphaimg,0,0);  
};


FullScreenCanvas.prototype.rawClearAndDrawRect = function(x,y,w,h, fstyle) { 
  if(fstyle === undefined)
    fstyle = "#000000";
  this.ctx.clearRect(x,y,w,h);
  this.ctx.fillStyle = fstyle;
  this.ctx.fillRect(x,y,w,h);
};


FullScreenCanvas.prototype.text = function(v2, txt) { 
  var tmp = v2.cloneMinus(this.offs);
  var x = tmp.dot(this.right);
  var y = tmp.dot(this.up);
  this.ctx.fillText(txt, x*this.sc + this.canv.width/2, -y*this.sc + this.canv.height/2);
}
          
FullScreenCanvas.prototype.drawImage = function(v2pos, rotpos, img, scl) { 
  var tmp = v2pos.cloneMinus(this.offs);
  tmp.scale(this.sc);

  if(scl === undefined)
    scl = 1.0;
  
  rotpos -= this.rp;

  //TODO: what does the 50 mean here? does 50image px = 1.0 canv units?
  //NOTE: the pixels interact here too. So an 256 px image, will be 256/50 units
  var dsc = scl * this.sc / 50;
  var offs2 = new V2(img.width/2 * dsc, img.height/2 * dsc);
  offs2.rotateCW(-rotpos);
  //var tpos = new V2( this.canv.width/2 - img.width/2 + tmp.dot(this.right) , this.canv.height/2 - img.height/2 - tmp.dot(this.up)); 
  //var tpos = new V2( this.canv.width/2 + tmp.dot(this.right) , this.canv.height/2 - tmp.dot(this.up)); 
  var tpos = new V2( this.canv.width/2 + tmp.dot(this.right) -offs2.x, this.canv.height/2 - tmp.dot(this.up) - offs2.y); 
  
  //now we want to find an x,y point that when rotate by rotpos 
  //will end up at this.canv.width/2 - img.width/2 + x, this.canv.height/2 - img.height/2 - y 
  
  tpos.rotateCW(rotpos);

  //NOTE: the image is always a litte off-kilter for some reason
  this.ctx.rotate(rotpos);
  //TODO: check img.complete?
  this.ctx.drawImage(img,tpos.x, tpos.y, img.width*dsc, img.height*dsc);
  this.ctx.setTransform(1, 0, 0, 1, 0, 0);
}

FullScreenCanvas.prototype.fade = function(alpha) { 
  if(alpha === undefined)
    alpha="0.4";  
  //this.ctx.fillStyle = "rgb(255,255,255,"+alpha+")";
  this.ctx.fillStyle = "#FFFFFF";
  this.ctx.globalAlpha = alpha;
  //console.log("fill", set, this.ctx.fillStyle, alpha);
  this.ctx.fillRect(0,0,this.canv.width, this.canv.height);  
  this.ctx.fill();
  this.ctx.globalAlpha = 1.0;//alpha;
};

FullScreenCanvas.prototype.drawRect = function(v2a, v2b) { 
  v2a = this.screenCoord(v2a);
  v2b = this.screenCoord(v2b);
  this.ctx.strokeRect(Math.min(v2a.x,v2b.x), Math.min(v2a.y, v2b.y), Math.abs(v2a.x-v2b.x), Math.abs(v2a.y - v2b.y));
};

FullScreenCanvas.prototype.fillRect = function(v2a, v2b) { 
  v2a = this.screenCoord(v2a);
  v2b = this.screenCoord(v2b);
  this.ctx.fillRect(Math.min(v2a.x,v2b.x), Math.min(v2a.y, v2b.y), Math.abs(v2a.x-v2b.x), Math.abs(v2a.y - v2b.y));
};

FullScreenCanvas.prototype.screenCoord = function(v2) { 
  var tmp = v2.cloneMinus(this.offs);
  var x = tmp.dot(this.right);
  var y = tmp.dot(this.up);
  return new V2(x*this.sc + this.canv.width/2, -y*this.sc + this.canv.height/2);
};


FullScreenCanvas.prototype.moveTo = function(v2) { 
  var tmp = v2.cloneMinus(this.offs);
  var x = tmp.dot(this.right);
  var y = tmp.dot(this.up);
  this.ctx.moveTo(x*this.sc + this.canv.width/2, -y*this.sc + this.canv.height/2);
};

FullScreenCanvas.prototype.lineTo = function(v2) { 
  var tmp = v2.cloneMinus(this.offs);
  var x = tmp.dot(this.right);
  var y = tmp.dot(this.up);
  this.ctx.lineTo(x*this.sc + this.canv.width/2, -y*this.sc + this.canv.height/2);
}

FullScreenCanvas.prototype.twinkle = function(v2a, color) { 
  this.ctx.beginPath();
  if(color)
    this.ctx.strokeStyle = color;
  else
    this.ctx.strokeStyle = "#AAAAFF";
  //console.log(v2a, v2b);
  for(var i = 0; i < 7; i++) { 
    this.moveTo(v2a);
    this.lineTo(v2a.clonePlusXY( 
          (Math.random()-0.5)*2.0*4.0/this.sc, 
          (Math.random()-0.5)*2.0*4.0/this.sc )); 
  }
  this.ctx.stroke();
}

//0-255 values
FullScreenCanvas.prototype.setPixelRGBA255 = function(r,g,b,a) { 
  this.onepx.data[0] = r;
  this.onepx.data[1] = g;
  this.onepx.data[2] = b;
  this.onepx.data[3] = a;
}

FullScreenCanvas.prototype.pixel = function(v2) { 
  v2 = this.screenCoord(v2);
  v2.round();
  //holy crap this is slow
  this.ctx.putImageData(this.onepx, v2.x, v2.y);
};


FullScreenCanvas.prototype.reducePixel = function(v2, amt) {
  if(!amt)
   amt = 1;
  v2 = this.screenCoord(v2);
  v2.round();
  var px = this.ctx.getImageData(v2.x, v2.y, 1,1);
  //TODO: what happens if we modify px directly?
  this.onepx.data[3]=255;
  for(var i = 0; i < 3; i++) {
    this.onepx.data[i] = px.data[i];
    this.onepx.data[i] -= amt;
  }
  this.ctx.putImageData(this.onepx, v2.x, v2.y);
};

FullScreenCanvas.prototype.statReducePixel = function(v2, amt) {
  if(!amt)
   amt = 1;
  v2 = this.screenCoord(v2);
  v2.round();
  var px = this.ctx.getImageData(v2.x, v2.y, 1,1);
  //TODO: what happens if we modify px directly?
  this.onepx.data[3]=255;
  var sum = 0.0
  for(var i = 0; i < 3 ;i++) { 
    sum += px.data[i];
  }
  if(Math.random() < sum / 255.0 / 3.0) {
    for(var i = 0; i < 3; i++) {
      this.onepx.data[i] = px.data[i];
      this.onepx.data[i] -= amt;
    }
    this.ctx.putImageData(this.onepx, v2.x, v2.y);
  }
};



FullScreenCanvas.prototype.pixelDirect = function(v2) { 
  this.ctx.putImageData(this.onepx, Math.round(v2.x), Math.round(v2.y));
};


FullScreenCanvas.prototype.pixelLine = function(v2a, v2b) { 

  v2a = this.screenCoord(v2a);
  v2b = this.screenCoord(v2b);
  v2a.round();
  v2b.round();
  var xdir = -1
  if(v2a.x < v2b.x)
    xdir = 1;
  var ydir = -1
  if(v2a.y < v2b.y)
    ydir = 1;
  var xoffs = 0;
  var yoffs = 0;
  var dx = v2b.x - v2a.x;
  var dy = v2b.y - v2a.y;
  while(v2a.y+yoffs != v2b.y || v2a.x+xoffs != v2b.x) { 
    this.ctx.putImageData(this.onepx, v2a.x+xoffs, v2a.y+yoffs);
    if(v2a.y+yoffs == v2b.y) { 
      xoffs+= xdir;
    } else if(v2a.x+xoffs == v2b.x) { 
      yoffs += ydir; 
    } else if(Math.abs(dx * yoffs) == Math.abs(dy * xoffs)) { 
      xoffs+=xdir;
      yoffs+=ydir;
    } else if(Math.abs(dx * yoffs) > Math.abs(dy * xoffs)) { 
      xoffs+=xdir;
    } else { 
      yoffs+=ydir;
    }
  }
};

FullScreenCanvas.prototype.addLine = function(v2a, v2b, dr, dg, db) { 

  v2a = this.screenCoord(v2a);
  v2b = this.screenCoord(v2b);
  v2a.round();
  v2b.round();
  var xdir = -1
  if(v2a.x < v2b.x)
    xdir = 1;
  var ydir = -1
  if(v2a.y < v2b.y)
    ydir = 1;
  var xoffs = 0;
  var yoffs = 0;
  var dx = v2b.x - v2a.x;
  var dy = v2b.y - v2a.y;
  while(v2a.y+yoffs != v2b.y || v2a.x+xoffs != v2b.x) { 
    
    var px = this.ctx.getImageData(v2a.x+xoffs, v2a.y+yoffs, 1, 1);
    

    if(dr === undefined) { 


      if(false) { 
        //this adds 1.5 bits of resolution
        if(px.data[0] == 255) { 
          if(px.data[1] == 255) {
            if(px.data[2] < 125) { 
              if(Math.random() < 0.25)
                px.data[2]++;
            } else { 
              if(Math.random() < 0.05)
                px.data[2]++;
            }
          } else {
            if(px.data[1] < 125) 
              px.data[1]++;
            else if(Math.random() < 0.5)  
              px.data[1]++;
          }
        } else 
          px.data[0]++;
      } else { 

        //this uses the full 24 bits, but produces horrible banding
        if(px.data[0] == 255) {
          px.data[0] = 0;
          if(px.data[1] == 255) { 
            px.data[1] = 0;
            px.data[2]++;
          } else  {
            px.data[1]++;
          }
        } else 
          px.data[0]++;
      
      }

    } else { 


      px.data[0] += dr;
      px.data[1] += dg;
      px.data[2] += db;
    
    }
    
    this.ctx.putImageData(px, v2a.x+xoffs, v2a.y+yoffs); 


    if(v2a.y+yoffs == v2b.y) { 
      xoffs+= xdir;
    } else if(v2a.x+xoffs == v2b.x) { 
      yoffs += ydir; 
    } else if(Math.abs(dx * yoffs) == Math.abs(dy * xoffs)) { 
      xoffs+=xdir;
      yoffs+=ydir;
    } else if(Math.abs(dx * yoffs) > Math.abs(dy * xoffs)) { 
      xoffs+=xdir;
    } else { 
      yoffs+=ydir;
    }
  }
};


FullScreenCanvas.prototype.line = function(v2a, v2b, color) { 
  if(color)
    this.ctx.strokeStyle = color;
  else
    this.ctx.strokeStyle = "#000000";
  //console.log(v2a, v2b);
  this.ctx.beginPath();
  this.moveTo(v2a);
  this.lineTo(v2b);
  this.ctx.stroke();
};

//NC means no color. it's annoying that some functions here require a color
//and otherwise provide a default one - someitmes we want to keep the color
//the same for multiple operations without having to store it. 
FullScreenCanvas.prototype.lineNC = function(v2a, v2b) { 
  this.ctx.beginPath();
  this.moveTo(v2a);
  this.lineTo(v2b);
  this.ctx.stroke();
};

FullScreenCanvas.prototype.drawDot = function(v2) { 
  this.ctx.beginPath();
  var tmp = v2.cloneMinus(this.offs);
  var x = tmp.dot(this.right);
  var y = tmp.dot(this.up);
  this.ctx.arc(x*this.sc + this.canv.width/2, -y*this.sc + this.canv.height/2, 2.0, 0, Math.PI*2);
  this.ctx.stroke();
}

FullScreenCanvas.prototype.drawCircle = function(v2, r) { 
  if(r === undefined)
    r = 2.0;
  var tmp = v2.cloneMinus(this.offs);
  var x = tmp.dot(this.right);
  var y = tmp.dot(this.up);
  //this.ctx.fillStyle = "#FFFF00";
  this.ctx.beginPath();
  this.ctx.arc(x*this.sc + this.canv.width/2, -y*this.sc + this.canv.height/2, r*this.sc, 0, Math.PI*2);
  this.ctx.stroke();
}

FullScreenCanvas.prototype.drawCircleFill = function(v2, r) { 
  if(r === undefined)
    r = 2.0;
  var tmp = v2.cloneMinus(this.offs);
  var x = tmp.dot(this.right);
  var y = tmp.dot(this.up);
  //this.ctx.fillStyle = "#FFFF00";
  this.ctx.beginPath();
  this.ctx.arc(x*this.sc + this.canv.width/2, -y*this.sc + this.canv.height/2, r*this.sc, 0, Math.PI*2);
  this.ctx.fill();
}

FullScreenCanvas.prototype.drawCircleGrad = function(v2, r, c0, c1) { 
  if(r == undefined)
    r = 2.0;
  var tmp = v2.cloneMinus(this.offs);
  var x = tmp.dot(this.right)*this.sc + this.canv.width/2;
  var y = -tmp.dot(this.up)*this.sc + this.canv.height/2;
  //this.ctx.fillStyle = "#FFFF00";  
  
  var grd=this.ctx.createRadialGradient(x,y,0,x,y,r*this.sc);
  if(c0)
    grd.addColorStop(0,c0);
  else
    grd.addColorStop(0,"rgba(255,255,200, 40)");
  
  if(c0)
    grd.addColorStop(1,c1);
  else
    grd.addColorStop(1,"rgba(0,0,0,0)");
  
  this.ctx.fillStyle = grd;
  this.ctx.beginPath();
  this.ctx.arc(x, y, r*this.sc, 0, Math.PI*2);
    
  this.ctx.fill();
};

FullScreenCanvas.prototype.drawBurstGrad = function(v2, r, c0, c1, angle) { 
  if(r == undefined)
    r = 2.0;
  if(angle == undefined)
    angle = 0.0;

  var tmp = v2.cloneMinus(this.offs);
  var x = tmp.dot(this.right)*this.sc + this.canv.width/2;
  var y = -tmp.dot(this.up)*this.sc + this.canv.height/2;
  //this.ctx.fillStyle = "#FFFF00";  
 
  var cos = Math.cos(angle);
  var sin = Math.sin(angle);
  var offs = [-1,-1, 1,1, 1,-1, -1,1]
  for(var i = 0; i < offs.length; i+=2){   
    var xt = (offs[i]*cos + offs[i+1]*sin) * this.sc;
    var yt = (-offs[i]*sin + offs[i+1]*cos) * this.sc;

    var grd=this.ctx.createRadialGradient(x,y,0,x+r*xt,y+r*yt,r*this.sc);
    if(c0)
      grd.addColorStop(0,c0);
    else
      grd.addColorStop(0,"rgba(255,255,200, 40)");
    
    if(c0)
      grd.addColorStop(1,c1);
    else
      grd.addColorStop(1,"rgba(0,0,0,0)");
    
    this.ctx.fillStyle = grd;
    this.ctx.beginPath();
    this.ctx.arc(x, y, r*this.sc, 0, Math.PI*2);
      
    this.ctx.fill();
  }
};




FullScreenCanvas.prototype.drawInnerArc = function(ia) { 
  this.ctx.beginPath();
  this.ctx.arc((ia.center.x-this.offs.x) * this.sc + this.canv.width/2, -(ia.center.y-this.offs.y) * this.sc + this.canv.height/2, ia.r0*this.sc, ia.a0, ia.a1);
  //this.ctx.stroke();
  //this.ctx.beginPath();
  this.ctx.arc((ia.center.x-this.offs.x) * this.sc + this.canv.width/2, -(ia.center.y-this.offs.y) * this.sc + this.canv.height/2, ia.r1*this.sc, ia.a1, ia.a0, true);
  this.ctx.closePath();
  this.ctx.stroke();
};

FullScreenCanvas.prototype.arrow = function(pos0, pos1, side_len, shrink_amt, color) { 
  if(!shrink_amt)
    shrink_amt = 0;
  if(!side_len)
    side_len = 0.2;
  if(color !== undefined)
    this.ctx.strokeStyle = color;

  var diff = pos1.cloneMinus(pos0);
  if(shrink_amt) {
    var mag = diff.mag();
    if(shrink_amt > mag *0.8)
      shrink_amt = mag * 0.8;
    pos0 = pos0.clonePlusScaled(diff, 1 / mag * shrink_amt / 2.0);
    pos1 = pos1.clonePlusScaled(diff, -1 / mag * shrink_amt / 2.0);
    diff.scale((mag-shrink_amt) / mag);
  }
  diff.normalize();
  diff.scale(-side_len);
  var diff2 = diff.cloneRotate90CW();
  

 
  this.ctx.beginPath();
  this.moveTo(pos0);
  this.lineTo(pos1);
  this.moveTo(pos1.clonePlus(diff).clonePlus(diff2));
  this.lineTo(pos1);
  diff2.scale(-1);
  this.moveTo(pos1.clonePlus(diff).clonePlus(diff2));
  this.lineTo(pos1);
  this.ctx.stroke();

};

FullScreenCanvas.prototype.renderLineLoop = function(line_loop, color) { 
  this.ctx.beginPath();
  if(color)
    this.ctx.strokeStyle = color;
  else
    this.ctx.strokeStyle = "#444444";
  this.moveTo(line_loop.pts[0]);
  for(var i =0; i < line_loop.pts.length; i++) { 
    this.lineTo(line_loop.pts[i]);
  }
  this.ctx.fillStyle = "#aaaaaa";//rgba(128,128,128,128)";
  this.ctx.fill();
  this.ctx.lineWidth = "2";
  this.ctx.closePath();
  this.ctx.stroke();

};
FullScreenCanvas.prototype.renderLineLoop2 = function(line_loop) { 
  this.ctx.beginPath();
  this.moveTo(line_loop.pts[0]);
  for(var i =0; i < line_loop.pts.length; i++) { 
    this.lineTo(line_loop.pts[i]);
  }
  this.ctx.closePath();
  this.ctx.stroke();
};

FullScreenCanvas.prototype.renderLineLoopOffs = function(ll, radius, f) { 
  if(ll.pts.length >= 2) { 
    this.ctx.beginPath();
    if(ll.pts.length == 2) { 
      var offsA = ll.pts[1].cloneMinus(ll.pts[0]);
      offsA.normalize();
      var offsB = offsA.clone();
      offsB.rotate90CW();
      this.moveTo(ll.pts[0].clonePlusScaled(offsB, radius));
      this.lineTo(ll.pts[0].clonePlusScaled(offsB, -radius));
      this.lineTo(ll.pts[1].clonePlusScaled(offsB, -radius));
      this.lineTo(ll.pts[1].clonePlusScaled(offsB, radius));
    } else {   
      for(var i = 0; i < ll.pts.length; i++) { 
        var left =  ((i == 0) ? ll.pts[ll.pts.length-1] : ll.pts[i-1]);
        var mid = ll.pts[i];
        var right =  ((i == ll.pts.length-1) ? ll.pts[0] : ll.pts[i+1]);
        
        var offs0A = mid.cloneMinus(left);
        offs0A.normalize();
        var offs0B = offs0A.clone();
        offs0B.rotate90CCW();

        var offs1B = right.cloneMinus(mid);
        offs1B.normalize();
        offs1B.rotate90CCW();
        
        var delta_angle = offs0B.angleTo(offs1B);

        if(delta_angle <= 0.0) { 
          var h = -radius / Math.cos(delta_angle/2.0);
          var ox = Math.sin(delta_angle/2.0) * h;
          var np = V2Util.sumScale3(mid, 1.0, offs0B, radius, offs0A, -ox);
          if(i == 0)
            this.moveTo(np);
          else
            this.lineTo(np);
        } else {
          //TODO: do we care about edge cases when the points are too close?  
          var j = 0;
          var np = mid.clonePlusScaled(offs0B, radius);
          if(i == 0)
            this.moveTo(np);
          else
            this.lineTo(np);        
          this.lineTo(mid.clonePlusScaled(offs1B, radius));
        }
      }
    }
    this.ctx.closePath();
    if(!f) { 
      this.ctx.fill();
    } else {
      this.ctx.stroke();
    }
  }
  if(!f ) { 
    for(var i = 0; i < ll.pts.length; i++)
      this.drawCircleFill(ll.pts[i], radius);
  } else { 
    for(var i = 0; i < ll.pts.length; i++)
      this.drawCircle(ll.pts[i], radius);
  }
};


//this draws a grid that takes up the entire visible screen
FullScreenCanvas.prototype.gridHelper = function(spacing, color) { 
  var w = this.canv.width + this.canv.height;
  
  this.ctx.beginPath();
  if(color)
    this.ctx.strokeStyle = color;
  else
    this.ctx.strokeStyle = "#F00000";
  
  var x0 = Math.floor((-w/this.sc/2.0 + this.offs.x) / spacing); 
  var y0 = Math.floor((-w/this.sc/2.0 + this.offs.y) / spacing); 
  for(var v = 0 ; v < Math.ceil(w / this.sc / spacing); v++) { 
    this.moveTo(new V2((x0 + v)*spacing, this.offs.y+w/2.0));
    this.lineTo(new V2((x0 + v)*spacing, this.offs.y-w/2.0));
    this.moveTo(new V2(this.offs.x+w/2.0, (y0 + v)*spacing));
    this.lineTo(new V2(this.offs.x-w/2.0, (y0 + v)*spacing));
  }

  this.ctx.stroke();
};


FullScreenCanvas.prototype.dustHelper2 = function(cell_size, rand_offs, dust_size, style) { 
  
  var x0 = Math.floor((-this.canv.width/this.sc/2.0 + this.offs.x) / cell_size);
  var y0 = Math.floor((-this.canv.height/this.sc/2.0 + this.offs.y) / cell_size);
  var xC = Math.ceil((this.canv.width+0.001) / this.sc / cell_size);
  var yC = Math.ceil((this.canv.height+0.001) / this.sc / cell_size);

  if(xC < 2)
    xC = 2;
  if(yC < 2)
    yC = 2;

  var clr;
  for(var i = 0; i < yC; i++) { 
    for(var j = 0; j < xC; j++) { 
      
      //var xr = 2*(x0+i) + 1;
      //var yr = 2*(y0+i) + 1;
      
      //var xr = 0x9e3779b9*((((x0+j)<<1) ^ ((y0+i) << 17)));
      var xr = 0x9e3779b9*((((x0+j)<<1) ^ ((y0+i) << 17)) + (rand_offs << 7));
      var yr = xr + 0x9e3779b9;
          
      xr = ((xr ^ (xr >>> 16)) * 0x85EBCA6B) >>> 0;
      xr = ((xr ^ (xr >>> 13)) * 0xC2B2AE35) >>> 0;
      xr = (xr ^ (xr >>> 16)) >>> 0;
      
      
      yr = ((yr ^ (yr >>> 16)) * 0x85EBCA6B) >>> 0;
      yr = ((yr ^ (yr >>> 13)) * 0xC2B2AE35) >>> 0;
      yr = (yr ^ (yr >>> 16)) >>> 0;

      if(xr % 2 == 0) { 
        clr = "yellow";
      } else {
        clr = "blue";
      }

      xr = xr /  Math.pow(2,32);
      yr = yr /  Math.pow(2,32);
      //yr /= (1<<32);

      if(style == 0) {
        this.ctx.fillStyle = clr;
        this.drawCircleFill(new V2((x0 + j + xr)*cell_size, (y0 + i + yr)*cell_size), dust_size);
      } else if(style == 1) { 
        this.drawBurstGrad(new V2((x0 + j + xr)*cell_size, (y0 + i + yr)*cell_size), dust_size*10, clr, "rgba(0,0,0,0)");
      }
      //FullScreenCanvas.prototype.drawBurstGrad = function(v2, r, c0, c1, angle) { 
    }
  }
  
};

//This is kind of a hacky way for drawing space dust that is "stationary"
//this is so ghetto. basically it splits the canvas up into cells of "cell_size"
//and for each cell it takes a hash of w/h value and places a point randomly 
//in the cell based on that hash.
FullScreenCanvas.prototype.dustHelper = function(cell_size, ignore1, ignore2, perc_drop) { 
        
  this.setPixelRGBA255(255, 255, 255, 100);

  //if(!scale2)
  //  scale2 = 0.5;
  var scale2 = 0.4;

  var w = this.canv.width + this.canv.height;
  
  var x0 = Math.floor((-w/this.sc/2.0 + this.offs.x) / cell_size);
  var y0 = Math.floor((-w/this.sc/2.0 + this.offs.y) / cell_size);

  for(var x = 0; x < Math.ceil(w / this.sc / cell_size); x++) { 
    for(var y = 0; y < Math.ceil(w / this.sc / cell_size); y++) {       
      
      //this is kind of a iffy hashing funciton...
      var h0 = (x0+x)*2654435761;
      h0 = h0 & h0;
      var h1 = (y0+y)*2654435761;
      h1 = h1 & h1;
      var h2 = ((h0 + 3631) * 3631) ^ h1;
      h2 = h2 & h2;

      //use the hash function to get pseudorandom colors, offsets, etc

      if(h2 % 3 == 1 && ((h2 % 1000) < perc_drop * 1000.0)) { 
        
        if(h2 % 5 <= 3) {
          this.ctx.fillStyle = "#7777FF";
          this.ctx.strokeStyle = "#7777FF";
        } else {
          this.ctx.fillStyle = "#FFFF00";
          this.ctx.strokeStyle = "#FFFF00";
        }
        var xperc = (h1 % 10000)/10000.0;
        var yperc = (h0 % 10000)/10000.0;
        var sz =    (h2 % 10000)/10000.0
      
        //this.twinkle(new V2((x0+x)*cell_size, (y0+y)*cell_size), "#FFFFFF");//, 0.1);
        var pos = new V2((x0+x+xperc)*cell_size, (y0+y+yperc)*cell_size)
        //this.drawCircleFill(pos, (0.1*sz+0.001)*cell_size*scale2);
        //this.line(pos, pos.clonePlusScaled(vel, -1.0*0.05), "#FF0000");

        var pos2 = this.screenCoord(pos);
        this.ctx.fillRect(Math.round(pos2.x), Math.round(pos2.y), 1,1);
        
        //holy crap this is slow:
        //this.pixel(pos);
      }
    }
  }  
};


FullScreenCanvas.prototype.starHelper = function(v2pos, t) { 
  this.ctx.lineWidth = "1";
  this.ctx.beginPath();
  var center = this.screenCoord(v2pos);
  this.ctx.strokeStyle ="#ffffff";
  this.ctx.moveTo(center.x - 5, center.y);
  this.ctx.lineTo(center.x + 5, center.y);
  this.ctx.moveTo(center.x, center.y - 5);
  this.ctx.lineTo(center.x, center.y + 5);
  this.ctx.stroke();
  this.ctx.beginPath();
  this.ctx.strokeStyle ="#9999ff";
  this.ctx.moveTo(center.x - 3, center.y-3);
  this.ctx.lineTo(center.x + 3, center.y+3);
  this.ctx.moveTo(center.x+3, center.y - 3);
  this.ctx.lineTo(center.x-3, center.y + 3);
  this.ctx.stroke();
};


//draw a line on the right edge of the canvas using y pixel coordinates
FullScreenCanvas.prototype.rightEdgePixelLine = function(y0, y1) { 
  this.ctx.beginPath();
  this.ctx.moveTo(this.canv.width-1, y0);
  this.ctx.lineTo(this.canv.width-1, y1);
  this.ctx.stroke();
};

//shift the entire screen using pixel coordinates
FullScreenCanvas.prototype.shiftPixels = function(dx, dy) { 
  var imageData = this.ctx.getImageData(0, 0, this.canv.width, this.canv.height);
  this.ctx.clearRect(0, 0, this.canv.width, this.canv.height);
  this.ctx.putImageData(imageData, dx, dy);
};


