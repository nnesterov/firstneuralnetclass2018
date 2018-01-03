function MtxTable(dst, rows, cols) { 
  
  var tbl = document.createElement("table");
  dst.appendChild(tbl);

  tbl.style = "border:1px solid black; border-collapse: collapse";
  
  this.hide_zeros = true;

  var data = [];
  var remd = false;

  console.log("make mtx", rows,"x",cols);
  for(var i = 0; i < rows; i++) { 
    
    var row = document.createElement("tr");
    row.style = "border:1px solid black; border-collapse: collapse";
    var row2 = [];
    for(var j =0; j < cols; j++) { 
      var cell = document.createElement("td");
      cell.innerHTML = "0";
      row.appendChild(cell);
      row2.push(cell);
      cell.style = "border:1px solid black; border-collapse: collapse; padding:2px";
    }
    data.push(row2);
    tbl.appendChild(row);
  }

  this.remove = function() { 
    if(remd)
      return;
    remd = true;
    dst.removeChild(tbl);
  };

  this.setMtx = function(mtx) { 
    for(var i = 0; i < mtx.h; i++) { 
      for(var j = 0; j < mtx.w; j++) { 
        this.set(i,j, mtx[i][j]);
      }
    }
  };

  this.set = function(i,j,value) { 
    if(!data[i][j]) { 
      console.log(i,j, "oob");
    }
    if(this.hide_zeros && value == 0.0)
      data[i][j].innerHTML = " ";
    else
      data[i][j].innerHTML = value.toFixed(3)+"";
    data[i][j].rawval = value;
  };
  this.get = function(i,j) { 
    return data[i][j].rawval;
  };
};
