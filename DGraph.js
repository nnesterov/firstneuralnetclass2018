
var DGraph = { 

  Edge : function() { 
    this.id = -1;
    this.s = undefined;
    this.d = undefined;
    this.data = undefined;
  },

  Node : function() { 
    this.id = -1;
    this.p = [];
    this.c = [];
    this.data = undefined;
    this.tag = -1; 
  },

  EdgeNodeDataPair : function() { 
    this.ndata = undefined;
    this.nid = -1;
    this.edata = undefined;
    this.eid = -1;
  },

  Graph : function() { 
    //when new edges/nodes are created they are given these id's
    this.node_id_ctr = 0;
    this.edge_id_ctr = 0;

    //an array containing all nodes in the graph
    this.n = [];
    //an array containing all edges in the graph
    this.e = [];
  },

  
  //template <typename LAMBDA>
  //void eachEdgeLV(LAMBDA lambda) { 
  //  for(auto & e : e_) { 
  //    lambda(e->data);
  //  }
  //}
  //
  //template <typename LAMBDA>
  //void eachEdgeWithSrcDstNodeDataLV(LAMBDA lambda) { 
  //  for(auto & e : e_) { 
  //    lambda(e->data, e->s->data, e->d->data);
  //  }
  //}
  //
  //template <typename LAMBDA>
  //void eachEdgeWithSrcDstNodeDataLVCV(LAMBDA lambda) const { 
  //  for(auto & e : e_) { 
  //    lambda(e->data, e->s->data, e->d->data);
  //  }
  //}

  //template <typename LAMBDA>
  //void eachChildEdge_NodeRawIdx_LV(int raw_idx, LAMBDA lambda) { 
  //  if(raw_idx < 0 || raw_idx >= (int) n_.size())
  //    return;
  //  for(auto * ch : n_[raw_idx]->c) { 
  //    lambda(ch->data);
  //  }
  //}
  //    
  //template <typename LAMBDA>
  //void eachEdgeWithSrcDstNodeIdsLV(LAMBDA lambda) { 
  //  for(auto * edge : e_) { 
  //    lambda(edge->data, edge->s->id, edge->d->id);
  //  } 
  //}
  //
  //template <typename LAMBDA>
  //void eachEdgeToRawIdx_WithSrcNodeIdsLV(int raw_dst_idx, LAMBDA lambda) { 
  //  for(auto * edge : n_[raw_dst_idx]->p) { 
  //    lambda(edge->data, edge->s->id);
  //  }
  //}
  //
  //template <typename LAMBDA>
  //void eachEdgeFromRawIdx_WithDstNodeIdsLV(int raw_src_idx, LAMBDA lambda) { 
  //  for(auto * edge : n_[raw_src_idx]->c) { 
  //    lambda(edge->data, edge->d->id);
  //  }
  //}

  //template <typename LAMBDA>
  //void eachEdgeFromRawIdx_WithDstNodeIdsLV_ChildIdx(int raw_src_idx, LAMBDA lambda) { 
  //  for(int i = 0; i < (int) n_[raw_src_idx]->c.size(); i++) { 
  //    auto * edge = n_[raw_src_idx]->c[i];
  //    lambda(edge->data, edge->d->id, i);
  //  }
  //}

  //template <typename LAMBDA>
  //void eachNodeChildCt(LAMBDA lambda) { 
  //  for(int i = 0; i < (int) n_.size(); i++)
  //   lambda(n_[i]->id, n_[i]->c.size()); 
  //}

  //void deleteAllEdgesFromTo(int from_id, int to_id) { 
  //  Node * s = nById_(from_id);
  //  Node * d = nById_(to_id);
  //  deleteEdgesFromTo_(s, d);
  //}

  //std::vector<int> unconnectedNodeIds(int src_id, bool search_parent_dir=true, bool search_child_dir=true) { 
  //  std::vector<int> ret;
  //  setAllTags_(0);
  //  Node * s = nById_(src_id);
  //  s->tag = 1;
  //  std::list<Node *> tmp;
  //  tmp.push_back(s);
  //  while(!tmp.empty()) { 
  //    auto * f = tmp.front();
  //    tmp.pop_front();
  //    if(search_child_dir) { 
  //      for(auto * c_e : f->c) { 
  //        if(!c_e->d->tag) { 
  //          c_e->d->tag = 1;
  //          tmp.push_back(c_e->d);
  //        }
  //      }
  //    }
  //    if(search_parent_dir) { 
  //      for(auto * p_e : f->p) { 
  //        if(!p_e->s->tag) { 
  //          p_e->s->tag = 1;
  //          tmp.push_back(p_e->s);
  //        }
  //      }
  //    }
  //  }
  //  for(int i = 0; i < (int) n_.size(); i++) { 
  //    if(!(n_[i]->tag))
  //      ret.push_back(n_[i]->id);
  //  }
  //  return ret;
  //}

  //void deleteAll() { 
  //  for(auto * n : n_)
  //    delete n;
  //  for(auto * e : e_)
  //    delete e;
  //  n_.resize(0);
  //  e_.resize(0);
  //}

};

DGraph.Node.prototype.eachParentNodeDataEdgeDataWithIdx = function(fxn) { 
  for(var i = 0; i < this.p.length; i++)
    fxn(this.p[i].s.data, this.p[i].data, i);
};    
DGraph.Node.prototype.eachParentNodeDataEdgeData = function(fxn) { 
  for(var i = 0; i < this.p.length; i++)
    fxn(this.p[i].s.data, this.p[i].data);
};
DGraph.Node.prototype.eachParentEdge = function(fxn) { 
  for(var i = 0; i < this.p.length; i++)
    fxn(this.p[i]);
};
DGraph.Node.prototype.eachParentNode = function(fxn) { 
  for(var i = 0; i < this.p.length; i++)
    fxn(this.p[i].s);
};

DGraph.Node.prototype.eachChildNodeDataEdgeDataWithIdx = function(fxn) { 
  for(var i = 0; i < this.c.length; i++)
    fxn(this.c[i].d.data, this.c[i].data, i);
};

DGraph.Node.prototype.eachChildNodeDataEdgeData = function(fxn) { 
  for(var i = 0; i < this.c.length; i++)
    fxn(this.c[i].d.data, this.c[i].data);
};

DGraph.Node.prototype.eachChildEdge = function(fxn) { 
  for(var i = 0; i < this.c.length; i++)
    fxn(this.c[i]);
};

DGraph.Node.prototype.eachChildNode = function(fxn) { 
  for(var i = 0; i < this.c.length; i++)
    fxn(this.c[i].d);
};

DGraph.Node.prototype.hasParents = function() { 
  return this.p.length > 0;
};

DGraph.Node.prototype.hasChildren = function() { 
  return this.c.length > 0;
};




///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////




DGraph.Graph.prototype.nodeCt = function() { 
  return this.n.length;
};

DGraph.Graph.prototype.eachNode = function(fxn) { 
  for(var i = 0; i < this.n.length; i++) 
    fxn(this.n[i]);
};

DGraph.Graph.prototype.lastNode = function() { 
  return this.n[this.n.length - 1];
};

DGraph.Graph.prototype.eachEdge = function(fxn) { 
  for(var i = 0; i < this.e.length; i++) 
    fxn(this.e[i]);
};


DGraph.Graph.prototype.addNode = function() { 
  var n0 = new DGraph.Node();
  n0.id = this.node_id_ctr++;
  this.n.push(n0);
  return n0;
};

//TODO...
DGraph.Graph.prototype.createDeepCopy = function() {
  var ret = new DGraph.Graph();
  ret.edge_id_ctr = this.edge_id_ctr;
  ret.node_id_ctr = this.node_id_ctr;
  
  //TODO...

  return ret;    
};


DGraph.Graph.prototype.findNodeById = function(id) { 
  for(var i = 0; i < this.n.length; i++) { 
    if(this.n[i].id == id)
      return this.n[i];
  }
};

DGraph.Graph.prototype.findEdgeById = function(id) { 
  for(var i = 0; i < this.e.length; i++) { 
    if(this.e[i].id == id)
      return this.e[i];
  }
};


DGraph.Graph.prototype.nodesWithoutParents = function() { 
  var ret = [];
  for(var i = 0; i < this.n.length; i++) { 
    if(this.n[i].p.length == 0)
      ret.push(this.n[i]);
  }
  return ret;
};

//aka nodesWithoutChildren
DGraph.Graph.prototype.leafs = function() { 
  var ret = [];
  for(var i = 0; i < this.n.length; i++) { 
    if(this.n[i].c.length == 0)
      ret.push(this.n[i]);
  }
  return ret;
};

//this is used by most of the algos to do DFS, BFS
DGraph.Graph.prototype.setAllTags = function(tagv) { 
  for(var i = 0; i < this.n.length; i++)
    this.n[i].tag = tagv;
};


DGraph.Graph.prototype.reverseEdges = function() { 
  for(var i = 0; i < this.n.length ; i++) { 
    var tmp = this.n[i].c;
    this.n[i].c = this.n[i].p;
    this.n[i].p = tmp;
  }
  for(var i = 0; i < this.e.length; i++) { 
    var tmp = this.e[i].s;
    this.e[i].s = this.e[i].d;
    this.e[i].d = tmp;
  }
};


DGraph.Graph.prototype.pathExistsFromTo = function(from, to) { 
  //TODO
};

DGraph.Graph.prototype.minPathFromTo = function(from, to) { 
  //TODO
};

DGraph.Graph.prototype.minCutFromTo = function(from, to) { 
  //TODO
};

//this is used by neural networks for feed-forward calcs and backprop
DGraph.Graph.prototype.FFOrder = function() { 
  this.setAllTags(0);
  var sources = this.nodesWithoutParents();
  for(var i = 0; i < sources.length; i++)
    sources[i].tag = this.n.length + this.e.length + 1;

  var idx = 0;
  while(idx < sources.length) { 
    sources[idx].eachChildNode(function(node) { 
      node.tag++;
      if(node.tag == node.p.length) { 
        sources.push(node);
      }
    });
    idx++;
  }

  return sources;
};

DGraph.Graph.prototype.directedDFSOrder = function() { 
  this.setAllTags(0);
  var sources = this.nodesWithoutParents();
  var ret = [];
  var stack = [];
  var stack_ctr = 0;
  for(var i = 0; i < sources.length; i++) {
    var cur = sources[i];
    ret.push(cur);
    stack[0] = cur;
    stack_ctr = 0;

    while(1) { 
      
      //increment for a visit  
      cur.tag++;
     
      //we cannot visit children that have already been visited 
      while(cur.tag <= cur.c.length) { 
        if(!cur.c[cur.tag-1].d.tag)
          break;
        cur.tag++;
      }

      //if we cannot visit a child, then go back up
      if(cur.tag <= cur.c.length) {
        cur = cur.c[cur.tag-1].d;
        ret.push(cur);
        stack_ctr++;
        stack[stack_ctr] = cur;
      } else { 
        stack_ctr--;
        if(stack_ctr < 0)
          break;
        cur = stack[stack_ctr];
      }
    }
  }
  return ret;
};
    
DGraph.Graph.prototype.connectByIds = function(id0, id1, silent) {
  var n0 = this.findNodeById(id0);
  var n1 = this.findNodeById(id1);
  if(!n0) {
    if(!silent)
      console.log("ERROR, cannot connect nonexisting src node with id", id0);
    return undefined;
  }
  if(!n1) {
    if(!silent)
      console.log("ERROR, cannot connect nonexisting dst node with id", id1);
    return undefined;
  }
  return this.connectByNodes(n0, n1);
};
    
DGraph.Graph.prototype.connectByArrayIdxs = function(idx0, idx1, silent) {
  if(idx0 < 0 || idx0 >= this.n.length) {
    if(!silent)
      console.log("ERROR, cannot connect nonexisting src node with array idx", idx0, "node ct is", this.n.length);
    return undefined;
  }
  if(idx1 < 0 || idx1 >= this.n.length) {
    if(!silent)
      console.log("ERROR, cannot connect nonexisting dst node with array idx", idx1, "node ct is", this.n.length);
    return undefined;
  }
  return this.connectByNodes(this.n[idx0], this.n[idx1]);
};

DGraph.Graph.prototype.connectByNodes = function(n_src, n_dst) { 
  var edge = new DGraph.Edge();
  edge.id = this.edge_id_ctr++; 
  edge.s = n_src;
  edge.d = n_dst;
  n_src.c.push(edge);
  n_dst.p.push(edge);
  this.e.push(edge);
  return edge;
};



if(typeof(module) == 'object' && typeof(module.exports) == 'object') {
  module.exports.DGraph = DGraph;
};






