#!/usr/bin/env python3
#
# (C) 2018-2020 Ed Bueler

# Create PETSc binary files .vec,.is from ascii .msh mesh file generated by Gmsh.

# This script is based on the Gmsh ASCII format (version 4.1) documented at
#   http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
# and the legacy format (version 2.2) at
#   http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format-version-2-_0028Legacy_0029

# example: put PETSc Vec with locations (node coordinates x,y) in meshes/trap.vec
# and PETSc ISs (e,bfn,s,bfs) in meshes/trap.is
#    $ make petscPyScripts
#    $ gmsh -2 meshes/trap.geo
#    $ ./msh2petsc.py meshes/trap.msh

import numpy as np
import sys

# debug print
def dprint(debug,s):
    if debug:
        print(s)

def fail(k,s):
    print('ERROR: %s ... stopping' % s)
    sys.exit(k)

def get_mesh_format(filename):
    MFread = False
    with open(filename, 'r') as mshfile:
        for line in mshfile:
            line = line.strip()  # remove leading and trailing whitespace
            if line: # only look at nonempty lines
                if line == '$MeshFormat':
                    assert (not MFread), '$MeshFormat repeated'
                    MFread = True
                elif MFread:
                    nums = line.split(' ')
                    gmshversion = nums[0]
                    assert (gmshversion in ['2.2','4.1']), \
                        'unknown Gmsh format version %s' % gmshversion
                    assert (nums[1:] == ['0','8']), 'unexpected MeshFormat data'
                    break
        return gmshversion

# this is the same format for 2.2 and 4.1
def read_physical_names(filename):
    PNread = False
    nPN = 0
    physical = {}   # empty dictionary
    with open(filename, 'r') as mshfile:
        for line in mshfile:
            line = line.strip()  # remove leading and trailing whitespace
            if line: # only look at nonempty lines
                if line == '$PhysicalNames':
                    PNread = True
                elif line == '$EndPhysicalNames':
                    assert (PNread), '$EndPhysicalNames before $PhysicalNames'
                    break
                elif PNread:
                    ls = line.split(' ')
                    if nPN == 0 and len(ls) == 1:
                        try:
                            nPN = int(ls[0])
                        except ValueError:
                            fail(2,'nPN not an integer')
                    else:
                        assert (len(ls) == 3), 'expected three items on line'
                        try:
                            dim = int(ls[0])
                        except ValueError:
                            fail(2,'dim not an integer')
                        try:
                            num = int(ls[1])
                        except ValueError:
                            fail(2,'num not an integer')
                        physical[ls[2].strip('"').lower()] = num
    assert (nPN == len(physical)), 'expected number of physical names does not equal number read'
    for key in ['dirichlet','neumann','interior']:
        assert (key in physical), 'no key "%s" in dictionary' % key
    return physical


#Gmsh format version 2.2 (legacy):
#$Nodes
#number-of-nodes
#node-number x-coord y-coord z-coord       # ignore z-coord
#…
#$EndNodes

def read_nodes_22(filename):
    Nodesread = False
    EndNodesread = False
    N = 0   # number of nodes
    count = 0
    coords = []
    with open(filename, 'r') as mshfile:
        for line in mshfile:
            line = line.strip()  # remove leading and trailing whitespace
            if line: # only look at nonempty lines
                if line == '$Nodes':
                    assert (not Nodesread), '$Nodes repeated'
                    Nodesread = True
                elif line == '$EndNodes':
                    assert (Nodesread), '$EndNodes before $Nodes'
                    assert (len(coords) >= 2), '$EndNodes reached before any nodes read'
                    break  # apparent success reading the nodes
                elif Nodesread:
                    ls = line.split(' ')
                    if len(ls) == 1:
                        assert (N == 0), 'N found again but already read'
                        try:
                            N = int(ls[0])
                        except ValueError:
                            fail(7,'N not an integer')
                        assert (N > 0), 'N invalid'
                        coords = np.zeros(2*N)  # allocate space for nodes
                    else:
                        assert (N > 0), 'expected to read N by now'
                        assert (len(ls) == 4), 'expected to read four values on node line'
                        try:
                            rcount = int(ls[0])
                        except ValueError:
                            fail(10,'node index not an integer')
                        count += 1
                        assert (count == rcount), 'unexpected (noncontiguous?) node indexing'
                        try:
                            xy = [float(s) for s in ls[1:3]]
                        except ValueError:
                            fail(12,'could not convert node coordinates to float')
                        coords[2*(count-1):2*count] = xy            
    assert (count == N), 'N does not agree with index'
    return N,coords


#Gmsh format version 4.1:
#$Nodes
#  numEntityBlocks numNodes minNodeTag maxNodeTag    # use: numNodes
#  entityDim entityTag parametric numNodesInBlock    # use: numNodesInBlock
#                                                    # check: parametric == 0
#    nodeTag
#    ...
#    x(double) y(double) z(double)                   # ignore z
#    ...
#  ...
#$EndNodes

def read_nodes_41(filename):
    Nodesread = False
    EndNodesread = False
    firstlineread = False
    entitylineread = False
    N = 0           # number of nodes
    count = 0       # count of nodes read
    blocksize = 0   # number of nodes in block
    blocknodecount = 0    # count of node tags read in block
    nodetag = []    # node tag as read
    coords = []     # pairs (x-coord, y-coord)
    with open(filename, 'r') as mshfile:
        for line in mshfile:
            line = line.strip()  # remove leading and trailing whitespace
            if line: # only look at nonempty lines
                if line == '$Nodes':
                    assert (not Nodesread), '$Nodes repeated'
                    Nodesread = True
                elif line == '$EndNodes':
                    assert (Nodesread), '$EndNodes before $Nodes'
                    assert (len(coords) >= 2), '$EndNodes reached before any nodes read'
                    break  # apparent success reading the nodes
                elif Nodesread:
                    ls = line.split(' ')
                    assert (len(ls) in [1,3,4]), 'unexpected line format'
                    if len(ls) == 4:
                        if not firstlineread:
                            try:
                                N = int(ls[1])
                            except ValueError:
                                fail(7,'N not an integer')
                            firstlineread = True
                            nodetag = np.zeros(N,dtype=int)   # space for indexing map
                            coords = np.zeros(2*N)            # allocate space for coordinates
                        else:
                            assert (N > 0), 'N not defined'
                            try:
                                blocksize = int(ls[3])
                            except ValueError:
                                fail(8,'numNodesInBlock not an integer')
                            assert (ls[2] == '0'), 'parametric not equal to zero'
                            assert (blocksize <= N - count), 'expected to read fewer nodes'
                            blocknodecount = 0
                            blockcoordscount = 0
                        continue
                    elif len(ls) == 1:
                        assert (firstlineread), 'first line of nodes not yet read'
                        assert (blocknodecount < blocksize), 'not expecting a node tag'
                        try:
                            thistag = int(ls[0])
                        except ValueError:
                            fail(9,'nodeTag not an integer')
                        nodetag[count+blocknodecount] = thistag
                        blocknodecount += 1
                    elif len(ls) == 3:
                        assert (firstlineread), 'first line of nodes not yet read'
                        try:
                            xy = [float(s) for s in ls[0:2]]
                        except ValueError:
                            fail(12,'could not convert node coordinates to float')
                        count += 1
                        coords[2*(count-1):2*count] = xy
    assert (count == N), 'N does not agree with count'
    return N,coords,nodetag


def read_elements_22(filename,N,phys):
    Elementsread = False
    NE = 0   # number of Elements (in Gmsh sense; both triangles and boundary segments)
    tri = []
    ns = []
    bf = np.zeros(N,dtype=int)   # zero for interior
    with open(filename, 'r') as mshfile:
        for line in mshfile:
            line = line.strip()  # remove leading and trailing whitespace
            if line: # only look at nonempty lines
                if line == '$Elements':
                    assert (not Elementsread), '"$Elements" repeated'
                    Elementsread = True
                elif line == '$EndElements':
                    assert (Elementsread), '"$EndElements" before "$Elements"'
                    assert (len(tri) > 0), 'no triangles read'
                    break  # apparent success reading the elements
                elif Elementsread:
                    ls = line.split(' ')
                    if len(ls) == 1:
                        assert (NE == 0), 'NE found again but already read'
                        try:
                            NE = int(ls[0])
                        except ValueError:
                            fail(3,'NE not an integer')
                        assert (NE > 0), 'NE invalid'
                    else:
                        assert (NE > 0), 'expected to read NE by now'
                        assert (len(ls) == 7 or len(ls) == 8), 'expected to read 7 or 8 values on element line'
                        try:
                            dim = int(ls[1])
                        except ValueError:
                            fail(3,'dim not an integer')
                        assert (dim == 1 or dim == 2), 'dim not 1 or 2'
                        try:
                            etype = int(ls[3])
                        except ValueError:
                            fail(3,'etype not an integer')
                        if dim == 2 and etype == phys['interior'] and len(ls) == 8:
                            # reading a triangle
                            try:
                                thistri = [int(s) for s in ls[5:8]]
                            except:
                                fail(3,'unable to convert triangle vertices to integers')
                            # change to zero-indexing
                            tri.append(np.array(thistri,dtype=int) - 1)
                        elif dim == 1 and len(ls) == 7:
                            try:
                                ends = [int(s) for s in ls[5:7]]
                            except:
                                fail(3,'unable to convert segment ends to integers')
                            if etype == phys['dirichlet']:
                                # reading a Dirichlet boundary segment; note zero-indexing
                                bf[np.array(ends,dtype=int) - 1] = 2
                            elif etype == phys['neumann']:
                                # reading a Neumann boundary segment; note zero-indexing
                                ns.append(np.array(ends,dtype=int) - 1)
                                ends = np.array(ends,dtype=int) - 1
                                for j in range(2):
                                    if bf[ends[j]] == 0:
                                        bf[ends[j]] = 1
                            else:
                                fail(3,'should not be here: dim=1 and 7 entries but not etype')
                        else:
                            fail(3,'should not be here: neither triangle or boundary segment')
    return NE,np.array(tri).flatten(),bf,np.array(ns).flatten()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description= \
'''Converts .msh ASCII file from Gmsh into PETSc binary files with .vec
and .is extensions.  Reads both Gmsh file format version 2.2 (legacy) and 4.1.
Needs link to ${PETSC_DIR}/lib/petsc/bin/PetscBinaryIO.py.''')
    # required positional filename
    parser.add_argument('-v', default=False, action='store_true',
                        help='verbose output for debugging')
    parser.add_argument('inname', metavar='FILE',
                        help='input file name with .msh extension')
    args = parser.parse_args()

    import PetscBinaryIO

    if args.inname.split('.')[-1] == 'msh':
        outroot = '.'.join(args.inname.split('.')[:-1]) # strip .msh
    else:
        print('WARNING: expected .msh extension for input file')
    vecoutname = outroot + '.vec'
    isoutname = outroot + '.is'
    gmshversion = get_mesh_format(args.inname)
    print('  input file %s in Gmsh format v%s' % (args.inname,gmshversion))

    print('  reading physical names ...')
    phys = read_physical_names(args.inname)
    dprint(args.v,phys)

    print('  reading node coordinates ...')
    if gmshversion == '2.2':
        N,xy = read_nodes_22(args.inname)
        nodetag = []
    else:
        N,xy,nodetag = read_nodes_41(args.inname)
    dprint(args.v,'N=%d' % N)
    dprint(args.v,xy)
    dprint(args.v and (gmshversion == '4.1'),nodetag)

    print('  writing N=%d node coordinates as PETSc Vec to %s ...' \
          % (N,vecoutname))
    petsc = PetscBinaryIO.PetscBinaryIO()
    petsc.writeBinaryFile(vecoutname,[xy.view(PetscBinaryIO.Vec),])

    if gmshversion == '4.1':
        fail(99,'not implemented')

    print('  reading element tuples ...')
    NE,e,bf,ns = read_elements_22(args.inname,N,phys)
    assert (len(e) % 3 == 0), 'element index list length not 3 K'
    K = len(e) / 3
    assert (len(bf) == N), 'boundary flag list not length N'
    assert (len(ns) % 2 == 0), 'Neumann segment index list length not 2 P'
    P = len(ns) / 2
    if (P == 0):
        print('WARNING: P=0 so writing a bogus negative-valued Neumann boundary segment')
        ns = np.array([-1,-1],dtype=int)
    dprint(args.v,'NE=%d' % NE)
    dprint(args.v,e)
    dprint(args.v,bf)
    dprint(args.v,ns)
    print('  writing K=%d elements, N=%d boundary flags, and P=%d Neumann segments' \
          % (K,N,P))
    print('    as PETSc IS to %s ...' % isoutname)
    IS = PetscBinaryIO.IS
    petsc.writeBinaryFile(isoutname,[e.view(IS),bf.view(IS),ns.view(IS)])

