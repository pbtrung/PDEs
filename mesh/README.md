# gmsh

```
gmsh -v 4 -nt 6 -3 -clmax 0.05 cylinder.geo -o cylinder_mesh_big.msh
gmsh -v 4 -nt 6 -3 -clmax 0.03 cylinder.geo -o cylinder_mesh_medium.msh
gmsh -v 4 -nt 6 -3 -clmax 0.01 cylinder.geo -o cylinder_mesh_small.msh
gmsh -v 4 -nt 6 -3 -clmax 0.005 cylinder.geo -o cylinder_mesh_smallest.msh
```