SetFactory("OpenCASCADE");

// Parameters
r = 1.0;      // Radius of the cylinder
l = 2.0;      // Height of the cylinder
rtop = 0.4;   // Radius of the smaller circle on top

Mesh.MshFileVersion = 2.2;

// Create the cylinder
Cylinder(1) = {0, 0, 0, 0, 0, l, r};

// Create points and arcs to define the smaller circle on top of the cylinder
Point(100) = {0, 0, l};
Point(101) = {rtop, 0, l};
Point(102) = {0, rtop, l};
Point(103) = {-rtop, 0, l};
Point(104) = {0, -rtop, l};

Circle(201) = {101, 100, 102};
Circle(202) = {102, 100, 103};
Circle(203) = {103, 100, 104};
Circle(204) = {104, 100, 101};

Curve Loop(301) = {201, 202, 203, 204};
Plane Surface(302) = {301};

BooleanDifference(400) = {Surface{2}; Delete;}{Surface{302};};

// Define physical groups
Physical Surface("TopBoundary") = {302};
Physical Volume(1) = {1};
Physical Surface(2) = {1};
Physical Surface(3) = {400};
Physical Surface(4) = {3};

// Generate the 3D mesh
Mesh 3;
