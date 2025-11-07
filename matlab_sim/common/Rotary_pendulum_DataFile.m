% Simscape(TM) Multibody(TM) version: 23.2

% This is a model data file derived from a Simscape Multibody Import XML file using the smimport function.
% The data in this file sets the block parameter values in an imported Simscape Multibody model.
% For more information on this file, see the smimport function help page in the Simscape Multibody documentation.
% You can modify numerical values, but avoid any other changes to this file.
% Do not add code to this file. Do not edit the physical units shown in comments.

%%%VariableName:smiData


%============= RigidTransform =============%

%Initialize the RigidTransform structure array by filling in null values.
smiData.RigidTransform(10).translation = [0.0 0.0 0.0];
smiData.RigidTransform(10).angle = 0.0;
smiData.RigidTransform(10).axis = [0.0 0.0 0.0];
smiData.RigidTransform(10).ID = "";

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(1).translation = [-28.500000000000004 0 0];  % mm
smiData.RigidTransform(1).angle = 2.0943951023931953;  % rad
smiData.RigidTransform(1).axis = [0.57735026918962584 0.57735026918962584 0.57735026918962584];
smiData.RigidTransform(1).ID = "B[Module-1:-:Pivot-1]";

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(2).translation = [-27.00035691063016 1.2656542480726785e-14 -0.00034155424519988209];  % mm
smiData.RigidTransform(2).angle = 2.0943951023931953;  % rad
smiData.RigidTransform(2).axis = [0.57735026918962584 0.57735026918962584 0.57735026918962584];
smiData.RigidTransform(2).ID = "F[Module-1:-:Pivot-1]";

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(3).translation = [9.0285119292874902e-05 -10.000000000000009 0.00032362392247964333];  % mm
smiData.RigidTransform(3).angle = 2.0943951023931953;  % rad
smiData.RigidTransform(3).axis = [-0.57735026918962584 -0.57735026918962584 -0.57735026918962584];
smiData.RigidTransform(3).ID = "B[Pendulum-1:-:Pivot-1]";

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(4).translation = [96.399643089369832 -6.2172489379008766e-15 -0.00034155424513260257];  % mm
smiData.RigidTransform(4).angle = 2.6132803927706347;  % rad
smiData.RigidTransform(4).axis = [0.68075044320639466 0.27047674234320807 0.68075044320639477];
smiData.RigidTransform(4).ID = "F[Pendulum-1:-:Pivot-1]";

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(5).translation = [0 0 0];  % mm
smiData.RigidTransform(5).angle = 0;  % rad
smiData.RigidTransform(5).axis = [0 0 0];
smiData.RigidTransform(5).ID = "B[Base-1:-:]";

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(6).translation = [0 0 0];  % mm
smiData.RigidTransform(6).angle = 0;  % rad
smiData.RigidTransform(6).axis = [0 0 0];
smiData.RigidTransform(6).ID = "F[Base-1:-:]";

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(7).translation = [0 0 0];  % mm
smiData.RigidTransform(7).angle = 0;  % rad
smiData.RigidTransform(7).axis = [0 0 0];
smiData.RigidTransform(7).ID = "B[Base-1:-:Circle-1]";

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(8).translation = [0 0 0];  % mm
smiData.RigidTransform(8).angle = 0;  % rad
smiData.RigidTransform(8).axis = [0 0 0];
smiData.RigidTransform(8).ID = "F[Base-1:-:Circle-1]";

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(9).translation = [0 0 5];  % mm
smiData.RigidTransform(9).angle = 0;  % rad
smiData.RigidTransform(9).axis = [0 0 0];
smiData.RigidTransform(9).ID = "B[Circle-1:-:Module-1]";

%Translation Method - Cartesian
%Rotation Method - Arbitrary Axis
smiData.RigidTransform(10).translation = [19.000000000000405 -1.5543122344752192e-15 -11.5];  % mm
smiData.RigidTransform(10).angle = 3.1415926535897931;  % rad
smiData.RigidTransform(10).axis = [0 0 -1];
smiData.RigidTransform(10).ID = "F[Circle-1:-:Module-1]";


%============= Solid =============%
%Center of Mass (CoM) %Moments of Inertia (MoI) %Product of Inertia (PoI)

%Initialize the Solid structure array by filling in null values.
smiData.Solid(5).mass = 0.0;
smiData.Solid(5).CoM = [0.0 0.0 0.0];
smiData.Solid(5).MoI = [0.0 0.0 0.0];
smiData.Solid(5).PoI = [0.0 0.0 0.0];
smiData.Solid(5).color = [0.0 0.0 0.0];
smiData.Solid(5).opacity = 0.0;
smiData.Solid(5).ID = "";

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(1).mass = 5.0;  % kg
smiData.Solid(1).CoM = [0 0 -58.500000000000007];  % mm
smiData.Solid(1).MoI = [2443.9698269999985 2443.9698269999985 2110.7427119999988];  % kg*mm^2
smiData.Solid(1).PoI = [0 0 0];  % kg*mm^2
smiData.Solid(1).color = [0.65098039215686276 0.61960784313725492 0.58823529411764708];
smiData.Solid(1).opacity = 1;
smiData.Solid(1).ID = "Base*:*Default";

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(2).mass = 0.0625;  % kg
smiData.Solid(2).CoM = [0.00010408012550415582 -5.8319867968102008e-06 58.337751643752974];  % mm
smiData.Solid(2).MoI = [11.666524525457579 11.667790908160166 0.10003017193753305];  % kg*mm^2
smiData.Solid(2).PoI = [-2.7527127082215587e-06 5.8596256406177077e-06 -9.4098568246295415e-06];  % kg*mm^2
smiData.Solid(2).color = [0.792156862745098 0.81960784313725488 0.93333333333333335];
smiData.Solid(2).opacity = 1;
smiData.Solid(2).ID = "Pendulum*:*Default";

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(3).mass = 0.0425;  % kg
smiData.Solid(3).CoM = [55.574643089369857 0 -0.00034155424519351891];  % mm
smiData.Solid(3).MoI = [0.017672791093940439 3.6087418915243243 3.6087418915243243];  % kg*mm^2
smiData.Solid(3).PoI = [0 0 0];  % kg*mm^2
smiData.Solid(3).color = [0.792156862745098 0.81960784313725488 0.93333333333333335];
smiData.Solid(3).opacity = 1;
smiData.Solid(3).ID = "Pivot*:*Default";

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(4).mass = 0.03;  % kg
smiData.Solid(4).CoM = [26.596804910036894 0 0.27262927885218169];  % mm
smiData.Solid(4).MoI = [2.8596512843889403 7.4706713743702853 7.5625533508753557];  % kg*mm^2
smiData.Solid(4).PoI = [0 0.059290871539757477 0];  % kg*mm^2
smiData.Solid(4).color = [0.792156862745098 0.81960784313725488 0.93333333333333335];
smiData.Solid(4).opacity = 1;
smiData.Solid(4).ID = "Module*:*Default";

%Inertia Type - Custom
%Visual Properties - Simple
smiData.Solid(5).mass = 0.005;  % kg
smiData.Solid(5).CoM = [0 0 2.5];  % mm
smiData.Solid(5).MoI = [0.1121023960467226 0.1121023960467226 0.1952751415007426];  % kg*mm^2
smiData.Solid(5).PoI = [0 0 0];  % kg*mm^2
smiData.Solid(5).color = [0.65098039215686276 0.61960784313725492 0.58823529411764708];
smiData.Solid(5).opacity = 1;
smiData.Solid(5).ID = "Circle*:*Default";


%============= Joint =============%
%X Revolute Primitive (Rx) %Y Revolute Primitive (Ry) %Z Revolute Primitive (Rz)
%X Prismatic Primitive (Px) %Y Prismatic Primitive (Py) %Z Prismatic Primitive (Pz) %Spherical Primitive (S)
%Constant Velocity Primitive (CV) %Lead Screw Primitive (LS)
%Position Target (Pos)

%Initialize the RevoluteJoint structure array by filling in null values.
smiData.RevoluteJoint(2).Rz.Pos = 0.0;
smiData.RevoluteJoint(2).ID = "";

smiData.RevoluteJoint(1).Rz.Pos = -135.38038198290695;  % deg
smiData.RevoluteJoint(1).ID = "[Module-1:-:Pivot-1]";

smiData.RevoluteJoint(2).Rz.Pos = -177.19875123974819;  % deg
smiData.RevoluteJoint(2).ID = "[Base-1:-:Circle-1]";

