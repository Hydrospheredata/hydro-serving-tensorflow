¤╞
у╝
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
 
LSTMBlockCell
x"T
cs_prev"T
h_prev"T
w"T
wci"T
wcf"T
wco"T
b"T
i"T
cs"T
f"T
o"T
ci"T
co"T
h"T"
forget_biasfloat%  А?"
	cell_clipfloat%  @@"
use_peepholebool( "
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
А
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И"serve*1.7.02v1.7.0-3-g024aecf414Юн
v
PlaceholderPlaceholder*
dtype0*+
_output_shapes
:         * 
shape:         
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:         *
shape:         
J
ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
]
DropoutWrapperInit/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ф
JMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:
Ч
LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
Т
PMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
¤
KMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/concatConcatV2JMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/ConstLMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_1PMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Х
PMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
н
JMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/zerosFillKMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/concatPMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes
:	А
Ц
LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
Ч
LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_3Const*
valueB:А*
dtype0*
_output_shapes
:
Ц
LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
Ч
LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_5Const*
valueB:А*
dtype0*
_output_shapes
:
Ф
RMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Г
MMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1ConcatV2LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_4LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_5RMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
Ч
RMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
│
LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1FillMMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/concat_1RMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes
:	А
Ц
LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
Ч
LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/Const_7Const*
dtype0*
_output_shapes
:*
valueB:А
Л
$W/Initializer/truncated_normal/shapeConst*
valueB"      *
_class

loc:@W*
dtype0*
_output_shapes
:
~
#W/Initializer/truncated_normal/meanConst*
valueB
 *    *
_class

loc:@W*
dtype0*
_output_shapes
: 
А
%W/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *╨d╬=*
_class

loc:@W*
dtype0
═
.W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal$W/Initializer/truncated_normal/shape*
T0*
_class

loc:@W*
seed2 *
dtype0*
_output_shapes
:	А*

seed 
└
"W/Initializer/truncated_normal/mulMul.W/Initializer/truncated_normal/TruncatedNormal%W/Initializer/truncated_normal/stddev*
T0*
_class

loc:@W*
_output_shapes
:	А
о
W/Initializer/truncated_normalAdd"W/Initializer/truncated_normal/mul#W/Initializer/truncated_normal/mean*
_output_shapes
:	А*
T0*
_class

loc:@W
Н
W
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *
_class

loc:@W
Ю
W/AssignAssignWW/Initializer/truncated_normal*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*
_class

loc:@W
U
W/readIdentityW*
T0*
_class

loc:@W*
_output_shapes
:	А
v
b/Initializer/ConstConst*
_output_shapes
:*
valueB*    *
_class

loc:@b*
dtype0
Г
b
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class

loc:@b*
	container 
О
b/AssignAssignbb/Initializer/Const*
use_locking(*
T0*
_class

loc:@b*
validate_shape(*
_output_shapes
:
P
b/readIdentityb*
T0*
_class

loc:@b*
_output_shapes
:
I
Const_1Const*
_output_shapes
: *
value	B :*
dtype0
Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
s
splitSplitsplit/split_dimPlaceholder*
T0*+
_output_shapes
:         *
	num_split
f
RNN/SqueezeSqueezesplit*'
_output_shapes
:         *
squeeze_dims
*
T0
█
KRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"     *=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0*
_output_shapes
:
═
IRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *пыК╜*=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0*
_output_shapes
: 
═
IRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *пыК=*=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel
┴
SRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformKRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ША*

seed *
T0*=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel*
seed2 
╞
IRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/subSubIRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxIRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes
: 
┌
IRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulMulSRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformIRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:
ША
╠
ERNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniformAddIRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulIRNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
ША*
T0*=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel
с
*RNN/multi_rnn_cell/cell_0/lstm_cell/kernel
VariableV2*
shape:
ША*
dtype0* 
_output_shapes
:
ША*
shared_name *=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel*
	container 
┴
1RNN/multi_rnn_cell/cell_0/lstm_cell/kernel/AssignAssign*RNN/multi_rnn_cell/cell_0/lstm_cell/kernelERNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ША
╤
/RNN/multi_rnn_cell/cell_0/lstm_cell/kernel/readIdentity*RNN/multi_rnn_cell/cell_0/lstm_cell/kernel*
T0*=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:
ША
╞
:RNN/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/ConstConst*
valueBА*    *;
_class1
/-loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/bias*
dtype0*
_output_shapes	
:А
╙
(RNN/multi_rnn_cell/cell_0/lstm_cell/bias
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *;
_class1
/-loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/bias
л
/RNN/multi_rnn_cell/cell_0/lstm_cell/bias/AssignAssign(RNN/multi_rnn_cell/cell_0/lstm_cell/bias:RNN/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/Const*
use_locking(*
T0*;
_class1
/-loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
╞
-RNN/multi_rnn_cell/cell_0/lstm_cell/bias/readIdentity(RNN/multi_rnn_cell/cell_0/lstm_cell/bias*
T0*;
_class1
/-loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes	
:А
И
=RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/zeros/shape_as_tensorConst*
valueB:А*
dtype0*
_output_shapes
:
x
3RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
с
-RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/zerosFill=RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/zeros/shape_as_tensor3RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/zeros/Const*
T0*

index_type0*
_output_shapes	
:А
╞
5RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCellLSTMBlockCellRNN/SqueezeJMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/zerosLMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1/RNN/multi_rnn_cell/cell_0/lstm_cell/kernel/read-RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/zeros-RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/zeros-RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/zeros-RNN/multi_rnn_cell/cell_0/lstm_cell/bias/read*в
_output_shapesП
М:         А:         А:         А:         А:         А:         А:         А*
forget_bias%  А?*
use_peephole( *
	cell_clip%  А┐*
T0
й
MatMulMatMul7RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell:6W/read*
T0*'
_output_shapes
:         *
transpose_a( *
transpose_b( 
L
addAddMatMulb/read*
T0*'
_output_shapes
:         
К
9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientPlaceholder_1*
T0*'
_output_shapes
:         
k
)softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0*
_output_shapes
: 
m
*softmax_cross_entropy_with_logits_sg/ShapeShapeadd*
out_type0*
_output_shapes
:*
T0
m
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
o
,softmax_cross_entropy_with_logits_sg/Shape_1Shapeadd*
T0*
out_type0*
_output_shapes
:
l
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
й
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
_output_shapes
: *
T0
Ь
0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
y
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ў
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:
З
4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
r
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Е
+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
▓
,softmax_cross_entropy_with_logits_sg/ReshapeReshapeadd+softmax_cross_entropy_with_logits_sg/concat*0
_output_shapes
:                  *
T0*
Tshape0
m
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
е
,softmax_cross_entropy_with_logits_sg/Shape_2Shape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
n
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
н
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
_output_shapes
: *
T0
а
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*
_output_shapes
:*
T0*

axis *
N
{
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
№
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:
Й
6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
t
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Н
-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
ь
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:                  
э
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:         :                  
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
л
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
_output_shapes
: *
T0
|
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Я
1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:
Г
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*#
_output_shapes
:         *
Index0*
T0
╔
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*#
_output_shapes
:         *
T0*
Tshape0
Q
Const_2Const*
dtype0*
_output_shapes
:*
valueB: 
Г
MeanMean.softmax_cross_entropy_with_logits_sg/Reshape_2Const_2*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
╗
save/SaveV2/tensor_namesConst*o
valuefBdB(RNN/multi_rnn_cell/cell_0/lstm_cell/biasB*RNN/multi_rnn_cell/cell_0/lstm_cell/kernelBWBb*
dtype0*
_output_shapes
:
k
save/SaveV2/shape_and_slicesConst*
valueBB B B B *
dtype0*
_output_shapes
:
╔
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices(RNN/multi_rnn_cell/cell_0/lstm_cell/bias*RNN/multi_rnn_cell/cell_0/lstm_cell/kernelWb*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
═
save/RestoreV2/tensor_namesConst"/device:CPU:0*o
valuefBdB(RNN/multi_rnn_cell/cell_0/lstm_cell/biasB*RNN/multi_rnn_cell/cell_0/lstm_cell/kernelBWBb*
dtype0*
_output_shapes
:
}
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B *
dtype0*
_output_shapes
:
о
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*$
_output_shapes
::::*
dtypes
2
█
save/AssignAssign(RNN/multi_rnn_cell/cell_0/lstm_cell/biassave/RestoreV2*
use_locking(*
T0*;
_class1
/-loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ш
save/Assign_1Assign*RNN/multi_rnn_cell/cell_0/lstm_cell/kernelsave/RestoreV2:1*
use_locking(*
T0*=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ША
Х
save/Assign_2AssignWsave/RestoreV2:2*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0*
_class

loc:@W
Р
save/Assign_3Assignbsave/RestoreV2:3*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class

loc:@b
V
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3
И
initNoOp	^W/Assign	^b/Assign2^RNN/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign0^RNN/multi_rnn_cell/cell_0/lstm_cell/bias/Assign

init_all_tablesNoOp
(
legacy_init_opNoOp^init_all_tables
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ж
save_1/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_b56bbbb3dae64508805a7427a5d9b2e0/part*
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Ф
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
╠
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*o
valuefBdB(RNN/multi_rnn_cell/cell_0/lstm_cell/biasB*RNN/multi_rnn_cell/cell_0/lstm_cell/kernelBWBb*
dtype0
|
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueBB B B B 
ъ
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices(RNN/multi_rnn_cell/cell_0/lstm_cell/bias*RNN/multi_rnn_cell/cell_0/lstm_cell/kernelWb"/device:CPU:0*
dtypes
2
и
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
▓
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
Т
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
С
save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
╧
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*o
valuefBdB(RNN/multi_rnn_cell/cell_0/lstm_cell/biasB*RNN/multi_rnn_cell/cell_0/lstm_cell/kernelBWBb*
dtype0

!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B *
dtype0*
_output_shapes
:
╢
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*$
_output_shapes
::::*
dtypes
2
▀
save_1/AssignAssign(RNN/multi_rnn_cell/cell_0/lstm_cell/biassave_1/RestoreV2*
use_locking(*
T0*;
_class1
/-loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ь
save_1/Assign_1Assign*RNN/multi_rnn_cell/cell_0/lstm_cell/kernelsave_1/RestoreV2:1*
use_locking(*
T0*=
_class3
1/loc:@RNN/multi_rnn_cell/cell_0/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ША
Щ
save_1/Assign_2AssignWsave_1/RestoreV2:2*
use_locking(*
T0*
_class

loc:@W*
validate_shape(*
_output_shapes
:	А
Ф
save_1/Assign_3Assignbsave_1/RestoreV2:3*
use_locking(*
T0*
_class

loc:@b*
validate_shape(*
_output_shapes
:
b
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"│
h_zero_statesб
Ю
LMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros:0
NMultiRNNCellZeroState/DropoutWrapperZeroState/LSTMBlockCellZeroState/zeros_1:0"М
h_state_placeholderst
r
7RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell:1
7RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell:6*j
infera
0
data(
Placeholder:0         &
result
add:0         infer