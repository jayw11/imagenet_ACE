½/
ÿÉ
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
ì
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	

FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%·Ñ8"
data_formatstringNHWC"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype

ImageSummary
tag
tensor"T
summary"

max_imagesint(0"
Ttype0:
2"'
	bad_colortensorB:ÿ  ÿ
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
Ô
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

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
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
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
2	
D
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
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
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.8.0-dev201804082v1.7.0-1345-gb874783ccdÄ*

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 

global_step
VariableV2*
shared_name *
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: 
²
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
s
input_tensorPlaceholder*
shape:àà*
dtype0*)
_output_shapes
:àà
c
split_inputs/ConstConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
m
split_inputs/split/split_dimConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
®
split_inputs/splitSplitsplit_inputs/split/split_diminput_tensor"/device:CPU:0*
T0*<
_output_shapes*
(:@àà:@àà*
	num_split
`

images/tagConst"/device:GPU:0*
valueB Bimages*
dtype0*
_output_shapes
: 

imagesImageSummary
images/tagsplit_inputs/split"/device:GPU:0*

max_images*
T0*
	bad_colorB:ÿ  ÿ*
_output_shapes
: 

resnet_model/transpose/permConst"/device:GPU:0*%
valueB"             *
dtype0*
_output_shapes
:
£
resnet_model/transpose	Transposesplit_inputs/splitresnet_model/transpose/perm"/device:GPU:0*
T0*(
_output_shapes
:@àà*
Tperm0

resnet_model/Pad/paddingsConst"/device:GPU:0*9
value0B."                             *
dtype0*
_output_shapes

:

resnet_model/PadPadresnet_model/transposeresnet_model/Pad/paddings"/device:GPU:0*
T0*
	Tpaddings0*(
_output_shapes
:@ææ
Å
=resnet_model/conv2d/kernel/Initializer/truncated_normal/shapeConst*-
_class#
!loc:@resnet_model/conv2d/kernel*%
valueB"         @   *
dtype0*
_output_shapes
:
°
<resnet_model/conv2d/kernel/Initializer/truncated_normal/meanConst*-
_class#
!loc:@resnet_model/conv2d/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
²
>resnet_model/conv2d/kernel/Initializer/truncated_normal/stddevConst*-
_class#
!loc:@resnet_model/conv2d/kernel*
valueB
 *ê¨=*
dtype0*
_output_shapes
: 

Gresnet_model/conv2d/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal=resnet_model/conv2d/kernel/Initializer/truncated_normal/shape*
T0*-
_class#
!loc:@resnet_model/conv2d/kernel*
seed2 *
dtype0*&
_output_shapes
:@*

seed 
«
;resnet_model/conv2d/kernel/Initializer/truncated_normal/mulMulGresnet_model/conv2d/kernel/Initializer/truncated_normal/TruncatedNormal>resnet_model/conv2d/kernel/Initializer/truncated_normal/stddev*&
_output_shapes
:@*
T0*-
_class#
!loc:@resnet_model/conv2d/kernel

7resnet_model/conv2d/kernel/Initializer/truncated_normalAdd;resnet_model/conv2d/kernel/Initializer/truncated_normal/mul<resnet_model/conv2d/kernel/Initializer/truncated_normal/mean*
T0*-
_class#
!loc:@resnet_model/conv2d/kernel*&
_output_shapes
:@
Ü
resnet_model/conv2d/kernel
VariableV2"/device:CPU:0*
shared_name *-
_class#
!loc:@resnet_model/conv2d/kernel*
	container *
shape:@*
dtype0*&
_output_shapes
:@

!resnet_model/conv2d/kernel/AssignAssignresnet_model/conv2d/kernel7resnet_model/conv2d/kernel/Initializer/truncated_normal"/device:CPU:0*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*-
_class#
!loc:@resnet_model/conv2d/kernel
¶
resnet_model/conv2d/kernel/readIdentityresnet_model/conv2d/kernel"/device:CPU:0*
T0*-
_class#
!loc:@resnet_model/conv2d/kernel*&
_output_shapes
:@

!resnet_model/conv2d/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d/Conv2DConv2Dresnet_model/Padresnet_model/conv2d/kernel/read"/device:GPU:0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@@pp*
	dilations
*
T0

resnet_model/initial_convIdentityresnet_model/conv2d/Conv2D"/device:GPU:0*
T0*&
_output_shapes
:@@pp
Ú
"resnet_model/max_pooling2d/MaxPoolMaxPoolresnet_model/initial_conv"/device:GPU:0*
T0*
data_formatNCHW*
strides
*
ksize
*
paddingSAME*&
_output_shapes
:@@88

resnet_model/initial_max_poolIdentity"resnet_model/max_pooling2d/MaxPool"/device:GPU:0*&
_output_shapes
:@@88*
T0
¿
7resnet_model/batch_normalization/gamma/Initializer/onesConst*9
_class/
-+loc:@resnet_model/batch_normalization/gamma*
valueB@*  ?*
dtype0*
_output_shapes
:@
Ü
&resnet_model/batch_normalization/gamma
VariableV2"/device:CPU:0*
shared_name *9
_class/
-+loc:@resnet_model/batch_normalization/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
°
-resnet_model/batch_normalization/gamma/AssignAssign&resnet_model/batch_normalization/gamma7resnet_model/batch_normalization/gamma/Initializer/ones"/device:CPU:0*9
_class/
-+loc:@resnet_model/batch_normalization/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
Î
+resnet_model/batch_normalization/gamma/readIdentity&resnet_model/batch_normalization/gamma"/device:CPU:0*
T0*9
_class/
-+loc:@resnet_model/batch_normalization/gamma*
_output_shapes
:@
¾
7resnet_model/batch_normalization/beta/Initializer/zerosConst*8
_class.
,*loc:@resnet_model/batch_normalization/beta*
valueB@*    *
dtype0*
_output_shapes
:@
Ú
%resnet_model/batch_normalization/beta
VariableV2"/device:CPU:0*
shared_name *8
_class.
,*loc:@resnet_model/batch_normalization/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
­
,resnet_model/batch_normalization/beta/AssignAssign%resnet_model/batch_normalization/beta7resnet_model/batch_normalization/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*8
_class.
,*loc:@resnet_model/batch_normalization/beta*
validate_shape(*
_output_shapes
:@
Ë
*resnet_model/batch_normalization/beta/readIdentity%resnet_model/batch_normalization/beta"/device:CPU:0*8
_class.
,*loc:@resnet_model/batch_normalization/beta*
_output_shapes
:@*
T0
Ì
>resnet_model/batch_normalization/moving_mean/Initializer/zerosConst*?
_class5
31loc:@resnet_model/batch_normalization/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
è
,resnet_model/batch_normalization/moving_mean
VariableV2"/device:CPU:0*
shared_name *?
_class5
31loc:@resnet_model/batch_normalization/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@
É
3resnet_model/batch_normalization/moving_mean/AssignAssign,resnet_model/batch_normalization/moving_mean>resnet_model/batch_normalization/moving_mean/Initializer/zeros"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*?
_class5
31loc:@resnet_model/batch_normalization/moving_mean*
validate_shape(
à
1resnet_model/batch_normalization/moving_mean/readIdentity,resnet_model/batch_normalization/moving_mean"/device:CPU:0*
_output_shapes
:@*
T0*?
_class5
31loc:@resnet_model/batch_normalization/moving_mean
Ó
Aresnet_model/batch_normalization/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:@*C
_class9
75loc:@resnet_model/batch_normalization/moving_variance*
valueB@*  ?
ð
0resnet_model/batch_normalization/moving_variance
VariableV2"/device:CPU:0*C
_class9
75loc:@resnet_model/batch_normalization/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Ø
7resnet_model/batch_normalization/moving_variance/AssignAssign0resnet_model/batch_normalization/moving_varianceAresnet_model/batch_normalization/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*C
_class9
75loc:@resnet_model/batch_normalization/moving_variance*
validate_shape(*
_output_shapes
:@
ì
5resnet_model/batch_normalization/moving_variance/readIdentity0resnet_model/batch_normalization/moving_variance"/device:CPU:0*
T0*C
_class9
75loc:@resnet_model/batch_normalization/moving_variance*
_output_shapes
:@
·
/resnet_model/batch_normalization/FusedBatchNormFusedBatchNormresnet_model/initial_max_pool+resnet_model/batch_normalization/gamma/read*resnet_model/batch_normalization/beta/read1resnet_model/batch_normalization/moving_mean/read5resnet_model/batch_normalization/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( 
z
&resnet_model/batch_normalization/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/ReluRelu/resnet_model/batch_normalization/FusedBatchNorm"/device:GPU:0*
T0*&
_output_shapes
:@@88
É
?resnet_model/conv2d_1/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*/
_class%
#!loc:@resnet_model/conv2d_1/kernel*%
valueB"      @      *
dtype0
´
>resnet_model/conv2d_1/kernel/Initializer/truncated_normal/meanConst*/
_class%
#!loc:@resnet_model/conv2d_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¶
@resnet_model/conv2d_1/kernel/Initializer/truncated_normal/stddevConst*/
_class%
#!loc:@resnet_model/conv2d_1/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
¦
Iresnet_model/conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?resnet_model/conv2d_1/kernel/Initializer/truncated_normal/shape*
dtype0*'
_output_shapes
:@*

seed *
T0*/
_class%
#!loc:@resnet_model/conv2d_1/kernel*
seed2 
´
=resnet_model/conv2d_1/kernel/Initializer/truncated_normal/mulMulIresnet_model/conv2d_1/kernel/Initializer/truncated_normal/TruncatedNormal@resnet_model/conv2d_1/kernel/Initializer/truncated_normal/stddev*'
_output_shapes
:@*
T0*/
_class%
#!loc:@resnet_model/conv2d_1/kernel
¢
9resnet_model/conv2d_1/kernel/Initializer/truncated_normalAdd=resnet_model/conv2d_1/kernel/Initializer/truncated_normal/mul>resnet_model/conv2d_1/kernel/Initializer/truncated_normal/mean*
T0*/
_class%
#!loc:@resnet_model/conv2d_1/kernel*'
_output_shapes
:@
â
resnet_model/conv2d_1/kernel
VariableV2"/device:CPU:0*/
_class%
#!loc:@resnet_model/conv2d_1/kernel*
	container *
shape:@*
dtype0*'
_output_shapes
:@*
shared_name 
¡
#resnet_model/conv2d_1/kernel/AssignAssignresnet_model/conv2d_1/kernel9resnet_model/conv2d_1/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_1/kernel*
validate_shape(*'
_output_shapes
:@
½
!resnet_model/conv2d_1/kernel/readIdentityresnet_model/conv2d_1/kernel"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_1/kernel*'
_output_shapes
:@

#resnet_model/conv2d_1/dilation_rateConst"/device:GPU:0*
dtype0*
_output_shapes
:*
valueB"      

resnet_model/conv2d_1/Conv2DConv2Dresnet_model/Relu!resnet_model/conv2d_1/kernel/read"/device:GPU:0*
paddingSAME*'
_output_shapes
:@88*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
É
?resnet_model/conv2d_2/kernel/Initializer/truncated_normal/shapeConst*/
_class%
#!loc:@resnet_model/conv2d_2/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
´
>resnet_model/conv2d_2/kernel/Initializer/truncated_normal/meanConst*/
_class%
#!loc:@resnet_model/conv2d_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¶
@resnet_model/conv2d_2/kernel/Initializer/truncated_normal/stddevConst*/
_class%
#!loc:@resnet_model/conv2d_2/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
¥
Iresnet_model/conv2d_2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?resnet_model/conv2d_2/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0*&
_output_shapes
:@@*

seed *
T0*/
_class%
#!loc:@resnet_model/conv2d_2/kernel
³
=resnet_model/conv2d_2/kernel/Initializer/truncated_normal/mulMulIresnet_model/conv2d_2/kernel/Initializer/truncated_normal/TruncatedNormal@resnet_model/conv2d_2/kernel/Initializer/truncated_normal/stddev*
T0*/
_class%
#!loc:@resnet_model/conv2d_2/kernel*&
_output_shapes
:@@
¡
9resnet_model/conv2d_2/kernel/Initializer/truncated_normalAdd=resnet_model/conv2d_2/kernel/Initializer/truncated_normal/mul>resnet_model/conv2d_2/kernel/Initializer/truncated_normal/mean*/
_class%
#!loc:@resnet_model/conv2d_2/kernel*&
_output_shapes
:@@*
T0
à
resnet_model/conv2d_2/kernel
VariableV2"/device:CPU:0*
	container *
shape:@@*
dtype0*&
_output_shapes
:@@*
shared_name */
_class%
#!loc:@resnet_model/conv2d_2/kernel
 
#resnet_model/conv2d_2/kernel/AssignAssignresnet_model/conv2d_2/kernel9resnet_model/conv2d_2/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@
¼
!resnet_model/conv2d_2/kernel/readIdentityresnet_model/conv2d_2/kernel"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_2/kernel*&
_output_shapes
:@@

#resnet_model/conv2d_2/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_2/Conv2DConv2Dresnet_model/Relu!resnet_model/conv2d_2/kernel/read"/device:GPU:0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@88*
	dilations
*
T0
Ã
9resnet_model/batch_normalization_1/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:@*;
_class1
/-loc:@resnet_model/batch_normalization_1/gamma*
valueB@*  ?
à
(resnet_model/batch_normalization_1/gamma
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_1/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
¸
/resnet_model/batch_normalization_1/gamma/AssignAssign(resnet_model/batch_normalization_1/gamma9resnet_model/batch_normalization_1/gamma/Initializer/ones"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_1/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
Ô
-resnet_model/batch_normalization_1/gamma/readIdentity(resnet_model/batch_normalization_1/gamma"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_1/gamma*
_output_shapes
:@
Â
9resnet_model/batch_normalization_1/beta/Initializer/zerosConst*:
_class0
.,loc:@resnet_model/batch_normalization_1/beta*
valueB@*    *
dtype0*
_output_shapes
:@
Þ
'resnet_model/batch_normalization_1/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@resnet_model/batch_normalization_1/beta*
	container *
shape:@
µ
.resnet_model/batch_normalization_1/beta/AssignAssign'resnet_model/batch_normalization_1/beta9resnet_model/batch_normalization_1/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_1/beta*
validate_shape(*
_output_shapes
:@
Ñ
,resnet_model/batch_normalization_1/beta/readIdentity'resnet_model/batch_normalization_1/beta"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_1/beta*
_output_shapes
:@
Ð
@resnet_model/batch_normalization_1/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*A
_class7
53loc:@resnet_model/batch_normalization_1/moving_mean*
valueB@*    
ì
.resnet_model/batch_normalization_1/moving_mean
VariableV2"/device:CPU:0*A
_class7
53loc:@resnet_model/batch_normalization_1/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Ñ
5resnet_model/batch_normalization_1/moving_mean/AssignAssign.resnet_model/batch_normalization_1/moving_mean@resnet_model/batch_normalization_1/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
:@
æ
3resnet_model/batch_normalization_1/moving_mean/readIdentity.resnet_model/batch_normalization_1/moving_mean"/device:CPU:0*
T0*A
_class7
53loc:@resnet_model/batch_normalization_1/moving_mean*
_output_shapes
:@
×
Cresnet_model/batch_normalization_1/moving_variance/Initializer/onesConst*
_output_shapes
:@*E
_class;
97loc:@resnet_model/batch_normalization_1/moving_variance*
valueB@*  ?*
dtype0
ô
2resnet_model/batch_normalization_1/moving_variance
VariableV2"/device:CPU:0*
shared_name *E
_class;
97loc:@resnet_model/batch_normalization_1/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@
à
9resnet_model/batch_normalization_1/moving_variance/AssignAssign2resnet_model/batch_normalization_1/moving_varianceCresnet_model/batch_normalization_1/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
:@
ò
7resnet_model/batch_normalization_1/moving_variance/readIdentity2resnet_model/batch_normalization_1/moving_variance"/device:CPU:0*
T0*E
_class;
97loc:@resnet_model/batch_normalization_1/moving_variance*
_output_shapes
:@
À
1resnet_model/batch_normalization_1/FusedBatchNormFusedBatchNormresnet_model/conv2d_2/Conv2D-resnet_model/batch_normalization_1/gamma/read,resnet_model/batch_normalization_1/beta/read3resnet_model/batch_normalization_1/moving_mean/read7resnet_model/batch_normalization_1/moving_variance/read"/device:GPU:0*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW
|
(resnet_model/batch_normalization_1/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_1Relu1resnet_model/batch_normalization_1/FusedBatchNorm"/device:GPU:0*
T0*&
_output_shapes
:@@88
É
?resnet_model/conv2d_3/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*/
_class%
#!loc:@resnet_model/conv2d_3/kernel*%
valueB"      @   @   *
dtype0
´
>resnet_model/conv2d_3/kernel/Initializer/truncated_normal/meanConst*/
_class%
#!loc:@resnet_model/conv2d_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¶
@resnet_model/conv2d_3/kernel/Initializer/truncated_normal/stddevConst*/
_class%
#!loc:@resnet_model/conv2d_3/kernel*
valueB
 *«ª*=*
dtype0*
_output_shapes
: 
¥
Iresnet_model/conv2d_3/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?resnet_model/conv2d_3/kernel/Initializer/truncated_normal/shape*&
_output_shapes
:@@*

seed *
T0*/
_class%
#!loc:@resnet_model/conv2d_3/kernel*
seed2 *
dtype0
³
=resnet_model/conv2d_3/kernel/Initializer/truncated_normal/mulMulIresnet_model/conv2d_3/kernel/Initializer/truncated_normal/TruncatedNormal@resnet_model/conv2d_3/kernel/Initializer/truncated_normal/stddev*
T0*/
_class%
#!loc:@resnet_model/conv2d_3/kernel*&
_output_shapes
:@@
¡
9resnet_model/conv2d_3/kernel/Initializer/truncated_normalAdd=resnet_model/conv2d_3/kernel/Initializer/truncated_normal/mul>resnet_model/conv2d_3/kernel/Initializer/truncated_normal/mean*
T0*/
_class%
#!loc:@resnet_model/conv2d_3/kernel*&
_output_shapes
:@@
à
resnet_model/conv2d_3/kernel
VariableV2"/device:CPU:0*
dtype0*&
_output_shapes
:@@*
shared_name */
_class%
#!loc:@resnet_model/conv2d_3/kernel*
	container *
shape:@@
 
#resnet_model/conv2d_3/kernel/AssignAssignresnet_model/conv2d_3/kernel9resnet_model/conv2d_3/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_3/kernel*
validate_shape(*&
_output_shapes
:@@
¼
!resnet_model/conv2d_3/kernel/readIdentityresnet_model/conv2d_3/kernel"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_3/kernel*&
_output_shapes
:@@

#resnet_model/conv2d_3/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_3/Conv2DConv2Dresnet_model/Relu_1!resnet_model/conv2d_3/kernel/read"/device:GPU:0*&
_output_shapes
:@@88*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ã
9resnet_model/batch_normalization_2/gamma/Initializer/onesConst*;
_class1
/-loc:@resnet_model/batch_normalization_2/gamma*
valueB@*  ?*
dtype0*
_output_shapes
:@
à
(resnet_model/batch_normalization_2/gamma
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_2/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
¸
/resnet_model/batch_normalization_2/gamma/AssignAssign(resnet_model/batch_normalization_2/gamma9resnet_model/batch_normalization_2/gamma/Initializer/ones"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_2/gamma*
validate_shape(
Ô
-resnet_model/batch_normalization_2/gamma/readIdentity(resnet_model/batch_normalization_2/gamma"/device:CPU:0*
_output_shapes
:@*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_2/gamma
Â
9resnet_model/batch_normalization_2/beta/Initializer/zerosConst*:
_class0
.,loc:@resnet_model/batch_normalization_2/beta*
valueB@*    *
dtype0*
_output_shapes
:@
Þ
'resnet_model/batch_normalization_2/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@resnet_model/batch_normalization_2/beta*
	container *
shape:@
µ
.resnet_model/batch_normalization_2/beta/AssignAssign'resnet_model/batch_normalization_2/beta9resnet_model/batch_normalization_2/beta/Initializer/zeros"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_2/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
Ñ
,resnet_model/batch_normalization_2/beta/readIdentity'resnet_model/batch_normalization_2/beta"/device:CPU:0*
_output_shapes
:@*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_2/beta
Ð
@resnet_model/batch_normalization_2/moving_mean/Initializer/zerosConst*A
_class7
53loc:@resnet_model/batch_normalization_2/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
ì
.resnet_model/batch_normalization_2/moving_mean
VariableV2"/device:CPU:0*
shared_name *A
_class7
53loc:@resnet_model/batch_normalization_2/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@
Ñ
5resnet_model/batch_normalization_2/moving_mean/AssignAssign.resnet_model/batch_normalization_2/moving_mean@resnet_model/batch_normalization_2/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes
:@
æ
3resnet_model/batch_normalization_2/moving_mean/readIdentity.resnet_model/batch_normalization_2/moving_mean"/device:CPU:0*
T0*A
_class7
53loc:@resnet_model/batch_normalization_2/moving_mean*
_output_shapes
:@
×
Cresnet_model/batch_normalization_2/moving_variance/Initializer/onesConst*E
_class;
97loc:@resnet_model/batch_normalization_2/moving_variance*
valueB@*  ?*
dtype0*
_output_shapes
:@
ô
2resnet_model/batch_normalization_2/moving_variance
VariableV2"/device:CPU:0*
shared_name *E
_class;
97loc:@resnet_model/batch_normalization_2/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@
à
9resnet_model/batch_normalization_2/moving_variance/AssignAssign2resnet_model/batch_normalization_2/moving_varianceCresnet_model/batch_normalization_2/moving_variance/Initializer/ones"/device:CPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_2/moving_variance
ò
7resnet_model/batch_normalization_2/moving_variance/readIdentity2resnet_model/batch_normalization_2/moving_variance"/device:CPU:0*
T0*E
_class;
97loc:@resnet_model/batch_normalization_2/moving_variance*
_output_shapes
:@
À
1resnet_model/batch_normalization_2/FusedBatchNormFusedBatchNormresnet_model/conv2d_3/Conv2D-resnet_model/batch_normalization_2/gamma/read,resnet_model/batch_normalization_2/beta/read3resnet_model/batch_normalization_2/moving_mean/read7resnet_model/batch_normalization_2/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( *
epsilon%ð'7
|
(resnet_model/batch_normalization_2/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_2Relu1resnet_model/batch_normalization_2/FusedBatchNorm"/device:GPU:0*
T0*&
_output_shapes
:@@88
É
?resnet_model/conv2d_4/kernel/Initializer/truncated_normal/shapeConst*/
_class%
#!loc:@resnet_model/conv2d_4/kernel*%
valueB"      @      *
dtype0*
_output_shapes
:
´
>resnet_model/conv2d_4/kernel/Initializer/truncated_normal/meanConst*/
_class%
#!loc:@resnet_model/conv2d_4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¶
@resnet_model/conv2d_4/kernel/Initializer/truncated_normal/stddevConst*/
_class%
#!loc:@resnet_model/conv2d_4/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
¦
Iresnet_model/conv2d_4/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?resnet_model/conv2d_4/kernel/Initializer/truncated_normal/shape*
dtype0*'
_output_shapes
:@*

seed *
T0*/
_class%
#!loc:@resnet_model/conv2d_4/kernel*
seed2 
´
=resnet_model/conv2d_4/kernel/Initializer/truncated_normal/mulMulIresnet_model/conv2d_4/kernel/Initializer/truncated_normal/TruncatedNormal@resnet_model/conv2d_4/kernel/Initializer/truncated_normal/stddev*
T0*/
_class%
#!loc:@resnet_model/conv2d_4/kernel*'
_output_shapes
:@
¢
9resnet_model/conv2d_4/kernel/Initializer/truncated_normalAdd=resnet_model/conv2d_4/kernel/Initializer/truncated_normal/mul>resnet_model/conv2d_4/kernel/Initializer/truncated_normal/mean*'
_output_shapes
:@*
T0*/
_class%
#!loc:@resnet_model/conv2d_4/kernel
â
resnet_model/conv2d_4/kernel
VariableV2"/device:CPU:0*
shared_name */
_class%
#!loc:@resnet_model/conv2d_4/kernel*
	container *
shape:@*
dtype0*'
_output_shapes
:@
¡
#resnet_model/conv2d_4/kernel/AssignAssignresnet_model/conv2d_4/kernel9resnet_model/conv2d_4/kernel/Initializer/truncated_normal"/device:CPU:0*/
_class%
#!loc:@resnet_model/conv2d_4/kernel*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0
½
!resnet_model/conv2d_4/kernel/readIdentityresnet_model/conv2d_4/kernel"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_4/kernel*'
_output_shapes
:@

#resnet_model/conv2d_4/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_4/Conv2DConv2Dresnet_model/Relu_2!resnet_model/conv2d_4/kernel/read"/device:GPU:0*'
_output_shapes
:@88*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME

resnet_model/addAddresnet_model/conv2d_4/Conv2Dresnet_model/conv2d_1/Conv2D"/device:GPU:0*'
_output_shapes
:@88*
T0
Å
9resnet_model/batch_normalization_3/gamma/Initializer/onesConst*;
_class1
/-loc:@resnet_model/batch_normalization_3/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_3/gamma
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_3/gamma
¹
/resnet_model/batch_normalization_3/gamma/AssignAssign(resnet_model/batch_normalization_3/gamma9resnet_model/batch_normalization_3/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_3/gamma/readIdentity(resnet_model/batch_normalization_3/gamma"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_3/gamma*
_output_shapes	
:
Ä
9resnet_model/batch_normalization_3/beta/Initializer/zerosConst*:
_class0
.,loc:@resnet_model/batch_normalization_3/beta*
valueB*    *
dtype0*
_output_shapes	
:
à
'resnet_model/batch_normalization_3/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *:
_class0
.,loc:@resnet_model/batch_normalization_3/beta*
	container *
shape:
¶
.resnet_model/batch_normalization_3/beta/AssignAssign'resnet_model/batch_normalization_3/beta9resnet_model/batch_normalization_3/beta/Initializer/zeros"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
Ò
,resnet_model/batch_normalization_3/beta/readIdentity'resnet_model/batch_normalization_3/beta"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_3/beta*
_output_shapes	
:
Ò
@resnet_model/batch_normalization_3/moving_mean/Initializer/zerosConst*A
_class7
53loc:@resnet_model/batch_normalization_3/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
î
.resnet_model/batch_normalization_3/moving_mean
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@resnet_model/batch_normalization_3/moving_mean*
	container 
Ò
5resnet_model/batch_normalization_3/moving_mean/AssignAssign.resnet_model/batch_normalization_3/moving_mean@resnet_model/batch_normalization_3/moving_mean/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_3/moving_mean
ç
3resnet_model/batch_normalization_3/moving_mean/readIdentity.resnet_model/batch_normalization_3/moving_mean"/device:CPU:0*
T0*A
_class7
53loc:@resnet_model/batch_normalization_3/moving_mean*
_output_shapes	
:
Ù
Cresnet_model/batch_normalization_3/moving_variance/Initializer/onesConst*E
_class;
97loc:@resnet_model/batch_normalization_3/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ö
2resnet_model/batch_normalization_3/moving_variance
VariableV2"/device:CPU:0*
shared_name *E
_class;
97loc:@resnet_model/batch_normalization_3/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
á
9resnet_model/batch_normalization_3/moving_variance/AssignAssign2resnet_model/batch_normalization_3/moving_varianceCresnet_model/batch_normalization_3/moving_variance/Initializer/ones"/device:CPU:0*
T0*E
_class;
97loc:@resnet_model/batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ó
7resnet_model/batch_normalization_3/moving_variance/readIdentity2resnet_model/batch_normalization_3/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*E
_class;
97loc:@resnet_model/batch_normalization_3/moving_variance
¹
1resnet_model/batch_normalization_3/FusedBatchNormFusedBatchNormresnet_model/add-resnet_model/batch_normalization_3/gamma/read,resnet_model/batch_normalization_3/beta/read3resnet_model/batch_normalization_3/moving_mean/read7resnet_model/batch_normalization_3/moving_variance/read"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:@88::::*
is_training( *
epsilon%ð'7*
T0
|
(resnet_model/batch_normalization_3/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_3Relu1resnet_model/batch_normalization_3/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@88
É
?resnet_model/conv2d_5/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*/
_class%
#!loc:@resnet_model/conv2d_5/kernel*%
valueB"         @   
´
>resnet_model/conv2d_5/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: */
_class%
#!loc:@resnet_model/conv2d_5/kernel*
valueB
 *    *
dtype0
¶
@resnet_model/conv2d_5/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: */
_class%
#!loc:@resnet_model/conv2d_5/kernel*
valueB
 *  =
¦
Iresnet_model/conv2d_5/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?resnet_model/conv2d_5/kernel/Initializer/truncated_normal/shape*
T0*/
_class%
#!loc:@resnet_model/conv2d_5/kernel*
seed2 *
dtype0*'
_output_shapes
:@*

seed 
´
=resnet_model/conv2d_5/kernel/Initializer/truncated_normal/mulMulIresnet_model/conv2d_5/kernel/Initializer/truncated_normal/TruncatedNormal@resnet_model/conv2d_5/kernel/Initializer/truncated_normal/stddev*
T0*/
_class%
#!loc:@resnet_model/conv2d_5/kernel*'
_output_shapes
:@
¢
9resnet_model/conv2d_5/kernel/Initializer/truncated_normalAdd=resnet_model/conv2d_5/kernel/Initializer/truncated_normal/mul>resnet_model/conv2d_5/kernel/Initializer/truncated_normal/mean*'
_output_shapes
:@*
T0*/
_class%
#!loc:@resnet_model/conv2d_5/kernel
â
resnet_model/conv2d_5/kernel
VariableV2"/device:CPU:0*
shape:@*
dtype0*'
_output_shapes
:@*
shared_name */
_class%
#!loc:@resnet_model/conv2d_5/kernel*
	container 
¡
#resnet_model/conv2d_5/kernel/AssignAssignresnet_model/conv2d_5/kernel9resnet_model/conv2d_5/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_5/kernel*
validate_shape(*'
_output_shapes
:@
½
!resnet_model/conv2d_5/kernel/readIdentityresnet_model/conv2d_5/kernel"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_5/kernel*'
_output_shapes
:@

#resnet_model/conv2d_5/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_5/Conv2DConv2Dresnet_model/Relu_3!resnet_model/conv2d_5/kernel/read"/device:GPU:0*&
_output_shapes
:@@88*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ã
9resnet_model/batch_normalization_4/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:@*;
_class1
/-loc:@resnet_model/batch_normalization_4/gamma*
valueB@*  ?
à
(resnet_model/batch_normalization_4/gamma
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_4/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
¸
/resnet_model/batch_normalization_4/gamma/AssignAssign(resnet_model/batch_normalization_4/gamma9resnet_model/batch_normalization_4/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_4/gamma*
validate_shape(*
_output_shapes
:@
Ô
-resnet_model/batch_normalization_4/gamma/readIdentity(resnet_model/batch_normalization_4/gamma"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_4/gamma*
_output_shapes
:@
Â
9resnet_model/batch_normalization_4/beta/Initializer/zerosConst*:
_class0
.,loc:@resnet_model/batch_normalization_4/beta*
valueB@*    *
dtype0*
_output_shapes
:@
Þ
'resnet_model/batch_normalization_4/beta
VariableV2"/device:CPU:0*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@resnet_model/batch_normalization_4/beta*
	container 
µ
.resnet_model/batch_normalization_4/beta/AssignAssign'resnet_model/batch_normalization_4/beta9resnet_model/batch_normalization_4/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_4/beta*
validate_shape(*
_output_shapes
:@
Ñ
,resnet_model/batch_normalization_4/beta/readIdentity'resnet_model/batch_normalization_4/beta"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_4/beta*
_output_shapes
:@
Ð
@resnet_model/batch_normalization_4/moving_mean/Initializer/zerosConst*A
_class7
53loc:@resnet_model/batch_normalization_4/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
ì
.resnet_model/batch_normalization_4/moving_mean
VariableV2"/device:CPU:0*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *A
_class7
53loc:@resnet_model/batch_normalization_4/moving_mean*
	container 
Ñ
5resnet_model/batch_normalization_4/moving_mean/AssignAssign.resnet_model/batch_normalization_4/moving_mean@resnet_model/batch_normalization_4/moving_mean/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_4/moving_mean
æ
3resnet_model/batch_normalization_4/moving_mean/readIdentity.resnet_model/batch_normalization_4/moving_mean"/device:CPU:0*
T0*A
_class7
53loc:@resnet_model/batch_normalization_4/moving_mean*
_output_shapes
:@
×
Cresnet_model/batch_normalization_4/moving_variance/Initializer/onesConst*
_output_shapes
:@*E
_class;
97loc:@resnet_model/batch_normalization_4/moving_variance*
valueB@*  ?*
dtype0
ô
2resnet_model/batch_normalization_4/moving_variance
VariableV2"/device:CPU:0*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *E
_class;
97loc:@resnet_model/batch_normalization_4/moving_variance*
	container 
à
9resnet_model/batch_normalization_4/moving_variance/AssignAssign2resnet_model/batch_normalization_4/moving_varianceCresnet_model/batch_normalization_4/moving_variance/Initializer/ones"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_4/moving_variance*
validate_shape(
ò
7resnet_model/batch_normalization_4/moving_variance/readIdentity2resnet_model/batch_normalization_4/moving_variance"/device:CPU:0*
T0*E
_class;
97loc:@resnet_model/batch_normalization_4/moving_variance*
_output_shapes
:@
À
1resnet_model/batch_normalization_4/FusedBatchNormFusedBatchNormresnet_model/conv2d_5/Conv2D-resnet_model/batch_normalization_4/gamma/read,resnet_model/batch_normalization_4/beta/read3resnet_model/batch_normalization_4/moving_mean/read7resnet_model/batch_normalization_4/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( *
epsilon%ð'7
|
(resnet_model/batch_normalization_4/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_4Relu1resnet_model/batch_normalization_4/FusedBatchNorm"/device:GPU:0*&
_output_shapes
:@@88*
T0
É
?resnet_model/conv2d_6/kernel/Initializer/truncated_normal/shapeConst*/
_class%
#!loc:@resnet_model/conv2d_6/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:
´
>resnet_model/conv2d_6/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: */
_class%
#!loc:@resnet_model/conv2d_6/kernel*
valueB
 *    *
dtype0
¶
@resnet_model/conv2d_6/kernel/Initializer/truncated_normal/stddevConst*/
_class%
#!loc:@resnet_model/conv2d_6/kernel*
valueB
 *«ª*=*
dtype0*
_output_shapes
: 
¥
Iresnet_model/conv2d_6/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?resnet_model/conv2d_6/kernel/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*/
_class%
#!loc:@resnet_model/conv2d_6/kernel*
seed2 
³
=resnet_model/conv2d_6/kernel/Initializer/truncated_normal/mulMulIresnet_model/conv2d_6/kernel/Initializer/truncated_normal/TruncatedNormal@resnet_model/conv2d_6/kernel/Initializer/truncated_normal/stddev*
T0*/
_class%
#!loc:@resnet_model/conv2d_6/kernel*&
_output_shapes
:@@
¡
9resnet_model/conv2d_6/kernel/Initializer/truncated_normalAdd=resnet_model/conv2d_6/kernel/Initializer/truncated_normal/mul>resnet_model/conv2d_6/kernel/Initializer/truncated_normal/mean*
T0*/
_class%
#!loc:@resnet_model/conv2d_6/kernel*&
_output_shapes
:@@
à
resnet_model/conv2d_6/kernel
VariableV2"/device:CPU:0*
dtype0*&
_output_shapes
:@@*
shared_name */
_class%
#!loc:@resnet_model/conv2d_6/kernel*
	container *
shape:@@
 
#resnet_model/conv2d_6/kernel/AssignAssignresnet_model/conv2d_6/kernel9resnet_model/conv2d_6/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_6/kernel*
validate_shape(*&
_output_shapes
:@@
¼
!resnet_model/conv2d_6/kernel/readIdentityresnet_model/conv2d_6/kernel"/device:CPU:0*&
_output_shapes
:@@*
T0*/
_class%
#!loc:@resnet_model/conv2d_6/kernel

#resnet_model/conv2d_6/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_6/Conv2DConv2Dresnet_model/Relu_4!resnet_model/conv2d_6/kernel/read"/device:GPU:0*&
_output_shapes
:@@88*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ã
9resnet_model/batch_normalization_5/gamma/Initializer/onesConst*;
_class1
/-loc:@resnet_model/batch_normalization_5/gamma*
valueB@*  ?*
dtype0*
_output_shapes
:@
à
(resnet_model/batch_normalization_5/gamma
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_5/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
¸
/resnet_model/batch_normalization_5/gamma/AssignAssign(resnet_model/batch_normalization_5/gamma9resnet_model/batch_normalization_5/gamma/Initializer/ones"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_5/gamma*
validate_shape(
Ô
-resnet_model/batch_normalization_5/gamma/readIdentity(resnet_model/batch_normalization_5/gamma"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_5/gamma*
_output_shapes
:@*
T0
Â
9resnet_model/batch_normalization_5/beta/Initializer/zerosConst*
_output_shapes
:@*:
_class0
.,loc:@resnet_model/batch_normalization_5/beta*
valueB@*    *
dtype0
Þ
'resnet_model/batch_normalization_5/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@resnet_model/batch_normalization_5/beta*
	container *
shape:@
µ
.resnet_model/batch_normalization_5/beta/AssignAssign'resnet_model/batch_normalization_5/beta9resnet_model/batch_normalization_5/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_5/beta*
validate_shape(*
_output_shapes
:@
Ñ
,resnet_model/batch_normalization_5/beta/readIdentity'resnet_model/batch_normalization_5/beta"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_5/beta*
_output_shapes
:@
Ð
@resnet_model/batch_normalization_5/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*A
_class7
53loc:@resnet_model/batch_normalization_5/moving_mean*
valueB@*    
ì
.resnet_model/batch_normalization_5/moving_mean
VariableV2"/device:CPU:0*
shared_name *A
_class7
53loc:@resnet_model/batch_normalization_5/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@
Ñ
5resnet_model/batch_normalization_5/moving_mean/AssignAssign.resnet_model/batch_normalization_5/moving_mean@resnet_model/batch_normalization_5/moving_mean/Initializer/zeros"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_5/moving_mean*
validate_shape(
æ
3resnet_model/batch_normalization_5/moving_mean/readIdentity.resnet_model/batch_normalization_5/moving_mean"/device:CPU:0*
T0*A
_class7
53loc:@resnet_model/batch_normalization_5/moving_mean*
_output_shapes
:@
×
Cresnet_model/batch_normalization_5/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:@*E
_class;
97loc:@resnet_model/batch_normalization_5/moving_variance*
valueB@*  ?
ô
2resnet_model/batch_normalization_5/moving_variance
VariableV2"/device:CPU:0*E
_class;
97loc:@resnet_model/batch_normalization_5/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
à
9resnet_model/batch_normalization_5/moving_variance/AssignAssign2resnet_model/batch_normalization_5/moving_varianceCresnet_model/batch_normalization_5/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_5/moving_variance*
validate_shape(*
_output_shapes
:@
ò
7resnet_model/batch_normalization_5/moving_variance/readIdentity2resnet_model/batch_normalization_5/moving_variance"/device:CPU:0*
_output_shapes
:@*
T0*E
_class;
97loc:@resnet_model/batch_normalization_5/moving_variance
À
1resnet_model/batch_normalization_5/FusedBatchNormFusedBatchNormresnet_model/conv2d_6/Conv2D-resnet_model/batch_normalization_5/gamma/read,resnet_model/batch_normalization_5/beta/read3resnet_model/batch_normalization_5/moving_mean/read7resnet_model/batch_normalization_5/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( 
|
(resnet_model/batch_normalization_5/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_5Relu1resnet_model/batch_normalization_5/FusedBatchNorm"/device:GPU:0*
T0*&
_output_shapes
:@@88
É
?resnet_model/conv2d_7/kernel/Initializer/truncated_normal/shapeConst*/
_class%
#!loc:@resnet_model/conv2d_7/kernel*%
valueB"      @      *
dtype0*
_output_shapes
:
´
>resnet_model/conv2d_7/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: */
_class%
#!loc:@resnet_model/conv2d_7/kernel*
valueB
 *    
¶
@resnet_model/conv2d_7/kernel/Initializer/truncated_normal/stddevConst*/
_class%
#!loc:@resnet_model/conv2d_7/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
¦
Iresnet_model/conv2d_7/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?resnet_model/conv2d_7/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0*'
_output_shapes
:@*

seed *
T0*/
_class%
#!loc:@resnet_model/conv2d_7/kernel
´
=resnet_model/conv2d_7/kernel/Initializer/truncated_normal/mulMulIresnet_model/conv2d_7/kernel/Initializer/truncated_normal/TruncatedNormal@resnet_model/conv2d_7/kernel/Initializer/truncated_normal/stddev*'
_output_shapes
:@*
T0*/
_class%
#!loc:@resnet_model/conv2d_7/kernel
¢
9resnet_model/conv2d_7/kernel/Initializer/truncated_normalAdd=resnet_model/conv2d_7/kernel/Initializer/truncated_normal/mul>resnet_model/conv2d_7/kernel/Initializer/truncated_normal/mean*
T0*/
_class%
#!loc:@resnet_model/conv2d_7/kernel*'
_output_shapes
:@
â
resnet_model/conv2d_7/kernel
VariableV2"/device:CPU:0*
shared_name */
_class%
#!loc:@resnet_model/conv2d_7/kernel*
	container *
shape:@*
dtype0*'
_output_shapes
:@
¡
#resnet_model/conv2d_7/kernel/AssignAssignresnet_model/conv2d_7/kernel9resnet_model/conv2d_7/kernel/Initializer/truncated_normal"/device:CPU:0*/
_class%
#!loc:@resnet_model/conv2d_7/kernel*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0
½
!resnet_model/conv2d_7/kernel/readIdentityresnet_model/conv2d_7/kernel"/device:CPU:0*'
_output_shapes
:@*
T0*/
_class%
#!loc:@resnet_model/conv2d_7/kernel

#resnet_model/conv2d_7/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_7/Conv2DConv2Dresnet_model/Relu_5!resnet_model/conv2d_7/kernel/read"/device:GPU:0*
paddingSAME*'
_output_shapes
:@88*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(

resnet_model/add_1Addresnet_model/conv2d_7/Conv2Dresnet_model/add"/device:GPU:0*
T0*'
_output_shapes
:@88
Å
9resnet_model/batch_normalization_6/gamma/Initializer/onesConst*
_output_shapes	
:*;
_class1
/-loc:@resnet_model/batch_normalization_6/gamma*
valueB*  ?*
dtype0
â
(resnet_model/batch_normalization_6/gamma
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_6/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
¹
/resnet_model/batch_normalization_6/gamma/AssignAssign(resnet_model/batch_normalization_6/gamma9resnet_model/batch_normalization_6/gamma/Initializer/ones"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_6/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Õ
-resnet_model/batch_normalization_6/gamma/readIdentity(resnet_model/batch_normalization_6/gamma"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_6/gamma*
_output_shapes	
:
Ä
9resnet_model/batch_normalization_6/beta/Initializer/zerosConst*:
_class0
.,loc:@resnet_model/batch_normalization_6/beta*
valueB*    *
dtype0*
_output_shapes	
:
à
'resnet_model/batch_normalization_6/beta
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *:
_class0
.,loc:@resnet_model/batch_normalization_6/beta*
	container 
¶
.resnet_model/batch_normalization_6/beta/AssignAssign'resnet_model/batch_normalization_6/beta9resnet_model/batch_normalization_6/beta/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_6/beta
Ò
,resnet_model/batch_normalization_6/beta/readIdentity'resnet_model/batch_normalization_6/beta"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_6/beta*
_output_shapes	
:
Ò
@resnet_model/batch_normalization_6/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*A
_class7
53loc:@resnet_model/batch_normalization_6/moving_mean*
valueB*    
î
.resnet_model/batch_normalization_6/moving_mean
VariableV2"/device:CPU:0*
shared_name *A
_class7
53loc:@resnet_model/batch_normalization_6/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ò
5resnet_model/batch_normalization_6/moving_mean/AssignAssign.resnet_model/batch_normalization_6/moving_mean@resnet_model/batch_normalization_6/moving_mean/Initializer/zeros"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_6/moving_mean*
validate_shape(
ç
3resnet_model/batch_normalization_6/moving_mean/readIdentity.resnet_model/batch_normalization_6/moving_mean"/device:CPU:0*A
_class7
53loc:@resnet_model/batch_normalization_6/moving_mean*
_output_shapes	
:*
T0
Ù
Cresnet_model/batch_normalization_6/moving_variance/Initializer/onesConst*E
_class;
97loc:@resnet_model/batch_normalization_6/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ö
2resnet_model/batch_normalization_6/moving_variance
VariableV2"/device:CPU:0*
shared_name *E
_class;
97loc:@resnet_model/batch_normalization_6/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
á
9resnet_model/batch_normalization_6/moving_variance/AssignAssign2resnet_model/batch_normalization_6/moving_varianceCresnet_model/batch_normalization_6/moving_variance/Initializer/ones"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_6/moving_variance*
validate_shape(
ó
7resnet_model/batch_normalization_6/moving_variance/readIdentity2resnet_model/batch_normalization_6/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*E
_class;
97loc:@resnet_model/batch_normalization_6/moving_variance
»
1resnet_model/batch_normalization_6/FusedBatchNormFusedBatchNormresnet_model/add_1-resnet_model/batch_normalization_6/gamma/read,resnet_model/batch_normalization_6/beta/read3resnet_model/batch_normalization_6/moving_mean/read7resnet_model/batch_normalization_6/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:@88::::*
is_training( *
epsilon%ð'7
|
(resnet_model/batch_normalization_6/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_6Relu1resnet_model/batch_normalization_6/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@88
É
?resnet_model/conv2d_8/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*/
_class%
#!loc:@resnet_model/conv2d_8/kernel*%
valueB"         @   
´
>resnet_model/conv2d_8/kernel/Initializer/truncated_normal/meanConst*/
_class%
#!loc:@resnet_model/conv2d_8/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¶
@resnet_model/conv2d_8/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: */
_class%
#!loc:@resnet_model/conv2d_8/kernel*
valueB
 *  =
¦
Iresnet_model/conv2d_8/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?resnet_model/conv2d_8/kernel/Initializer/truncated_normal/shape*

seed *
T0*/
_class%
#!loc:@resnet_model/conv2d_8/kernel*
seed2 *
dtype0*'
_output_shapes
:@
´
=resnet_model/conv2d_8/kernel/Initializer/truncated_normal/mulMulIresnet_model/conv2d_8/kernel/Initializer/truncated_normal/TruncatedNormal@resnet_model/conv2d_8/kernel/Initializer/truncated_normal/stddev*
T0*/
_class%
#!loc:@resnet_model/conv2d_8/kernel*'
_output_shapes
:@
¢
9resnet_model/conv2d_8/kernel/Initializer/truncated_normalAdd=resnet_model/conv2d_8/kernel/Initializer/truncated_normal/mul>resnet_model/conv2d_8/kernel/Initializer/truncated_normal/mean*
T0*/
_class%
#!loc:@resnet_model/conv2d_8/kernel*'
_output_shapes
:@
â
resnet_model/conv2d_8/kernel
VariableV2"/device:CPU:0*
	container *
shape:@*
dtype0*'
_output_shapes
:@*
shared_name */
_class%
#!loc:@resnet_model/conv2d_8/kernel
¡
#resnet_model/conv2d_8/kernel/AssignAssignresnet_model/conv2d_8/kernel9resnet_model/conv2d_8/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_8/kernel*
validate_shape(*'
_output_shapes
:@
½
!resnet_model/conv2d_8/kernel/readIdentityresnet_model/conv2d_8/kernel"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_8/kernel*'
_output_shapes
:@

#resnet_model/conv2d_8/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_8/Conv2DConv2Dresnet_model/Relu_6!resnet_model/conv2d_8/kernel/read"/device:GPU:0*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@88*
	dilations
*
T0*
data_formatNCHW*
strides

Ã
9resnet_model/batch_normalization_7/gamma/Initializer/onesConst*;
_class1
/-loc:@resnet_model/batch_normalization_7/gamma*
valueB@*  ?*
dtype0*
_output_shapes
:@
à
(resnet_model/batch_normalization_7/gamma
VariableV2"/device:CPU:0*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_7/gamma
¸
/resnet_model/batch_normalization_7/gamma/AssignAssign(resnet_model/batch_normalization_7/gamma9resnet_model/batch_normalization_7/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_7/gamma*
validate_shape(*
_output_shapes
:@
Ô
-resnet_model/batch_normalization_7/gamma/readIdentity(resnet_model/batch_normalization_7/gamma"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_7/gamma*
_output_shapes
:@
Â
9resnet_model/batch_normalization_7/beta/Initializer/zerosConst*:
_class0
.,loc:@resnet_model/batch_normalization_7/beta*
valueB@*    *
dtype0*
_output_shapes
:@
Þ
'resnet_model/batch_normalization_7/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@resnet_model/batch_normalization_7/beta*
	container *
shape:@
µ
.resnet_model/batch_normalization_7/beta/AssignAssign'resnet_model/batch_normalization_7/beta9resnet_model/batch_normalization_7/beta/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_7/beta
Ñ
,resnet_model/batch_normalization_7/beta/readIdentity'resnet_model/batch_normalization_7/beta"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_7/beta*
_output_shapes
:@
Ð
@resnet_model/batch_normalization_7/moving_mean/Initializer/zerosConst*A
_class7
53loc:@resnet_model/batch_normalization_7/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
ì
.resnet_model/batch_normalization_7/moving_mean
VariableV2"/device:CPU:0*A
_class7
53loc:@resnet_model/batch_normalization_7/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
Ñ
5resnet_model/batch_normalization_7/moving_mean/AssignAssign.resnet_model/batch_normalization_7/moving_mean@resnet_model/batch_normalization_7/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_7/moving_mean*
validate_shape(*
_output_shapes
:@
æ
3resnet_model/batch_normalization_7/moving_mean/readIdentity.resnet_model/batch_normalization_7/moving_mean"/device:CPU:0*
T0*A
_class7
53loc:@resnet_model/batch_normalization_7/moving_mean*
_output_shapes
:@
×
Cresnet_model/batch_normalization_7/moving_variance/Initializer/onesConst*E
_class;
97loc:@resnet_model/batch_normalization_7/moving_variance*
valueB@*  ?*
dtype0*
_output_shapes
:@
ô
2resnet_model/batch_normalization_7/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:@*
shared_name *E
_class;
97loc:@resnet_model/batch_normalization_7/moving_variance*
	container *
shape:@
à
9resnet_model/batch_normalization_7/moving_variance/AssignAssign2resnet_model/batch_normalization_7/moving_varianceCresnet_model/batch_normalization_7/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_7/moving_variance*
validate_shape(*
_output_shapes
:@
ò
7resnet_model/batch_normalization_7/moving_variance/readIdentity2resnet_model/batch_normalization_7/moving_variance"/device:CPU:0*
T0*E
_class;
97loc:@resnet_model/batch_normalization_7/moving_variance*
_output_shapes
:@
À
1resnet_model/batch_normalization_7/FusedBatchNormFusedBatchNormresnet_model/conv2d_8/Conv2D-resnet_model/batch_normalization_7/gamma/read,resnet_model/batch_normalization_7/beta/read3resnet_model/batch_normalization_7/moving_mean/read7resnet_model/batch_normalization_7/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( *
epsilon%ð'7
|
(resnet_model/batch_normalization_7/ConstConst"/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *d;?

resnet_model/Relu_7Relu1resnet_model/batch_normalization_7/FusedBatchNorm"/device:GPU:0*
T0*&
_output_shapes
:@@88
É
?resnet_model/conv2d_9/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*/
_class%
#!loc:@resnet_model/conv2d_9/kernel*%
valueB"      @   @   
´
>resnet_model/conv2d_9/kernel/Initializer/truncated_normal/meanConst*/
_class%
#!loc:@resnet_model/conv2d_9/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¶
@resnet_model/conv2d_9/kernel/Initializer/truncated_normal/stddevConst*/
_class%
#!loc:@resnet_model/conv2d_9/kernel*
valueB
 *«ª*=*
dtype0*
_output_shapes
: 
¥
Iresnet_model/conv2d_9/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?resnet_model/conv2d_9/kernel/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*/
_class%
#!loc:@resnet_model/conv2d_9/kernel*
seed2 
³
=resnet_model/conv2d_9/kernel/Initializer/truncated_normal/mulMulIresnet_model/conv2d_9/kernel/Initializer/truncated_normal/TruncatedNormal@resnet_model/conv2d_9/kernel/Initializer/truncated_normal/stddev*&
_output_shapes
:@@*
T0*/
_class%
#!loc:@resnet_model/conv2d_9/kernel
¡
9resnet_model/conv2d_9/kernel/Initializer/truncated_normalAdd=resnet_model/conv2d_9/kernel/Initializer/truncated_normal/mul>resnet_model/conv2d_9/kernel/Initializer/truncated_normal/mean*
T0*/
_class%
#!loc:@resnet_model/conv2d_9/kernel*&
_output_shapes
:@@
à
resnet_model/conv2d_9/kernel
VariableV2"/device:CPU:0*
dtype0*&
_output_shapes
:@@*
shared_name */
_class%
#!loc:@resnet_model/conv2d_9/kernel*
	container *
shape:@@
 
#resnet_model/conv2d_9/kernel/AssignAssignresnet_model/conv2d_9/kernel9resnet_model/conv2d_9/kernel/Initializer/truncated_normal"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_9/kernel*
validate_shape(*&
_output_shapes
:@@*
use_locking(
¼
!resnet_model/conv2d_9/kernel/readIdentityresnet_model/conv2d_9/kernel"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_9/kernel*&
_output_shapes
:@@

#resnet_model/conv2d_9/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_9/Conv2DConv2Dresnet_model/Relu_7!resnet_model/conv2d_9/kernel/read"/device:GPU:0*&
_output_shapes
:@@88*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ã
9resnet_model/batch_normalization_8/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:@*;
_class1
/-loc:@resnet_model/batch_normalization_8/gamma*
valueB@*  ?
à
(resnet_model/batch_normalization_8/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:@*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_8/gamma*
	container *
shape:@
¸
/resnet_model/batch_normalization_8/gamma/AssignAssign(resnet_model/batch_normalization_8/gamma9resnet_model/batch_normalization_8/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_8/gamma*
validate_shape(*
_output_shapes
:@
Ô
-resnet_model/batch_normalization_8/gamma/readIdentity(resnet_model/batch_normalization_8/gamma"/device:CPU:0*
_output_shapes
:@*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_8/gamma
Â
9resnet_model/batch_normalization_8/beta/Initializer/zerosConst*:
_class0
.,loc:@resnet_model/batch_normalization_8/beta*
valueB@*    *
dtype0*
_output_shapes
:@
Þ
'resnet_model/batch_normalization_8/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes
:@*
shared_name *:
_class0
.,loc:@resnet_model/batch_normalization_8/beta*
	container *
shape:@
µ
.resnet_model/batch_normalization_8/beta/AssignAssign'resnet_model/batch_normalization_8/beta9resnet_model/batch_normalization_8/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_8/beta*
validate_shape(*
_output_shapes
:@
Ñ
,resnet_model/batch_normalization_8/beta/readIdentity'resnet_model/batch_normalization_8/beta"/device:CPU:0*
_output_shapes
:@*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_8/beta
Ð
@resnet_model/batch_normalization_8/moving_mean/Initializer/zerosConst*A
_class7
53loc:@resnet_model/batch_normalization_8/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
ì
.resnet_model/batch_normalization_8/moving_mean
VariableV2"/device:CPU:0*
shared_name *A
_class7
53loc:@resnet_model/batch_normalization_8/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@
Ñ
5resnet_model/batch_normalization_8/moving_mean/AssignAssign.resnet_model/batch_normalization_8/moving_mean@resnet_model/batch_normalization_8/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_8/moving_mean*
validate_shape(*
_output_shapes
:@
æ
3resnet_model/batch_normalization_8/moving_mean/readIdentity.resnet_model/batch_normalization_8/moving_mean"/device:CPU:0*
_output_shapes
:@*
T0*A
_class7
53loc:@resnet_model/batch_normalization_8/moving_mean
×
Cresnet_model/batch_normalization_8/moving_variance/Initializer/onesConst*E
_class;
97loc:@resnet_model/batch_normalization_8/moving_variance*
valueB@*  ?*
dtype0*
_output_shapes
:@
ô
2resnet_model/batch_normalization_8/moving_variance
VariableV2"/device:CPU:0*
shared_name *E
_class;
97loc:@resnet_model/batch_normalization_8/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@
à
9resnet_model/batch_normalization_8/moving_variance/AssignAssign2resnet_model/batch_normalization_8/moving_varianceCresnet_model/batch_normalization_8/moving_variance/Initializer/ones"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_8/moving_variance*
validate_shape(
ò
7resnet_model/batch_normalization_8/moving_variance/readIdentity2resnet_model/batch_normalization_8/moving_variance"/device:CPU:0*
_output_shapes
:@*
T0*E
_class;
97loc:@resnet_model/batch_normalization_8/moving_variance
À
1resnet_model/batch_normalization_8/FusedBatchNormFusedBatchNormresnet_model/conv2d_9/Conv2D-resnet_model/batch_normalization_8/gamma/read,resnet_model/batch_normalization_8/beta/read3resnet_model/batch_normalization_8/moving_mean/read7resnet_model/batch_normalization_8/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( 
|
(resnet_model/batch_normalization_8/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_8Relu1resnet_model/batch_normalization_8/FusedBatchNorm"/device:GPU:0*&
_output_shapes
:@@88*
T0
Ë
@resnet_model/conv2d_10/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_10/kernel*%
valueB"      @      *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_10/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_10/kernel*
valueB
 *    
¸
Aresnet_model/conv2d_10/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_10/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
©
Jresnet_model/conv2d_10/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_10/kernel/Initializer/truncated_normal/shape*
dtype0*'
_output_shapes
:@*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_10/kernel*
seed2 
¸
>resnet_model/conv2d_10/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_10/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_10/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_10/kernel*'
_output_shapes
:@
¦
:resnet_model/conv2d_10/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_10/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_10/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_10/kernel*'
_output_shapes
:@
ä
resnet_model/conv2d_10/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_10/kernel*
	container *
shape:@*
dtype0*'
_output_shapes
:@*
shared_name 
¥
$resnet_model/conv2d_10/kernel/AssignAssignresnet_model/conv2d_10/kernel:resnet_model/conv2d_10/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_10/kernel*
validate_shape(*'
_output_shapes
:@
À
"resnet_model/conv2d_10/kernel/readIdentityresnet_model/conv2d_10/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_10/kernel*'
_output_shapes
:@

$resnet_model/conv2d_10/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_10/Conv2DConv2Dresnet_model/Relu_8"resnet_model/conv2d_10/kernel/read"/device:GPU:0*'
_output_shapes
:@88*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME

resnet_model/add_2Addresnet_model/conv2d_10/Conv2Dresnet_model/add_1"/device:GPU:0*'
_output_shapes
:@88*
T0
z
resnet_model/block_layer1Identityresnet_model/add_2"/device:GPU:0*
T0*'
_output_shapes
:@88
Å
9resnet_model/batch_normalization_9/gamma/Initializer/onesConst*;
_class1
/-loc:@resnet_model/batch_normalization_9/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_9/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_9/gamma*
	container *
shape:
¹
/resnet_model/batch_normalization_9/gamma/AssignAssign(resnet_model/batch_normalization_9/gamma9resnet_model/batch_normalization_9/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_9/gamma*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_9/gamma/readIdentity(resnet_model/batch_normalization_9/gamma"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_9/gamma
Ä
9resnet_model/batch_normalization_9/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*:
_class0
.,loc:@resnet_model/batch_normalization_9/beta*
valueB*    
à
'resnet_model/batch_normalization_9/beta
VariableV2"/device:CPU:0*
shared_name *:
_class0
.,loc:@resnet_model/batch_normalization_9/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
¶
.resnet_model/batch_normalization_9/beta/AssignAssign'resnet_model/batch_normalization_9/beta9resnet_model/batch_normalization_9/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_9/beta*
validate_shape(*
_output_shapes	
:
Ò
,resnet_model/batch_normalization_9/beta/readIdentity'resnet_model/batch_normalization_9/beta"/device:CPU:0*:
_class0
.,loc:@resnet_model/batch_normalization_9/beta*
_output_shapes	
:*
T0
Ò
@resnet_model/batch_normalization_9/moving_mean/Initializer/zerosConst*A
_class7
53loc:@resnet_model/batch_normalization_9/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
î
.resnet_model/batch_normalization_9/moving_mean
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@resnet_model/batch_normalization_9/moving_mean
Ò
5resnet_model/batch_normalization_9/moving_mean/AssignAssign.resnet_model/batch_normalization_9/moving_mean@resnet_model/batch_normalization_9/moving_mean/Initializer/zeros"/device:CPU:0*A
_class7
53loc:@resnet_model/batch_normalization_9/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ç
3resnet_model/batch_normalization_9/moving_mean/readIdentity.resnet_model/batch_normalization_9/moving_mean"/device:CPU:0*
T0*A
_class7
53loc:@resnet_model/batch_normalization_9/moving_mean*
_output_shapes	
:
Ù
Cresnet_model/batch_normalization_9/moving_variance/Initializer/onesConst*
_output_shapes	
:*E
_class;
97loc:@resnet_model/batch_normalization_9/moving_variance*
valueB*  ?*
dtype0
ö
2resnet_model/batch_normalization_9/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *E
_class;
97loc:@resnet_model/batch_normalization_9/moving_variance*
	container *
shape:
á
9resnet_model/batch_normalization_9/moving_variance/AssignAssign2resnet_model/batch_normalization_9/moving_varianceCresnet_model/batch_normalization_9/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_9/moving_variance*
validate_shape(*
_output_shapes	
:
ó
7resnet_model/batch_normalization_9/moving_variance/readIdentity2resnet_model/batch_normalization_9/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*E
_class;
97loc:@resnet_model/batch_normalization_9/moving_variance
Â
1resnet_model/batch_normalization_9/FusedBatchNormFusedBatchNormresnet_model/block_layer1-resnet_model/batch_normalization_9/gamma/read,resnet_model/batch_normalization_9/beta/read3resnet_model/batch_normalization_9/moving_mean/read7resnet_model/batch_normalization_9/moving_variance/read"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:@88::::*
is_training( *
epsilon%ð'7*
T0
|
(resnet_model/batch_normalization_9/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_9Relu1resnet_model/batch_normalization_9/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@88

resnet_model/Pad_1/paddingsConst"/device:GPU:0*9
value0B."                                 *
dtype0*
_output_shapes

:

resnet_model/Pad_1Padresnet_model/Relu_9resnet_model/Pad_1/paddings"/device:GPU:0*
T0*
	Tpaddings0*'
_output_shapes
:@88
Ë
@resnet_model/conv2d_11/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*%
valueB"            
¶
?resnet_model/conv2d_11/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_11/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_11/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_11/kernel/Initializer/truncated_normal/shape*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*
seed2 *
dtype0
¹
>resnet_model/conv2d_11/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_11/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_11/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_11/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_11/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_11/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_11/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
¦
$resnet_model/conv2d_11/kernel/AssignAssignresnet_model/conv2d_11/kernel:resnet_model/conv2d_11/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_11/kernel/readIdentityresnet_model/conv2d_11/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*(
_output_shapes
:

$resnet_model/conv2d_11/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_11/Conv2DConv2Dresnet_model/Pad_1"resnet_model/conv2d_11/kernel/read"/device:GPU:0*
paddingVALID*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(
Ë
@resnet_model/conv2d_12/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_12/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_12/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_12/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_12/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_12/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_12/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_12/kernel/Initializer/truncated_normal/shape*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_12/kernel*
seed2 *
dtype0
¹
>resnet_model/conv2d_12/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_12/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_12/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_12/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_12/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_12/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_12/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_12/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_12/kernel
VariableV2"/device:CPU:0*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_12/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:
¦
$resnet_model/conv2d_12/kernel/AssignAssignresnet_model/conv2d_12/kernel:resnet_model/conv2d_12/kernel/Initializer/truncated_normal"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_12/kernel*
validate_shape(
Á
"resnet_model/conv2d_12/kernel/readIdentityresnet_model/conv2d_12/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_12/kernel*(
_output_shapes
:

$resnet_model/conv2d_12/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_12/Conv2DConv2Dresnet_model/Relu_9"resnet_model/conv2d_12/kernel/read"/device:GPU:0*'
_output_shapes
:@88*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ç
:resnet_model/batch_normalization_10/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_10/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_10/gamma
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_10/gamma
½
0resnet_model/batch_normalization_10/gamma/AssignAssign)resnet_model/batch_normalization_10/gamma:resnet_model/batch_normalization_10/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_10/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_10/gamma/readIdentity)resnet_model/batch_normalization_10/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_10/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_10/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*;
_class1
/-loc:@resnet_model/batch_normalization_10/beta*
valueB*    
â
(resnet_model/batch_normalization_10/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_10/beta*
	container *
shape:
º
/resnet_model/batch_normalization_10/beta/AssignAssign(resnet_model/batch_normalization_10/beta:resnet_model/batch_normalization_10/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_10/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_10/beta/readIdentity(resnet_model/batch_normalization_10/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_10/beta
Ô
Aresnet_model/batch_normalization_10/moving_mean/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_10/moving_mean*
valueB*    *
dtype0
ð
/resnet_model/batch_normalization_10/moving_mean
VariableV2"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_10/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ö
6resnet_model/batch_normalization_10/moving_mean/AssignAssign/resnet_model/batch_normalization_10/moving_meanAresnet_model/batch_normalization_10/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_10/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_10/moving_mean/readIdentity/resnet_model/batch_normalization_10/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_10/moving_mean
Û
Dresnet_model/batch_normalization_10/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_10/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_10/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_10/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_10/moving_variance/AssignAssign3resnet_model/batch_normalization_10/moving_varianceDresnet_model/batch_normalization_10/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_10/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_10/moving_variance/readIdentity3resnet_model/batch_normalization_10/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_10/moving_variance
Ë
2resnet_model/batch_normalization_10/FusedBatchNormFusedBatchNormresnet_model/conv2d_12/Conv2D.resnet_model/batch_normalization_10/gamma/read-resnet_model/batch_normalization_10/beta/read4resnet_model/batch_normalization_10/moving_mean/read8resnet_model/batch_normalization_10/moving_variance/read"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:@88::::*
is_training( *
epsilon%ð'7*
T0
}
)resnet_model/batch_normalization_10/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_10Relu2resnet_model/batch_normalization_10/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@88

resnet_model/Pad_2/paddingsConst"/device:GPU:0*9
value0B."                             *
dtype0*
_output_shapes

:

resnet_model/Pad_2Padresnet_model/Relu_10resnet_model/Pad_2/paddings"/device:GPU:0*'
_output_shapes
:@::*
T0*
	Tpaddings0
Ë
@resnet_model/conv2d_13/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_13/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_13/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_13/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_13/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_13/kernel*
valueB
 *ï[ñ<*
dtype0
ª
Jresnet_model/conv2d_13/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_13/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_13/kernel*
seed2 
¹
>resnet_model/conv2d_13/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_13/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_13/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_13/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_13/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_13/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_13/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_13/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_13/kernel
VariableV2"/device:CPU:0*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_13/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:
¦
$resnet_model/conv2d_13/kernel/AssignAssignresnet_model/conv2d_13/kernel:resnet_model/conv2d_13/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_13/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_13/kernel/readIdentityresnet_model/conv2d_13/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_13/kernel*(
_output_shapes
:

$resnet_model/conv2d_13/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_13/Conv2DConv2Dresnet_model/Pad_2"resnet_model/conv2d_13/kernel/read"/device:GPU:0*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:@*
	dilations

Ç
:resnet_model/batch_normalization_11/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_11/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_11/gamma
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_11/gamma*
	container 
½
0resnet_model/batch_normalization_11/gamma/AssignAssign)resnet_model/batch_normalization_11/gamma:resnet_model/batch_normalization_11/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_11/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_11/gamma/readIdentity)resnet_model/batch_normalization_11/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_11/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_11/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_11/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_11/beta
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_11/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
º
/resnet_model/batch_normalization_11/beta/AssignAssign(resnet_model/batch_normalization_11/beta:resnet_model/batch_normalization_11/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_11/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_11/beta/readIdentity(resnet_model/batch_normalization_11/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_11/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_11/moving_mean/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_11/moving_mean*
valueB*    *
dtype0
ð
/resnet_model/batch_normalization_11/moving_mean
VariableV2"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_11/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ö
6resnet_model/batch_normalization_11/moving_mean/AssignAssign/resnet_model/batch_normalization_11/moving_meanAresnet_model/batch_normalization_11/moving_mean/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_11/moving_mean
ê
4resnet_model/batch_normalization_11/moving_mean/readIdentity/resnet_model/batch_normalization_11/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_11/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_11/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_11/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_11/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_11/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_11/moving_variance/AssignAssign3resnet_model/batch_normalization_11/moving_varianceDresnet_model/batch_normalization_11/moving_variance/Initializer/ones"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_11/moving_variance*
validate_shape(
ö
8resnet_model/batch_normalization_11/moving_variance/readIdentity3resnet_model/batch_normalization_11/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_11/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_11/FusedBatchNormFusedBatchNormresnet_model/conv2d_13/Conv2D.resnet_model/batch_normalization_11/gamma/read-resnet_model/batch_normalization_11/beta/read4resnet_model/batch_normalization_11/moving_mean/read8resnet_model/batch_normalization_11/moving_variance/read"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0
}
)resnet_model/batch_normalization_11/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_11Relu2resnet_model/batch_normalization_11/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_14/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_14/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_14/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_14/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_14/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_14/kernel*
valueB
 *óµ=*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_14/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_14/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_14/kernel*
seed2 
¹
>resnet_model/conv2d_14/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_14/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_14/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_14/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_14/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_14/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_14/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_14/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_14/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_14/kernel*
	container *
shape:
¦
$resnet_model/conv2d_14/kernel/AssignAssignresnet_model/conv2d_14/kernel:resnet_model/conv2d_14/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_14/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_14/kernel/readIdentityresnet_model/conv2d_14/kernel"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_14/kernel*(
_output_shapes
:*
T0

$resnet_model/conv2d_14/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_14/Conv2DConv2Dresnet_model/Relu_11"resnet_model/conv2d_14/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME

resnet_model/add_3Addresnet_model/conv2d_14/Conv2Dresnet_model/conv2d_11/Conv2D"/device:GPU:0*
T0*'
_output_shapes
:@
Ç
:resnet_model/batch_normalization_12/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_12/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_12/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_12/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_12/gamma/AssignAssign)resnet_model/batch_normalization_12/gamma:resnet_model/batch_normalization_12/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_12/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_12/gamma/readIdentity)resnet_model/batch_normalization_12/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_12/gamma
Æ
:resnet_model/batch_normalization_12/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_12/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_12/beta
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_12/beta*
	container 
º
/resnet_model/batch_normalization_12/beta/AssignAssign(resnet_model/batch_normalization_12/beta:resnet_model/batch_normalization_12/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_12/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_12/beta/readIdentity(resnet_model/batch_normalization_12/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_12/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_12/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_12/moving_mean*
valueB*    
ð
/resnet_model/batch_normalization_12/moving_mean
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_12/moving_mean*
	container *
shape:
Ö
6resnet_model/batch_normalization_12/moving_mean/AssignAssign/resnet_model/batch_normalization_12/moving_meanAresnet_model/batch_normalization_12/moving_mean/Initializer/zeros"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_12/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ê
4resnet_model/batch_normalization_12/moving_mean/readIdentity/resnet_model/batch_normalization_12/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_12/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_12/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_12/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_12/moving_variance
VariableV2"/device:CPU:0*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_12/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
å
:resnet_model/batch_normalization_12/moving_variance/AssignAssign3resnet_model/batch_normalization_12/moving_varianceDresnet_model/batch_normalization_12/moving_variance/Initializer/ones"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_12/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
8resnet_model/batch_normalization_12/moving_variance/readIdentity3resnet_model/batch_normalization_12/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_12/moving_variance
À
2resnet_model/batch_normalization_12/FusedBatchNormFusedBatchNormresnet_model/add_3.resnet_model/batch_normalization_12/gamma/read-resnet_model/batch_normalization_12/beta/read4resnet_model/batch_normalization_12/moving_mean/read8resnet_model/batch_normalization_12/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_12/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_12Relu2resnet_model/batch_normalization_12/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_15/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*0
_class&
$"loc:@resnet_model/conv2d_15/kernel*%
valueB"            
¶
?resnet_model/conv2d_15/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_15/kernel*
valueB
 *    *
dtype0
¸
Aresnet_model/conv2d_15/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_15/kernel*
valueB
 *ó5=
ª
Jresnet_model/conv2d_15/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_15/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@resnet_model/conv2d_15/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed 
¹
>resnet_model/conv2d_15/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_15/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_15/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_15/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_15/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_15/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_15/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_15/kernel
æ
resnet_model/conv2d_15/kernel
VariableV2"/device:CPU:0*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_15/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:
¦
$resnet_model/conv2d_15/kernel/AssignAssignresnet_model/conv2d_15/kernel:resnet_model/conv2d_15/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_15/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_15/kernel/readIdentityresnet_model/conv2d_15/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_15/kernel*(
_output_shapes
:

$resnet_model/conv2d_15/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_15/Conv2DConv2Dresnet_model/Relu_12"resnet_model/conv2d_15/kernel/read"/device:GPU:0*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides

Ç
:resnet_model/batch_normalization_13/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_13/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_13/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_13/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_13/gamma/AssignAssign)resnet_model/batch_normalization_13/gamma:resnet_model/batch_normalization_13/gamma/Initializer/ones"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_13/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Ø
.resnet_model/batch_normalization_13/gamma/readIdentity)resnet_model/batch_normalization_13/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_13/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_13/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_13/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_13/beta
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_13/beta*
	container 
º
/resnet_model/batch_normalization_13/beta/AssignAssign(resnet_model/batch_normalization_13/beta:resnet_model/batch_normalization_13/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_13/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_13/beta/readIdentity(resnet_model/batch_normalization_13/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_13/beta
Ô
Aresnet_model/batch_normalization_13/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_13/moving_mean*
valueB*    
ð
/resnet_model/batch_normalization_13/moving_mean
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_13/moving_mean*
	container *
shape:
Ö
6resnet_model/batch_normalization_13/moving_mean/AssignAssign/resnet_model/batch_normalization_13/moving_meanAresnet_model/batch_normalization_13/moving_mean/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_13/moving_mean
ê
4resnet_model/batch_normalization_13/moving_mean/readIdentity/resnet_model/batch_normalization_13/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_13/moving_mean
Û
Dresnet_model/batch_normalization_13/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_13/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_13/moving_variance
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_13/moving_variance
å
:resnet_model/batch_normalization_13/moving_variance/AssignAssign3resnet_model/batch_normalization_13/moving_varianceDresnet_model/batch_normalization_13/moving_variance/Initializer/ones"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_13/moving_variance*
validate_shape(
ö
8resnet_model/batch_normalization_13/moving_variance/readIdentity3resnet_model/batch_normalization_13/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_13/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_13/FusedBatchNormFusedBatchNormresnet_model/conv2d_15/Conv2D.resnet_model/batch_normalization_13/gamma/read-resnet_model/batch_normalization_13/beta/read4resnet_model/batch_normalization_13/moving_mean/read8resnet_model/batch_normalization_13/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_13/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_13Relu2resnet_model/batch_normalization_13/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_16/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_16/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_16/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_16/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_16/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_16/kernel*
valueB
 *ï[ñ<*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_16/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_16/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@resnet_model/conv2d_16/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed 
¹
>resnet_model/conv2d_16/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_16/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_16/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_16/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_16/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_16/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_16/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_16/kernel
æ
resnet_model/conv2d_16/kernel
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_16/kernel
¦
$resnet_model/conv2d_16/kernel/AssignAssignresnet_model/conv2d_16/kernel:resnet_model/conv2d_16/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_16/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_16/kernel/readIdentityresnet_model/conv2d_16/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_16/kernel*(
_output_shapes
:

$resnet_model/conv2d_16/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_16/Conv2DConv2Dresnet_model/Relu_13"resnet_model/conv2d_16/kernel/read"/device:GPU:0*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Ç
:resnet_model/batch_normalization_14/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_14/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_14/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_14/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_14/gamma/AssignAssign)resnet_model/batch_normalization_14/gamma:resnet_model/batch_normalization_14/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_14/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_14/gamma/readIdentity)resnet_model/batch_normalization_14/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_14/gamma
Æ
:resnet_model/batch_normalization_14/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_14/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_14/beta
VariableV2"/device:CPU:0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_14/beta*
	container *
shape:*
dtype0
º
/resnet_model/batch_normalization_14/beta/AssignAssign(resnet_model/batch_normalization_14/beta:resnet_model/batch_normalization_14/beta/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_14/beta
Õ
-resnet_model/batch_normalization_14/beta/readIdentity(resnet_model/batch_normalization_14/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_14/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_14/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_14/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_14/moving_mean
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_14/moving_mean
Ö
6resnet_model/batch_normalization_14/moving_mean/AssignAssign/resnet_model/batch_normalization_14/moving_meanAresnet_model/batch_normalization_14/moving_mean/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_14/moving_mean
ê
4resnet_model/batch_normalization_14/moving_mean/readIdentity/resnet_model/batch_normalization_14/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_14/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_14/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_14/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_14/moving_variance
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_14/moving_variance
å
:resnet_model/batch_normalization_14/moving_variance/AssignAssign3resnet_model/batch_normalization_14/moving_varianceDresnet_model/batch_normalization_14/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_14/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_14/moving_variance/readIdentity3resnet_model/batch_normalization_14/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_14/moving_variance
Ë
2resnet_model/batch_normalization_14/FusedBatchNormFusedBatchNormresnet_model/conv2d_16/Conv2D.resnet_model/batch_normalization_14/gamma/read-resnet_model/batch_normalization_14/beta/read4resnet_model/batch_normalization_14/moving_mean/read8resnet_model/batch_normalization_14/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7
}
)resnet_model/batch_normalization_14/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_14Relu2resnet_model/batch_normalization_14/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_17/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_17/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_17/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_17/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_17/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_17/kernel*
valueB
 *óµ=
ª
Jresnet_model/conv2d_17/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_17/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_17/kernel
¹
>resnet_model/conv2d_17/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_17/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_17/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_17/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_17/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_17/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_17/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_17/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_17/kernel
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_17/kernel
¦
$resnet_model/conv2d_17/kernel/AssignAssignresnet_model/conv2d_17/kernel:resnet_model/conv2d_17/kernel/Initializer/truncated_normal"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_17/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Á
"resnet_model/conv2d_17/kernel/readIdentityresnet_model/conv2d_17/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_17/kernel*(
_output_shapes
:

$resnet_model/conv2d_17/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_17/Conv2DConv2Dresnet_model/Relu_14"resnet_model/conv2d_17/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME

resnet_model/add_4Addresnet_model/conv2d_17/Conv2Dresnet_model/add_3"/device:GPU:0*'
_output_shapes
:@*
T0
Ç
:resnet_model/batch_normalization_15/gamma/Initializer/onesConst*
_output_shapes	
:*<
_class2
0.loc:@resnet_model/batch_normalization_15/gamma*
valueB*  ?*
dtype0
ä
)resnet_model/batch_normalization_15/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_15/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_15/gamma/AssignAssign)resnet_model/batch_normalization_15/gamma:resnet_model/batch_normalization_15/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_15/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_15/gamma/readIdentity)resnet_model/batch_normalization_15/gamma"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_15/gamma*
_output_shapes	
:*
T0
Æ
:resnet_model/batch_normalization_15/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_15/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_15/beta
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_15/beta
º
/resnet_model/batch_normalization_15/beta/AssignAssign(resnet_model/batch_normalization_15/beta:resnet_model/batch_normalization_15/beta/Initializer/zeros"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_15/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Õ
-resnet_model/batch_normalization_15/beta/readIdentity(resnet_model/batch_normalization_15/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_15/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_15/moving_mean/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_15/moving_mean*
valueB*    *
dtype0
ð
/resnet_model/batch_normalization_15/moving_mean
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_15/moving_mean
Ö
6resnet_model/batch_normalization_15/moving_mean/AssignAssign/resnet_model/batch_normalization_15/moving_meanAresnet_model/batch_normalization_15/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_15/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_15/moving_mean/readIdentity/resnet_model/batch_normalization_15/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_15/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_15/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_15/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_15/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_15/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_15/moving_variance/AssignAssign3resnet_model/batch_normalization_15/moving_varianceDresnet_model/batch_normalization_15/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_15/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_15/moving_variance/readIdentity3resnet_model/batch_normalization_15/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_15/moving_variance*
_output_shapes	
:
À
2resnet_model/batch_normalization_15/FusedBatchNormFusedBatchNormresnet_model/add_4.resnet_model/batch_normalization_15/gamma/read-resnet_model/batch_normalization_15/beta/read4resnet_model/batch_normalization_15/moving_mean/read8resnet_model/batch_normalization_15/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7
}
)resnet_model/batch_normalization_15/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_15Relu2resnet_model/batch_normalization_15/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_18/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*0
_class&
$"loc:@resnet_model/conv2d_18/kernel*%
valueB"            *
dtype0
¶
?resnet_model/conv2d_18/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_18/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_18/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_18/kernel*
valueB
 *ó5=*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_18/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_18/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_18/kernel*
seed2 
¹
>resnet_model/conv2d_18/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_18/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_18/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_18/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_18/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_18/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_18/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_18/kernel
æ
resnet_model/conv2d_18/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_18/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
¦
$resnet_model/conv2d_18/kernel/AssignAssignresnet_model/conv2d_18/kernel:resnet_model/conv2d_18/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_18/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_18/kernel/readIdentityresnet_model/conv2d_18/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_18/kernel*(
_output_shapes
:

$resnet_model/conv2d_18/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_18/Conv2DConv2Dresnet_model/Relu_15"resnet_model/conv2d_18/kernel/read"/device:GPU:0*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Ç
:resnet_model/batch_normalization_16/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_16/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_16/gamma
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_16/gamma
½
0resnet_model/batch_normalization_16/gamma/AssignAssign)resnet_model/batch_normalization_16/gamma:resnet_model/batch_normalization_16/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_16/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_16/gamma/readIdentity)resnet_model/batch_normalization_16/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_16/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_16/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_16/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_16/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_16/beta*
	container *
shape:
º
/resnet_model/batch_normalization_16/beta/AssignAssign(resnet_model/batch_normalization_16/beta:resnet_model/batch_normalization_16/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_16/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_16/beta/readIdentity(resnet_model/batch_normalization_16/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_16/beta
Ô
Aresnet_model/batch_normalization_16/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_16/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_16/moving_mean
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_16/moving_mean
Ö
6resnet_model/batch_normalization_16/moving_mean/AssignAssign/resnet_model/batch_normalization_16/moving_meanAresnet_model/batch_normalization_16/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_16/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_16/moving_mean/readIdentity/resnet_model/batch_normalization_16/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_16/moving_mean
Û
Dresnet_model/batch_normalization_16/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_16/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_16/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_16/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_16/moving_variance/AssignAssign3resnet_model/batch_normalization_16/moving_varianceDresnet_model/batch_normalization_16/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_16/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_16/moving_variance/readIdentity3resnet_model/batch_normalization_16/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_16/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_16/FusedBatchNormFusedBatchNormresnet_model/conv2d_18/Conv2D.resnet_model/batch_normalization_16/gamma/read-resnet_model/batch_normalization_16/beta/read4resnet_model/batch_normalization_16/moving_mean/read8resnet_model/batch_normalization_16/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_16/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_16Relu2resnet_model/batch_normalization_16/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_19/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_19/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_19/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_19/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_19/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_19/kernel*
valueB
 *ï[ñ<*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_19/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_19/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_19/kernel*
seed2 
¹
>resnet_model/conv2d_19/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_19/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_19/kernel/Initializer/truncated_normal/stddev*0
_class&
$"loc:@resnet_model/conv2d_19/kernel*(
_output_shapes
:*
T0
§
:resnet_model/conv2d_19/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_19/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_19/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_19/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_19/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_19/kernel*
	container *
shape:
¦
$resnet_model/conv2d_19/kernel/AssignAssignresnet_model/conv2d_19/kernel:resnet_model/conv2d_19/kernel/Initializer/truncated_normal"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_19/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Á
"resnet_model/conv2d_19/kernel/readIdentityresnet_model/conv2d_19/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_19/kernel*(
_output_shapes
:

$resnet_model/conv2d_19/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_19/Conv2DConv2Dresnet_model/Relu_16"resnet_model/conv2d_19/kernel/read"/device:GPU:0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0
Ç
:resnet_model/batch_normalization_17/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_17/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_17/gamma
VariableV2"/device:CPU:0*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_17/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
½
0resnet_model/batch_normalization_17/gamma/AssignAssign)resnet_model/batch_normalization_17/gamma:resnet_model/batch_normalization_17/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_17/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_17/gamma/readIdentity)resnet_model/batch_normalization_17/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_17/gamma
Æ
:resnet_model/batch_normalization_17/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_17/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_17/beta
VariableV2"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_17/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
º
/resnet_model/batch_normalization_17/beta/AssignAssign(resnet_model/batch_normalization_17/beta:resnet_model/batch_normalization_17/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_17/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_17/beta/readIdentity(resnet_model/batch_normalization_17/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_17/beta
Ô
Aresnet_model/batch_normalization_17/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_17/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_17/moving_mean
VariableV2"/device:CPU:0*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_17/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
6resnet_model/batch_normalization_17/moving_mean/AssignAssign/resnet_model/batch_normalization_17/moving_meanAresnet_model/batch_normalization_17/moving_mean/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_17/moving_mean
ê
4resnet_model/batch_normalization_17/moving_mean/readIdentity/resnet_model/batch_normalization_17/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_17/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_17/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_17/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_17/moving_variance
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_17/moving_variance*
	container 
å
:resnet_model/batch_normalization_17/moving_variance/AssignAssign3resnet_model/batch_normalization_17/moving_varianceDresnet_model/batch_normalization_17/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_17/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_17/moving_variance/readIdentity3resnet_model/batch_normalization_17/moving_variance"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_17/moving_variance*
_output_shapes	
:*
T0
Ë
2resnet_model/batch_normalization_17/FusedBatchNormFusedBatchNormresnet_model/conv2d_19/Conv2D.resnet_model/batch_normalization_17/gamma/read-resnet_model/batch_normalization_17/beta/read4resnet_model/batch_normalization_17/moving_mean/read8resnet_model/batch_normalization_17/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_17/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_17Relu2resnet_model/batch_normalization_17/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_20/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_20/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_20/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_20/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_20/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_20/kernel*
valueB
 *óµ=*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_20/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_20/kernel/Initializer/truncated_normal/shape*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_20/kernel*
seed2 *
dtype0
¹
>resnet_model/conv2d_20/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_20/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_20/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_20/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_20/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_20/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_20/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_20/kernel
æ
resnet_model/conv2d_20/kernel
VariableV2"/device:CPU:0*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_20/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:
¦
$resnet_model/conv2d_20/kernel/AssignAssignresnet_model/conv2d_20/kernel:resnet_model/conv2d_20/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_20/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_20/kernel/readIdentityresnet_model/conv2d_20/kernel"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_20/kernel*(
_output_shapes
:*
T0

$resnet_model/conv2d_20/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_20/Conv2DConv2Dresnet_model/Relu_17"resnet_model/conv2d_20/kernel/read"/device:GPU:0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0

resnet_model/add_5Addresnet_model/conv2d_20/Conv2Dresnet_model/add_4"/device:GPU:0*'
_output_shapes
:@*
T0
Ç
:resnet_model/batch_normalization_18/gamma/Initializer/onesConst*
_output_shapes	
:*<
_class2
0.loc:@resnet_model/batch_normalization_18/gamma*
valueB*  ?*
dtype0
ä
)resnet_model/batch_normalization_18/gamma
VariableV2"/device:CPU:0*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_18/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
½
0resnet_model/batch_normalization_18/gamma/AssignAssign)resnet_model/batch_normalization_18/gamma:resnet_model/batch_normalization_18/gamma/Initializer/ones"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_18/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Ø
.resnet_model/batch_normalization_18/gamma/readIdentity)resnet_model/batch_normalization_18/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_18/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_18/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*;
_class1
/-loc:@resnet_model/batch_normalization_18/beta*
valueB*    
â
(resnet_model/batch_normalization_18/beta
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_18/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
º
/resnet_model/batch_normalization_18/beta/AssignAssign(resnet_model/batch_normalization_18/beta:resnet_model/batch_normalization_18/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_18/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_18/beta/readIdentity(resnet_model/batch_normalization_18/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_18/beta
Ô
Aresnet_model/batch_normalization_18/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_18/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_18/moving_mean
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_18/moving_mean
Ö
6resnet_model/batch_normalization_18/moving_mean/AssignAssign/resnet_model/batch_normalization_18/moving_meanAresnet_model/batch_normalization_18/moving_mean/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_18/moving_mean
ê
4resnet_model/batch_normalization_18/moving_mean/readIdentity/resnet_model/batch_normalization_18/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_18/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_18/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_18/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_18/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_18/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_18/moving_variance/AssignAssign3resnet_model/batch_normalization_18/moving_varianceDresnet_model/batch_normalization_18/moving_variance/Initializer/ones"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_18/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
8resnet_model/batch_normalization_18/moving_variance/readIdentity3resnet_model/batch_normalization_18/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_18/moving_variance*
_output_shapes	
:
À
2resnet_model/batch_normalization_18/FusedBatchNormFusedBatchNormresnet_model/add_5.resnet_model/batch_normalization_18/gamma/read-resnet_model/batch_normalization_18/beta/read4resnet_model/batch_normalization_18/moving_mean/read8resnet_model/batch_normalization_18/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_18/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_18Relu2resnet_model/batch_normalization_18/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_21/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*0
_class&
$"loc:@resnet_model/conv2d_21/kernel*%
valueB"            
¶
?resnet_model/conv2d_21/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_21/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_21/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_21/kernel*
valueB
 *ó5=*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_21/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_21/kernel/Initializer/truncated_normal/shape*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_21/kernel*
seed2 *
dtype0*(
_output_shapes
:
¹
>resnet_model/conv2d_21/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_21/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_21/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_21/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_21/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_21/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_21/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_21/kernel
æ
resnet_model/conv2d_21/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_21/kernel*
	container *
shape:
¦
$resnet_model/conv2d_21/kernel/AssignAssignresnet_model/conv2d_21/kernel:resnet_model/conv2d_21/kernel/Initializer/truncated_normal"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_21/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
Á
"resnet_model/conv2d_21/kernel/readIdentityresnet_model/conv2d_21/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_21/kernel*(
_output_shapes
:

$resnet_model/conv2d_21/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_21/Conv2DConv2Dresnet_model/Relu_18"resnet_model/conv2d_21/kernel/read"/device:GPU:0*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations

Ç
:resnet_model/batch_normalization_19/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*<
_class2
0.loc:@resnet_model/batch_normalization_19/gamma*
valueB*  ?
ä
)resnet_model/batch_normalization_19/gamma
VariableV2"/device:CPU:0*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_19/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
½
0resnet_model/batch_normalization_19/gamma/AssignAssign)resnet_model/batch_normalization_19/gamma:resnet_model/batch_normalization_19/gamma/Initializer/ones"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_19/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Ø
.resnet_model/batch_normalization_19/gamma/readIdentity)resnet_model/batch_normalization_19/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_19/gamma
Æ
:resnet_model/batch_normalization_19/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_19/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_19/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_19/beta*
	container *
shape:
º
/resnet_model/batch_normalization_19/beta/AssignAssign(resnet_model/batch_normalization_19/beta:resnet_model/batch_normalization_19/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_19/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_19/beta/readIdentity(resnet_model/batch_normalization_19/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_19/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_19/moving_mean/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_19/moving_mean*
valueB*    *
dtype0
ð
/resnet_model/batch_normalization_19/moving_mean
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_19/moving_mean*
	container *
shape:
Ö
6resnet_model/batch_normalization_19/moving_mean/AssignAssign/resnet_model/batch_normalization_19/moving_meanAresnet_model/batch_normalization_19/moving_mean/Initializer/zeros"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_19/moving_mean*
validate_shape(
ê
4resnet_model/batch_normalization_19/moving_mean/readIdentity/resnet_model/batch_normalization_19/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_19/moving_mean
Û
Dresnet_model/batch_normalization_19/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_19/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_19/moving_variance
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_19/moving_variance*
	container 
å
:resnet_model/batch_normalization_19/moving_variance/AssignAssign3resnet_model/batch_normalization_19/moving_varianceDresnet_model/batch_normalization_19/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_19/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_19/moving_variance/readIdentity3resnet_model/batch_normalization_19/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_19/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_19/FusedBatchNormFusedBatchNormresnet_model/conv2d_21/Conv2D.resnet_model/batch_normalization_19/gamma/read-resnet_model/batch_normalization_19/beta/read4resnet_model/batch_normalization_19/moving_mean/read8resnet_model/batch_normalization_19/moving_variance/read"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0
}
)resnet_model/batch_normalization_19/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_19Relu2resnet_model/batch_normalization_19/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_22/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_22/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_22/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_22/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_22/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_22/kernel*
valueB
 *ï[ñ<*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_22/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_22/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@resnet_model/conv2d_22/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed 
¹
>resnet_model/conv2d_22/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_22/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_22/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_22/kernel
§
:resnet_model/conv2d_22/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_22/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_22/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_22/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_22/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_22/kernel*
	container *
shape:
¦
$resnet_model/conv2d_22/kernel/AssignAssignresnet_model/conv2d_22/kernel:resnet_model/conv2d_22/kernel/Initializer/truncated_normal"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_22/kernel
Á
"resnet_model/conv2d_22/kernel/readIdentityresnet_model/conv2d_22/kernel"/device:CPU:0*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_22/kernel

$resnet_model/conv2d_22/dilation_rateConst"/device:GPU:0*
dtype0*
_output_shapes
:*
valueB"      

resnet_model/conv2d_22/Conv2DConv2Dresnet_model/Relu_19"resnet_model/conv2d_22/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME
Ç
:resnet_model/batch_normalization_20/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_20/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_20/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_20/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_20/gamma/AssignAssign)resnet_model/batch_normalization_20/gamma:resnet_model/batch_normalization_20/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_20/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_20/gamma/readIdentity)resnet_model/batch_normalization_20/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_20/gamma
Æ
:resnet_model/batch_normalization_20/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_20/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_20/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_20/beta*
	container *
shape:
º
/resnet_model/batch_normalization_20/beta/AssignAssign(resnet_model/batch_normalization_20/beta:resnet_model/batch_normalization_20/beta/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_20/beta
Õ
-resnet_model/batch_normalization_20/beta/readIdentity(resnet_model/batch_normalization_20/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_20/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_20/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_20/moving_mean*
valueB*    
ð
/resnet_model/batch_normalization_20/moving_mean
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_20/moving_mean*
	container *
shape:
Ö
6resnet_model/batch_normalization_20/moving_mean/AssignAssign/resnet_model/batch_normalization_20/moving_meanAresnet_model/batch_normalization_20/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_20/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_20/moving_mean/readIdentity/resnet_model/batch_normalization_20/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_20/moving_mean
Û
Dresnet_model/batch_normalization_20/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_20/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_20/moving_variance
VariableV2"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_20/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
å
:resnet_model/batch_normalization_20/moving_variance/AssignAssign3resnet_model/batch_normalization_20/moving_varianceDresnet_model/batch_normalization_20/moving_variance/Initializer/ones"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_20/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
8resnet_model/batch_normalization_20/moving_variance/readIdentity3resnet_model/batch_normalization_20/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_20/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_20/FusedBatchNormFusedBatchNormresnet_model/conv2d_22/Conv2D.resnet_model/batch_normalization_20/gamma/read-resnet_model/batch_normalization_20/beta/read4resnet_model/batch_normalization_20/moving_mean/read8resnet_model/batch_normalization_20/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_20/ConstConst"/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *d;?

resnet_model/Relu_20Relu2resnet_model/batch_normalization_20/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_23/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*0
_class&
$"loc:@resnet_model/conv2d_23/kernel*%
valueB"            *
dtype0
¶
?resnet_model/conv2d_23/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_23/kernel*
valueB
 *    *
dtype0
¸
Aresnet_model/conv2d_23/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_23/kernel*
valueB
 *óµ=*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_23/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_23/kernel/Initializer/truncated_normal/shape*0
_class&
$"loc:@resnet_model/conv2d_23/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0
¹
>resnet_model/conv2d_23/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_23/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_23/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_23/kernel
§
:resnet_model/conv2d_23/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_23/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_23/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_23/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_23/kernel
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_23/kernel
¦
$resnet_model/conv2d_23/kernel/AssignAssignresnet_model/conv2d_23/kernel:resnet_model/conv2d_23/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_23/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_23/kernel/readIdentityresnet_model/conv2d_23/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_23/kernel*(
_output_shapes
:

$resnet_model/conv2d_23/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_23/Conv2DConv2Dresnet_model/Relu_20"resnet_model/conv2d_23/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME

resnet_model/add_6Addresnet_model/conv2d_23/Conv2Dresnet_model/add_5"/device:GPU:0*'
_output_shapes
:@*
T0
z
resnet_model/block_layer2Identityresnet_model/add_6"/device:GPU:0*'
_output_shapes
:@*
T0
Ç
:resnet_model/batch_normalization_21/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_21/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_21/gamma
VariableV2"/device:CPU:0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_21/gamma*
	container *
shape:*
dtype0
½
0resnet_model/batch_normalization_21/gamma/AssignAssign)resnet_model/batch_normalization_21/gamma:resnet_model/batch_normalization_21/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_21/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_21/gamma/readIdentity)resnet_model/batch_normalization_21/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_21/gamma
Æ
:resnet_model/batch_normalization_21/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_21/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_21/beta
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_21/beta*
	container 
º
/resnet_model/batch_normalization_21/beta/AssignAssign(resnet_model/batch_normalization_21/beta:resnet_model/batch_normalization_21/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_21/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_21/beta/readIdentity(resnet_model/batch_normalization_21/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_21/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_21/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_21/moving_mean*
valueB*    
ð
/resnet_model/batch_normalization_21/moving_mean
VariableV2"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_21/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ö
6resnet_model/batch_normalization_21/moving_mean/AssignAssign/resnet_model/batch_normalization_21/moving_meanAresnet_model/batch_normalization_21/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_21/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_21/moving_mean/readIdentity/resnet_model/batch_normalization_21/moving_mean"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_21/moving_mean*
_output_shapes	
:*
T0
Û
Dresnet_model/batch_normalization_21/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_21/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_21/moving_variance
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_21/moving_variance*
	container 
å
:resnet_model/batch_normalization_21/moving_variance/AssignAssign3resnet_model/batch_normalization_21/moving_varianceDresnet_model/batch_normalization_21/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_21/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_21/moving_variance/readIdentity3resnet_model/batch_normalization_21/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_21/moving_variance*
_output_shapes	
:
Ç
2resnet_model/batch_normalization_21/FusedBatchNormFusedBatchNormresnet_model/block_layer2.resnet_model/batch_normalization_21/gamma/read-resnet_model/batch_normalization_21/beta/read4resnet_model/batch_normalization_21/moving_mean/read8resnet_model/batch_normalization_21/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_21/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_21Relu2resnet_model/batch_normalization_21/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@

resnet_model/Pad_3/paddingsConst"/device:GPU:0*
dtype0*
_output_shapes

:*9
value0B."                                 

resnet_model/Pad_3Padresnet_model/Relu_21resnet_model/Pad_3/paddings"/device:GPU:0*
T0*
	Tpaddings0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_24/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_24/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_24/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_24/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_24/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_24/kernel*
valueB
 *ó5=*
dtype0
ª
Jresnet_model/conv2d_24/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_24/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_24/kernel*
seed2 
¹
>resnet_model/conv2d_24/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_24/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_24/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_24/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_24/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_24/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_24/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_24/kernel
æ
resnet_model/conv2d_24/kernel
VariableV2"/device:CPU:0*
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_24/kernel*
	container 
¦
$resnet_model/conv2d_24/kernel/AssignAssignresnet_model/conv2d_24/kernel:resnet_model/conv2d_24/kernel/Initializer/truncated_normal"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_24/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Á
"resnet_model/conv2d_24/kernel/readIdentityresnet_model/conv2d_24/kernel"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_24/kernel*(
_output_shapes
:*
T0

$resnet_model/conv2d_24/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_24/Conv2DConv2Dresnet_model/Pad_3"resnet_model/conv2d_24/kernel/read"/device:GPU:0*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides

Ë
@resnet_model/conv2d_25/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_25/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_25/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_25/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_25/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_25/kernel*
valueB
 *ó5=*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_25/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_25/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@resnet_model/conv2d_25/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed 
¹
>resnet_model/conv2d_25/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_25/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_25/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_25/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_25/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_25/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_25/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_25/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_25/kernel
VariableV2"/device:CPU:0*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_25/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:
¦
$resnet_model/conv2d_25/kernel/AssignAssignresnet_model/conv2d_25/kernel:resnet_model/conv2d_25/kernel/Initializer/truncated_normal"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_25/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Á
"resnet_model/conv2d_25/kernel/readIdentityresnet_model/conv2d_25/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_25/kernel*(
_output_shapes
:

$resnet_model/conv2d_25/dilation_rateConst"/device:GPU:0*
dtype0*
_output_shapes
:*
valueB"      

resnet_model/conv2d_25/Conv2DConv2Dresnet_model/Relu_21"resnet_model/conv2d_25/kernel/read"/device:GPU:0*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
Ç
:resnet_model/batch_normalization_22/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_22/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_22/gamma
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_22/gamma
½
0resnet_model/batch_normalization_22/gamma/AssignAssign)resnet_model/batch_normalization_22/gamma:resnet_model/batch_normalization_22/gamma/Initializer/ones"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_22/gamma*
validate_shape(
Ø
.resnet_model/batch_normalization_22/gamma/readIdentity)resnet_model/batch_normalization_22/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_22/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_22/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_22/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_22/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_22/beta*
	container *
shape:
º
/resnet_model/batch_normalization_22/beta/AssignAssign(resnet_model/batch_normalization_22/beta:resnet_model/batch_normalization_22/beta/Initializer/zeros"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_22/beta*
validate_shape(
Õ
-resnet_model/batch_normalization_22/beta/readIdentity(resnet_model/batch_normalization_22/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_22/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_22/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_22/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_22/moving_mean
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_22/moving_mean*
	container 
Ö
6resnet_model/batch_normalization_22/moving_mean/AssignAssign/resnet_model/batch_normalization_22/moving_meanAresnet_model/batch_normalization_22/moving_mean/Initializer/zeros"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_22/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ê
4resnet_model/batch_normalization_22/moving_mean/readIdentity/resnet_model/batch_normalization_22/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_22/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_22/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_22/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_22/moving_variance
VariableV2"/device:CPU:0*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_22/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
å
:resnet_model/batch_normalization_22/moving_variance/AssignAssign3resnet_model/batch_normalization_22/moving_varianceDresnet_model/batch_normalization_22/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_22/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_22/moving_variance/readIdentity3resnet_model/batch_normalization_22/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_22/moving_variance
Ë
2resnet_model/batch_normalization_22/FusedBatchNormFusedBatchNormresnet_model/conv2d_25/Conv2D.resnet_model/batch_normalization_22/gamma/read-resnet_model/batch_normalization_22/beta/read4resnet_model/batch_normalization_22/moving_mean/read8resnet_model/batch_normalization_22/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_22/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_22Relu2resnet_model/batch_normalization_22/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@

resnet_model/Pad_4/paddingsConst"/device:GPU:0*
dtype0*
_output_shapes

:*9
value0B."                             

resnet_model/Pad_4Padresnet_model/Relu_22resnet_model/Pad_4/paddings"/device:GPU:0*
T0*
	Tpaddings0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_26/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_26/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_26/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_26/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_26/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_26/kernel*
valueB
 *«ªª<*
dtype0
ª
Jresnet_model/conv2d_26/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_26/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_26/kernel*
seed2 
¹
>resnet_model/conv2d_26/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_26/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_26/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_26/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_26/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_26/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_26/kernel/Initializer/truncated_normal/mean*0
_class&
$"loc:@resnet_model/conv2d_26/kernel*(
_output_shapes
:*
T0
æ
resnet_model/conv2d_26/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_26/kernel*
	container *
shape:
¦
$resnet_model/conv2d_26/kernel/AssignAssignresnet_model/conv2d_26/kernel:resnet_model/conv2d_26/kernel/Initializer/truncated_normal"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_26/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
Á
"resnet_model/conv2d_26/kernel/readIdentityresnet_model/conv2d_26/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_26/kernel*(
_output_shapes
:

$resnet_model/conv2d_26/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_26/Conv2DConv2Dresnet_model/Pad_4"resnet_model/conv2d_26/kernel/read"/device:GPU:0*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:@
Ç
:resnet_model/batch_normalization_23/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_23/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_23/gamma
VariableV2"/device:CPU:0*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_23/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
½
0resnet_model/batch_normalization_23/gamma/AssignAssign)resnet_model/batch_normalization_23/gamma:resnet_model/batch_normalization_23/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_23/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_23/gamma/readIdentity)resnet_model/batch_normalization_23/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_23/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_23/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_23/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_23/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_23/beta*
	container *
shape:
º
/resnet_model/batch_normalization_23/beta/AssignAssign(resnet_model/batch_normalization_23/beta:resnet_model/batch_normalization_23/beta/Initializer/zeros"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_23/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Õ
-resnet_model/batch_normalization_23/beta/readIdentity(resnet_model/batch_normalization_23/beta"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_23/beta*
_output_shapes	
:*
T0
Ô
Aresnet_model/batch_normalization_23/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_23/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_23/moving_mean
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_23/moving_mean*
	container *
shape:
Ö
6resnet_model/batch_normalization_23/moving_mean/AssignAssign/resnet_model/batch_normalization_23/moving_meanAresnet_model/batch_normalization_23/moving_mean/Initializer/zeros"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_23/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ê
4resnet_model/batch_normalization_23/moving_mean/readIdentity/resnet_model/batch_normalization_23/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_23/moving_mean
Û
Dresnet_model/batch_normalization_23/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_23/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_23/moving_variance
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_23/moving_variance
å
:resnet_model/batch_normalization_23/moving_variance/AssignAssign3resnet_model/batch_normalization_23/moving_varianceDresnet_model/batch_normalization_23/moving_variance/Initializer/ones"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_23/moving_variance
ö
8resnet_model/batch_normalization_23/moving_variance/readIdentity3resnet_model/batch_normalization_23/moving_variance"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_23/moving_variance*
_output_shapes	
:*
T0
Ë
2resnet_model/batch_normalization_23/FusedBatchNormFusedBatchNormresnet_model/conv2d_26/Conv2D.resnet_model/batch_normalization_23/gamma/read-resnet_model/batch_normalization_23/beta/read4resnet_model/batch_normalization_23/moving_mean/read8resnet_model/batch_normalization_23/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_23/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_23Relu2resnet_model/batch_normalization_23/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_27/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_27/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_27/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_27/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_27/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_27/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_27/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_27/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_27/kernel
¹
>resnet_model/conv2d_27/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_27/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_27/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_27/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_27/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_27/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_27/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_27/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_27/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_27/kernel*
	container *
shape:
¦
$resnet_model/conv2d_27/kernel/AssignAssignresnet_model/conv2d_27/kernel:resnet_model/conv2d_27/kernel/Initializer/truncated_normal"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_27/kernel*
validate_shape(
Á
"resnet_model/conv2d_27/kernel/readIdentityresnet_model/conv2d_27/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_27/kernel*(
_output_shapes
:

$resnet_model/conv2d_27/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_27/Conv2DConv2Dresnet_model/Relu_23"resnet_model/conv2d_27/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME

resnet_model/add_7Addresnet_model/conv2d_27/Conv2Dresnet_model/conv2d_24/Conv2D"/device:GPU:0*
T0*'
_output_shapes
:@
Ó
Jresnet_model/batch_normalization_24/gamma/Initializer/ones/shape_as_tensorConst*<
_class2
0.loc:@resnet_model/batch_normalization_24/gamma*
valueB:*
dtype0*
_output_shapes
:
Ã
@resnet_model/batch_normalization_24/gamma/Initializer/ones/ConstConst*<
_class2
0.loc:@resnet_model/batch_normalization_24/gamma*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Æ
:resnet_model/batch_normalization_24/gamma/Initializer/onesFillJresnet_model/batch_normalization_24/gamma/Initializer/ones/shape_as_tensor@resnet_model/batch_normalization_24/gamma/Initializer/ones/Const*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_24/gamma*

index_type0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_24/gamma
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_24/gamma*
	container 
½
0resnet_model/batch_normalization_24/gamma/AssignAssign)resnet_model/batch_normalization_24/gamma:resnet_model/batch_normalization_24/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_24/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_24/gamma/readIdentity)resnet_model/batch_normalization_24/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_24/gamma*
_output_shapes	
:
Ò
Jresnet_model/batch_normalization_24/beta/Initializer/zeros/shape_as_tensorConst*;
_class1
/-loc:@resnet_model/batch_normalization_24/beta*
valueB:*
dtype0*
_output_shapes
:
Â
@resnet_model/batch_normalization_24/beta/Initializer/zeros/ConstConst*
_output_shapes
: *;
_class1
/-loc:@resnet_model/batch_normalization_24/beta*
valueB
 *    *
dtype0
Å
:resnet_model/batch_normalization_24/beta/Initializer/zerosFillJresnet_model/batch_normalization_24/beta/Initializer/zeros/shape_as_tensor@resnet_model/batch_normalization_24/beta/Initializer/zeros/Const*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_24/beta*

index_type0
â
(resnet_model/batch_normalization_24/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_24/beta*
	container *
shape:
º
/resnet_model/batch_normalization_24/beta/AssignAssign(resnet_model/batch_normalization_24/beta:resnet_model/batch_normalization_24/beta/Initializer/zeros"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_24/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
Õ
-resnet_model/batch_normalization_24/beta/readIdentity(resnet_model/batch_normalization_24/beta"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_24/beta*
_output_shapes	
:*
T0
à
Qresnet_model/batch_normalization_24/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@resnet_model/batch_normalization_24/moving_mean*
valueB:
Ð
Gresnet_model/batch_normalization_24/moving_mean/Initializer/zeros/ConstConst*B
_class8
64loc:@resnet_model/batch_normalization_24/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
á
Aresnet_model/batch_normalization_24/moving_mean/Initializer/zerosFillQresnet_model/batch_normalization_24/moving_mean/Initializer/zeros/shape_as_tensorGresnet_model/batch_normalization_24/moving_mean/Initializer/zeros/Const*
T0*B
_class8
64loc:@resnet_model/batch_normalization_24/moving_mean*

index_type0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_24/moving_mean
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_24/moving_mean
Ö
6resnet_model/batch_normalization_24/moving_mean/AssignAssign/resnet_model/batch_normalization_24/moving_meanAresnet_model/batch_normalization_24/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_24/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_24/moving_mean/readIdentity/resnet_model/batch_normalization_24/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_24/moving_mean
ç
Tresnet_model/batch_normalization_24/moving_variance/Initializer/ones/shape_as_tensorConst*F
_class<
:8loc:@resnet_model/batch_normalization_24/moving_variance*
valueB:*
dtype0*
_output_shapes
:
×
Jresnet_model/batch_normalization_24/moving_variance/Initializer/ones/ConstConst*
_output_shapes
: *F
_class<
:8loc:@resnet_model/batch_normalization_24/moving_variance*
valueB
 *  ?*
dtype0
î
Dresnet_model/batch_normalization_24/moving_variance/Initializer/onesFillTresnet_model/batch_normalization_24/moving_variance/Initializer/ones/shape_as_tensorJresnet_model/batch_normalization_24/moving_variance/Initializer/ones/Const*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_24/moving_variance*

index_type0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_24/moving_variance
VariableV2"/device:CPU:0*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_24/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
å
:resnet_model/batch_normalization_24/moving_variance/AssignAssign3resnet_model/batch_normalization_24/moving_varianceDresnet_model/batch_normalization_24/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_24/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_24/moving_variance/readIdentity3resnet_model/batch_normalization_24/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_24/moving_variance*
_output_shapes	
:
À
2resnet_model/batch_normalization_24/FusedBatchNormFusedBatchNormresnet_model/add_7.resnet_model/batch_normalization_24/gamma/read-resnet_model/batch_normalization_24/beta/read4resnet_model/batch_normalization_24/moving_mean/read8resnet_model/batch_normalization_24/moving_variance/read"/device:GPU:0*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW
}
)resnet_model/batch_normalization_24/ConstConst"/device:GPU:0*
_output_shapes
: *
valueB
 *d;?*
dtype0

resnet_model/Relu_24Relu2resnet_model/batch_normalization_24/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_28/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_28/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_28/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_28/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_28/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_28/kernel*
valueB
 *   =
ª
Jresnet_model/conv2d_28/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_28/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_28/kernel*
seed2 
¹
>resnet_model/conv2d_28/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_28/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_28/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_28/kernel
§
:resnet_model/conv2d_28/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_28/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_28/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_28/kernel
æ
resnet_model/conv2d_28/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_28/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
¦
$resnet_model/conv2d_28/kernel/AssignAssignresnet_model/conv2d_28/kernel:resnet_model/conv2d_28/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_28/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_28/kernel/readIdentityresnet_model/conv2d_28/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_28/kernel*(
_output_shapes
:

$resnet_model/conv2d_28/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_28/Conv2DConv2Dresnet_model/Relu_24"resnet_model/conv2d_28/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME
Ç
:resnet_model/batch_normalization_25/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*<
_class2
0.loc:@resnet_model/batch_normalization_25/gamma*
valueB*  ?
ä
)resnet_model/batch_normalization_25/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_25/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_25/gamma/AssignAssign)resnet_model/batch_normalization_25/gamma:resnet_model/batch_normalization_25/gamma/Initializer/ones"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_25/gamma
Ø
.resnet_model/batch_normalization_25/gamma/readIdentity)resnet_model/batch_normalization_25/gamma"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_25/gamma*
_output_shapes	
:*
T0
Æ
:resnet_model/batch_normalization_25/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_25/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_25/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_25/beta*
	container *
shape:
º
/resnet_model/batch_normalization_25/beta/AssignAssign(resnet_model/batch_normalization_25/beta:resnet_model/batch_normalization_25/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_25/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_25/beta/readIdentity(resnet_model/batch_normalization_25/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_25/beta
Ô
Aresnet_model/batch_normalization_25/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_25/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_25/moving_mean
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_25/moving_mean*
	container 
Ö
6resnet_model/batch_normalization_25/moving_mean/AssignAssign/resnet_model/batch_normalization_25/moving_meanAresnet_model/batch_normalization_25/moving_mean/Initializer/zeros"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_25/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ê
4resnet_model/batch_normalization_25/moving_mean/readIdentity/resnet_model/batch_normalization_25/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_25/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_25/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_25/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_25/moving_variance
VariableV2"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_25/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
å
:resnet_model/batch_normalization_25/moving_variance/AssignAssign3resnet_model/batch_normalization_25/moving_varianceDresnet_model/batch_normalization_25/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_25/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_25/moving_variance/readIdentity3resnet_model/batch_normalization_25/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_25/moving_variance
Ë
2resnet_model/batch_normalization_25/FusedBatchNormFusedBatchNormresnet_model/conv2d_28/Conv2D.resnet_model/batch_normalization_25/gamma/read-resnet_model/batch_normalization_25/beta/read4resnet_model/batch_normalization_25/moving_mean/read8resnet_model/batch_normalization_25/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7
}
)resnet_model/batch_normalization_25/ConstConst"/device:GPU:0*
_output_shapes
: *
valueB
 *d;?*
dtype0

resnet_model/Relu_25Relu2resnet_model/batch_normalization_25/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_29/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_29/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_29/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_29/kernel*
valueB
 *    
¸
Aresnet_model/conv2d_29/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_29/kernel*
valueB
 *«ªª<*
dtype0
ª
Jresnet_model/conv2d_29/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_29/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_29/kernel*
seed2 
¹
>resnet_model/conv2d_29/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_29/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_29/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_29/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_29/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_29/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_29/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_29/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_29/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_29/kernel*
	container *
shape:
¦
$resnet_model/conv2d_29/kernel/AssignAssignresnet_model/conv2d_29/kernel:resnet_model/conv2d_29/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_29/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_29/kernel/readIdentityresnet_model/conv2d_29/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_29/kernel*(
_output_shapes
:

$resnet_model/conv2d_29/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_29/Conv2DConv2Dresnet_model/Relu_25"resnet_model/conv2d_29/kernel/read"/device:GPU:0*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides

Ç
:resnet_model/batch_normalization_26/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_26/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_26/gamma
VariableV2"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_26/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
½
0resnet_model/batch_normalization_26/gamma/AssignAssign)resnet_model/batch_normalization_26/gamma:resnet_model/batch_normalization_26/gamma/Initializer/ones"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_26/gamma*
validate_shape(
Ø
.resnet_model/batch_normalization_26/gamma/readIdentity)resnet_model/batch_normalization_26/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_26/gamma
Æ
:resnet_model/batch_normalization_26/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_26/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_26/beta
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_26/beta
º
/resnet_model/batch_normalization_26/beta/AssignAssign(resnet_model/batch_normalization_26/beta:resnet_model/batch_normalization_26/beta/Initializer/zeros"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_26/beta*
validate_shape(
Õ
-resnet_model/batch_normalization_26/beta/readIdentity(resnet_model/batch_normalization_26/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_26/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_26/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_26/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_26/moving_mean
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_26/moving_mean*
	container *
shape:
Ö
6resnet_model/batch_normalization_26/moving_mean/AssignAssign/resnet_model/batch_normalization_26/moving_meanAresnet_model/batch_normalization_26/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_26/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_26/moving_mean/readIdentity/resnet_model/batch_normalization_26/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_26/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_26/moving_variance/Initializer/onesConst*
_output_shapes	
:*F
_class<
:8loc:@resnet_model/batch_normalization_26/moving_variance*
valueB*  ?*
dtype0
ø
3resnet_model/batch_normalization_26/moving_variance
VariableV2"/device:CPU:0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_26/moving_variance*
	container *
shape:*
dtype0
å
:resnet_model/batch_normalization_26/moving_variance/AssignAssign3resnet_model/batch_normalization_26/moving_varianceDresnet_model/batch_normalization_26/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_26/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_26/moving_variance/readIdentity3resnet_model/batch_normalization_26/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_26/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_26/FusedBatchNormFusedBatchNormresnet_model/conv2d_29/Conv2D.resnet_model/batch_normalization_26/gamma/read-resnet_model/batch_normalization_26/beta/read4resnet_model/batch_normalization_26/moving_mean/read8resnet_model/batch_normalization_26/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7
}
)resnet_model/batch_normalization_26/ConstConst"/device:GPU:0*
_output_shapes
: *
valueB
 *d;?*
dtype0

resnet_model/Relu_26Relu2resnet_model/batch_normalization_26/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_30/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_30/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_30/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_30/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_30/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_30/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_30/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_30/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@resnet_model/conv2d_30/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed 
¹
>resnet_model/conv2d_30/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_30/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_30/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_30/kernel
§
:resnet_model/conv2d_30/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_30/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_30/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_30/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_30/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_30/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
¦
$resnet_model/conv2d_30/kernel/AssignAssignresnet_model/conv2d_30/kernel:resnet_model/conv2d_30/kernel/Initializer/truncated_normal"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_30/kernel
Á
"resnet_model/conv2d_30/kernel/readIdentityresnet_model/conv2d_30/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_30/kernel*(
_output_shapes
:

$resnet_model/conv2d_30/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_30/Conv2DConv2Dresnet_model/Relu_26"resnet_model/conv2d_30/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME

resnet_model/add_8Addresnet_model/conv2d_30/Conv2Dresnet_model/add_7"/device:GPU:0*
T0*'
_output_shapes
:@
Ó
Jresnet_model/batch_normalization_27/gamma/Initializer/ones/shape_as_tensorConst*<
_class2
0.loc:@resnet_model/batch_normalization_27/gamma*
valueB:*
dtype0*
_output_shapes
:
Ã
@resnet_model/batch_normalization_27/gamma/Initializer/ones/ConstConst*<
_class2
0.loc:@resnet_model/batch_normalization_27/gamma*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Æ
:resnet_model/batch_normalization_27/gamma/Initializer/onesFillJresnet_model/batch_normalization_27/gamma/Initializer/ones/shape_as_tensor@resnet_model/batch_normalization_27/gamma/Initializer/ones/Const*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_27/gamma*

index_type0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_27/gamma
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_27/gamma*
	container 
½
0resnet_model/batch_normalization_27/gamma/AssignAssign)resnet_model/batch_normalization_27/gamma:resnet_model/batch_normalization_27/gamma/Initializer/ones"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_27/gamma
Ø
.resnet_model/batch_normalization_27/gamma/readIdentity)resnet_model/batch_normalization_27/gamma"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_27/gamma*
_output_shapes	
:*
T0
Ò
Jresnet_model/batch_normalization_27/beta/Initializer/zeros/shape_as_tensorConst*;
_class1
/-loc:@resnet_model/batch_normalization_27/beta*
valueB:*
dtype0*
_output_shapes
:
Â
@resnet_model/batch_normalization_27/beta/Initializer/zeros/ConstConst*;
_class1
/-loc:@resnet_model/batch_normalization_27/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
Å
:resnet_model/batch_normalization_27/beta/Initializer/zerosFillJresnet_model/batch_normalization_27/beta/Initializer/zeros/shape_as_tensor@resnet_model/batch_normalization_27/beta/Initializer/zeros/Const*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_27/beta*

index_type0
â
(resnet_model/batch_normalization_27/beta
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_27/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
º
/resnet_model/batch_normalization_27/beta/AssignAssign(resnet_model/batch_normalization_27/beta:resnet_model/batch_normalization_27/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_27/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_27/beta/readIdentity(resnet_model/batch_normalization_27/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_27/beta*
_output_shapes	
:
à
Qresnet_model/batch_normalization_27/moving_mean/Initializer/zeros/shape_as_tensorConst*B
_class8
64loc:@resnet_model/batch_normalization_27/moving_mean*
valueB:*
dtype0*
_output_shapes
:
Ð
Gresnet_model/batch_normalization_27/moving_mean/Initializer/zeros/ConstConst*B
_class8
64loc:@resnet_model/batch_normalization_27/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
á
Aresnet_model/batch_normalization_27/moving_mean/Initializer/zerosFillQresnet_model/batch_normalization_27/moving_mean/Initializer/zeros/shape_as_tensorGresnet_model/batch_normalization_27/moving_mean/Initializer/zeros/Const*
T0*B
_class8
64loc:@resnet_model/batch_normalization_27/moving_mean*

index_type0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_27/moving_mean
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_27/moving_mean*
	container 
Ö
6resnet_model/batch_normalization_27/moving_mean/AssignAssign/resnet_model/batch_normalization_27/moving_meanAresnet_model/batch_normalization_27/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_27/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_27/moving_mean/readIdentity/resnet_model/batch_normalization_27/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_27/moving_mean
ç
Tresnet_model/batch_normalization_27/moving_variance/Initializer/ones/shape_as_tensorConst*F
_class<
:8loc:@resnet_model/batch_normalization_27/moving_variance*
valueB:*
dtype0*
_output_shapes
:
×
Jresnet_model/batch_normalization_27/moving_variance/Initializer/ones/ConstConst*
_output_shapes
: *F
_class<
:8loc:@resnet_model/batch_normalization_27/moving_variance*
valueB
 *  ?*
dtype0
î
Dresnet_model/batch_normalization_27/moving_variance/Initializer/onesFillTresnet_model/batch_normalization_27/moving_variance/Initializer/ones/shape_as_tensorJresnet_model/batch_normalization_27/moving_variance/Initializer/ones/Const*F
_class<
:8loc:@resnet_model/batch_normalization_27/moving_variance*

index_type0*
_output_shapes	
:*
T0
ø
3resnet_model/batch_normalization_27/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_27/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_27/moving_variance/AssignAssign3resnet_model/batch_normalization_27/moving_varianceDresnet_model/batch_normalization_27/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_27/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_27/moving_variance/readIdentity3resnet_model/batch_normalization_27/moving_variance"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_27/moving_variance*
_output_shapes	
:*
T0
À
2resnet_model/batch_normalization_27/FusedBatchNormFusedBatchNormresnet_model/add_8.resnet_model/batch_normalization_27/gamma/read-resnet_model/batch_normalization_27/beta/read4resnet_model/batch_normalization_27/moving_mean/read8resnet_model/batch_normalization_27/moving_variance/read"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0
}
)resnet_model/batch_normalization_27/ConstConst"/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *d;?

resnet_model/Relu_27Relu2resnet_model/batch_normalization_27/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_31/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_31/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_31/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_31/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_31/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_31/kernel*
valueB
 *   =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_31/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_31/kernel/Initializer/truncated_normal/shape*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_31/kernel*
seed2 *
dtype0*(
_output_shapes
:
¹
>resnet_model/conv2d_31/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_31/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_31/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_31/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_31/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_31/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_31/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_31/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_31/kernel
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_31/kernel
¦
$resnet_model/conv2d_31/kernel/AssignAssignresnet_model/conv2d_31/kernel:resnet_model/conv2d_31/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_31/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_31/kernel/readIdentityresnet_model/conv2d_31/kernel"/device:CPU:0*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_31/kernel

$resnet_model/conv2d_31/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_31/Conv2DConv2Dresnet_model/Relu_27"resnet_model/conv2d_31/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME
Ç
:resnet_model/batch_normalization_28/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_28/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_28/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_28/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_28/gamma/AssignAssign)resnet_model/batch_normalization_28/gamma:resnet_model/batch_normalization_28/gamma/Initializer/ones"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_28/gamma
Ø
.resnet_model/batch_normalization_28/gamma/readIdentity)resnet_model/batch_normalization_28/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_28/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_28/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_28/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_28/beta
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_28/beta
º
/resnet_model/batch_normalization_28/beta/AssignAssign(resnet_model/batch_normalization_28/beta:resnet_model/batch_normalization_28/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_28/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_28/beta/readIdentity(resnet_model/batch_normalization_28/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_28/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_28/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_28/moving_mean*
valueB*    
ð
/resnet_model/batch_normalization_28/moving_mean
VariableV2"/device:CPU:0*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_28/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
6resnet_model/batch_normalization_28/moving_mean/AssignAssign/resnet_model/batch_normalization_28/moving_meanAresnet_model/batch_normalization_28/moving_mean/Initializer/zeros"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_28/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ê
4resnet_model/batch_normalization_28/moving_mean/readIdentity/resnet_model/batch_normalization_28/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_28/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_28/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:*F
_class<
:8loc:@resnet_model/batch_normalization_28/moving_variance*
valueB*  ?
ø
3resnet_model/batch_normalization_28/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_28/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_28/moving_variance/AssignAssign3resnet_model/batch_normalization_28/moving_varianceDresnet_model/batch_normalization_28/moving_variance/Initializer/ones"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_28/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
8resnet_model/batch_normalization_28/moving_variance/readIdentity3resnet_model/batch_normalization_28/moving_variance"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_28/moving_variance*
_output_shapes	
:*
T0
Ë
2resnet_model/batch_normalization_28/FusedBatchNormFusedBatchNormresnet_model/conv2d_31/Conv2D.resnet_model/batch_normalization_28/gamma/read-resnet_model/batch_normalization_28/beta/read4resnet_model/batch_normalization_28/moving_mean/read8resnet_model/batch_normalization_28/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_28/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_28Relu2resnet_model/batch_normalization_28/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_32/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_32/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_32/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_32/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_32/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_32/kernel*
valueB
 *«ªª<
ª
Jresnet_model/conv2d_32/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_32/kernel/Initializer/truncated_normal/shape*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_32/kernel*
seed2 *
dtype0*(
_output_shapes
:
¹
>resnet_model/conv2d_32/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_32/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_32/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_32/kernel
§
:resnet_model/conv2d_32/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_32/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_32/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_32/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_32/kernel
VariableV2"/device:CPU:0*
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_32/kernel*
	container 
¦
$resnet_model/conv2d_32/kernel/AssignAssignresnet_model/conv2d_32/kernel:resnet_model/conv2d_32/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_32/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_32/kernel/readIdentityresnet_model/conv2d_32/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_32/kernel*(
_output_shapes
:

$resnet_model/conv2d_32/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_32/Conv2DConv2Dresnet_model/Relu_28"resnet_model/conv2d_32/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME
Ç
:resnet_model/batch_normalization_29/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_29/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_29/gamma
VariableV2"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_29/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
½
0resnet_model/batch_normalization_29/gamma/AssignAssign)resnet_model/batch_normalization_29/gamma:resnet_model/batch_normalization_29/gamma/Initializer/ones"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_29/gamma*
validate_shape(
Ø
.resnet_model/batch_normalization_29/gamma/readIdentity)resnet_model/batch_normalization_29/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_29/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_29/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*;
_class1
/-loc:@resnet_model/batch_normalization_29/beta*
valueB*    
â
(resnet_model/batch_normalization_29/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_29/beta*
	container *
shape:
º
/resnet_model/batch_normalization_29/beta/AssignAssign(resnet_model/batch_normalization_29/beta:resnet_model/batch_normalization_29/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_29/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_29/beta/readIdentity(resnet_model/batch_normalization_29/beta"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_29/beta*
_output_shapes	
:*
T0
Ô
Aresnet_model/batch_normalization_29/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_29/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_29/moving_mean
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_29/moving_mean
Ö
6resnet_model/batch_normalization_29/moving_mean/AssignAssign/resnet_model/batch_normalization_29/moving_meanAresnet_model/batch_normalization_29/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_29/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_29/moving_mean/readIdentity/resnet_model/batch_normalization_29/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_29/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_29/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_29/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_29/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_29/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_29/moving_variance/AssignAssign3resnet_model/batch_normalization_29/moving_varianceDresnet_model/batch_normalization_29/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_29/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_29/moving_variance/readIdentity3resnet_model/batch_normalization_29/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_29/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_29/FusedBatchNormFusedBatchNormresnet_model/conv2d_32/Conv2D.resnet_model/batch_normalization_29/gamma/read-resnet_model/batch_normalization_29/beta/read4resnet_model/batch_normalization_29/moving_mean/read8resnet_model/batch_normalization_29/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_29/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_29Relu2resnet_model/batch_normalization_29/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_33/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_33/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_33/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_33/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_33/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_33/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_33/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_33/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@resnet_model/conv2d_33/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed 
¹
>resnet_model/conv2d_33/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_33/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_33/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_33/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_33/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_33/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_33/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_33/kernel
æ
resnet_model/conv2d_33/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_33/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
¦
$resnet_model/conv2d_33/kernel/AssignAssignresnet_model/conv2d_33/kernel:resnet_model/conv2d_33/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_33/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_33/kernel/readIdentityresnet_model/conv2d_33/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_33/kernel*(
_output_shapes
:

$resnet_model/conv2d_33/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_33/Conv2DConv2Dresnet_model/Relu_29"resnet_model/conv2d_33/kernel/read"/device:GPU:0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0

resnet_model/add_9Addresnet_model/conv2d_33/Conv2Dresnet_model/add_8"/device:GPU:0*
T0*'
_output_shapes
:@
Ó
Jresnet_model/batch_normalization_30/gamma/Initializer/ones/shape_as_tensorConst*<
_class2
0.loc:@resnet_model/batch_normalization_30/gamma*
valueB:*
dtype0*
_output_shapes
:
Ã
@resnet_model/batch_normalization_30/gamma/Initializer/ones/ConstConst*<
_class2
0.loc:@resnet_model/batch_normalization_30/gamma*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Æ
:resnet_model/batch_normalization_30/gamma/Initializer/onesFillJresnet_model/batch_normalization_30/gamma/Initializer/ones/shape_as_tensor@resnet_model/batch_normalization_30/gamma/Initializer/ones/Const*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_30/gamma*

index_type0
ä
)resnet_model/batch_normalization_30/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_30/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_30/gamma/AssignAssign)resnet_model/batch_normalization_30/gamma:resnet_model/batch_normalization_30/gamma/Initializer/ones"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_30/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ø
.resnet_model/batch_normalization_30/gamma/readIdentity)resnet_model/batch_normalization_30/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_30/gamma
Ò
Jresnet_model/batch_normalization_30/beta/Initializer/zeros/shape_as_tensorConst*;
_class1
/-loc:@resnet_model/batch_normalization_30/beta*
valueB:*
dtype0*
_output_shapes
:
Â
@resnet_model/batch_normalization_30/beta/Initializer/zeros/ConstConst*;
_class1
/-loc:@resnet_model/batch_normalization_30/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
Å
:resnet_model/batch_normalization_30/beta/Initializer/zerosFillJresnet_model/batch_normalization_30/beta/Initializer/zeros/shape_as_tensor@resnet_model/batch_normalization_30/beta/Initializer/zeros/Const*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_30/beta*

index_type0
â
(resnet_model/batch_normalization_30/beta
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_30/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
º
/resnet_model/batch_normalization_30/beta/AssignAssign(resnet_model/batch_normalization_30/beta:resnet_model/batch_normalization_30/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_30/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_30/beta/readIdentity(resnet_model/batch_normalization_30/beta"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_30/beta*
_output_shapes	
:*
T0
à
Qresnet_model/batch_normalization_30/moving_mean/Initializer/zeros/shape_as_tensorConst*B
_class8
64loc:@resnet_model/batch_normalization_30/moving_mean*
valueB:*
dtype0*
_output_shapes
:
Ð
Gresnet_model/batch_normalization_30/moving_mean/Initializer/zeros/ConstConst*B
_class8
64loc:@resnet_model/batch_normalization_30/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
á
Aresnet_model/batch_normalization_30/moving_mean/Initializer/zerosFillQresnet_model/batch_normalization_30/moving_mean/Initializer/zeros/shape_as_tensorGresnet_model/batch_normalization_30/moving_mean/Initializer/zeros/Const*
T0*B
_class8
64loc:@resnet_model/batch_normalization_30/moving_mean*

index_type0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_30/moving_mean
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_30/moving_mean*
	container 
Ö
6resnet_model/batch_normalization_30/moving_mean/AssignAssign/resnet_model/batch_normalization_30/moving_meanAresnet_model/batch_normalization_30/moving_mean/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_30/moving_mean
ê
4resnet_model/batch_normalization_30/moving_mean/readIdentity/resnet_model/batch_normalization_30/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_30/moving_mean*
_output_shapes	
:
ç
Tresnet_model/batch_normalization_30/moving_variance/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*F
_class<
:8loc:@resnet_model/batch_normalization_30/moving_variance*
valueB:*
dtype0
×
Jresnet_model/batch_normalization_30/moving_variance/Initializer/ones/ConstConst*F
_class<
:8loc:@resnet_model/batch_normalization_30/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 
î
Dresnet_model/batch_normalization_30/moving_variance/Initializer/onesFillTresnet_model/batch_normalization_30/moving_variance/Initializer/ones/shape_as_tensorJresnet_model/batch_normalization_30/moving_variance/Initializer/ones/Const*F
_class<
:8loc:@resnet_model/batch_normalization_30/moving_variance*

index_type0*
_output_shapes	
:*
T0
ø
3resnet_model/batch_normalization_30/moving_variance
VariableV2"/device:CPU:0*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_30/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
å
:resnet_model/batch_normalization_30/moving_variance/AssignAssign3resnet_model/batch_normalization_30/moving_varianceDresnet_model/batch_normalization_30/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_30/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_30/moving_variance/readIdentity3resnet_model/batch_normalization_30/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_30/moving_variance*
_output_shapes	
:
À
2resnet_model/batch_normalization_30/FusedBatchNormFusedBatchNormresnet_model/add_9.resnet_model/batch_normalization_30/gamma/read-resnet_model/batch_normalization_30/beta/read4resnet_model/batch_normalization_30/moving_mean/read8resnet_model/batch_normalization_30/moving_variance/read"/device:GPU:0*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW
}
)resnet_model/batch_normalization_30/ConstConst"/device:GPU:0*
_output_shapes
: *
valueB
 *d;?*
dtype0

resnet_model/Relu_30Relu2resnet_model/batch_normalization_30/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_34/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_34/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_34/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_34/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_34/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_34/kernel*
valueB
 *   =*
dtype0
ª
Jresnet_model/conv2d_34/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_34/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@resnet_model/conv2d_34/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed 
¹
>resnet_model/conv2d_34/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_34/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_34/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_34/kernel
§
:resnet_model/conv2d_34/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_34/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_34/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_34/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_34/kernel
VariableV2"/device:CPU:0*
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_34/kernel*
	container 
¦
$resnet_model/conv2d_34/kernel/AssignAssignresnet_model/conv2d_34/kernel:resnet_model/conv2d_34/kernel/Initializer/truncated_normal"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_34/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Á
"resnet_model/conv2d_34/kernel/readIdentityresnet_model/conv2d_34/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_34/kernel*(
_output_shapes
:

$resnet_model/conv2d_34/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_34/Conv2DConv2Dresnet_model/Relu_30"resnet_model/conv2d_34/kernel/read"/device:GPU:0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0
Ç
:resnet_model/batch_normalization_31/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_31/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_31/gamma
VariableV2"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_31/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
½
0resnet_model/batch_normalization_31/gamma/AssignAssign)resnet_model/batch_normalization_31/gamma:resnet_model/batch_normalization_31/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_31/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_31/gamma/readIdentity)resnet_model/batch_normalization_31/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_31/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_31/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_31/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_31/beta
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_31/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
º
/resnet_model/batch_normalization_31/beta/AssignAssign(resnet_model/batch_normalization_31/beta:resnet_model/batch_normalization_31/beta/Initializer/zeros"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_31/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
Õ
-resnet_model/batch_normalization_31/beta/readIdentity(resnet_model/batch_normalization_31/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_31/beta
Ô
Aresnet_model/batch_normalization_31/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_31/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_31/moving_mean
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_31/moving_mean*
	container *
shape:
Ö
6resnet_model/batch_normalization_31/moving_mean/AssignAssign/resnet_model/batch_normalization_31/moving_meanAresnet_model/batch_normalization_31/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_31/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_31/moving_mean/readIdentity/resnet_model/batch_normalization_31/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_31/moving_mean
Û
Dresnet_model/batch_normalization_31/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_31/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_31/moving_variance
VariableV2"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_31/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
å
:resnet_model/batch_normalization_31/moving_variance/AssignAssign3resnet_model/batch_normalization_31/moving_varianceDresnet_model/batch_normalization_31/moving_variance/Initializer/ones"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_31/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
8resnet_model/batch_normalization_31/moving_variance/readIdentity3resnet_model/batch_normalization_31/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_31/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_31/FusedBatchNormFusedBatchNormresnet_model/conv2d_34/Conv2D.resnet_model/batch_normalization_31/gamma/read-resnet_model/batch_normalization_31/beta/read4resnet_model/batch_normalization_31/moving_mean/read8resnet_model/batch_normalization_31/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7
}
)resnet_model/batch_normalization_31/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_31Relu2resnet_model/batch_normalization_31/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_35/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_35/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_35/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_35/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_35/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_35/kernel*
valueB
 *«ªª<*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_35/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_35/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@resnet_model/conv2d_35/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed 
¹
>resnet_model/conv2d_35/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_35/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_35/kernel/Initializer/truncated_normal/stddev*0
_class&
$"loc:@resnet_model/conv2d_35/kernel*(
_output_shapes
:*
T0
§
:resnet_model/conv2d_35/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_35/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_35/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_35/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_35/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_35/kernel*
	container *
shape:
¦
$resnet_model/conv2d_35/kernel/AssignAssignresnet_model/conv2d_35/kernel:resnet_model/conv2d_35/kernel/Initializer/truncated_normal"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_35/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
Á
"resnet_model/conv2d_35/kernel/readIdentityresnet_model/conv2d_35/kernel"/device:CPU:0*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_35/kernel

$resnet_model/conv2d_35/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_35/Conv2DConv2Dresnet_model/Relu_31"resnet_model/conv2d_35/kernel/read"/device:GPU:0*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations

Ç
:resnet_model/batch_normalization_32/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_32/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_32/gamma
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_32/gamma*
	container 
½
0resnet_model/batch_normalization_32/gamma/AssignAssign)resnet_model/batch_normalization_32/gamma:resnet_model/batch_normalization_32/gamma/Initializer/ones"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_32/gamma
Ø
.resnet_model/batch_normalization_32/gamma/readIdentity)resnet_model/batch_normalization_32/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_32/gamma
Æ
:resnet_model/batch_normalization_32/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_32/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_32/beta
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_32/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
º
/resnet_model/batch_normalization_32/beta/AssignAssign(resnet_model/batch_normalization_32/beta:resnet_model/batch_normalization_32/beta/Initializer/zeros"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_32/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Õ
-resnet_model/batch_normalization_32/beta/readIdentity(resnet_model/batch_normalization_32/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_32/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_32/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_32/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_32/moving_mean
VariableV2"/device:CPU:0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_32/moving_mean*
	container *
shape:*
dtype0
Ö
6resnet_model/batch_normalization_32/moving_mean/AssignAssign/resnet_model/batch_normalization_32/moving_meanAresnet_model/batch_normalization_32/moving_mean/Initializer/zeros"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_32/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ê
4resnet_model/batch_normalization_32/moving_mean/readIdentity/resnet_model/batch_normalization_32/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_32/moving_mean
Û
Dresnet_model/batch_normalization_32/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_32/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_32/moving_variance
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_32/moving_variance
å
:resnet_model/batch_normalization_32/moving_variance/AssignAssign3resnet_model/batch_normalization_32/moving_varianceDresnet_model/batch_normalization_32/moving_variance/Initializer/ones"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_32/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
8resnet_model/batch_normalization_32/moving_variance/readIdentity3resnet_model/batch_normalization_32/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_32/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_32/FusedBatchNormFusedBatchNormresnet_model/conv2d_35/Conv2D.resnet_model/batch_normalization_32/gamma/read-resnet_model/batch_normalization_32/beta/read4resnet_model/batch_normalization_32/moving_mean/read8resnet_model/batch_normalization_32/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_32/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_32Relu2resnet_model/batch_normalization_32/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_36/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_36/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_36/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_36/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_36/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_36/kernel*
valueB
 *  =
ª
Jresnet_model/conv2d_36/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_36/kernel/Initializer/truncated_normal/shape*0
_class&
$"loc:@resnet_model/conv2d_36/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0
¹
>resnet_model/conv2d_36/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_36/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_36/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_36/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_36/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_36/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_36/kernel/Initializer/truncated_normal/mean*0
_class&
$"loc:@resnet_model/conv2d_36/kernel*(
_output_shapes
:*
T0
æ
resnet_model/conv2d_36/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_36/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
¦
$resnet_model/conv2d_36/kernel/AssignAssignresnet_model/conv2d_36/kernel:resnet_model/conv2d_36/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_36/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_36/kernel/readIdentityresnet_model/conv2d_36/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_36/kernel*(
_output_shapes
:

$resnet_model/conv2d_36/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_36/Conv2DConv2Dresnet_model/Relu_32"resnet_model/conv2d_36/kernel/read"/device:GPU:0*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@

resnet_model/add_10Addresnet_model/conv2d_36/Conv2Dresnet_model/add_9"/device:GPU:0*'
_output_shapes
:@*
T0
Ó
Jresnet_model/batch_normalization_33/gamma/Initializer/ones/shape_as_tensorConst*<
_class2
0.loc:@resnet_model/batch_normalization_33/gamma*
valueB:*
dtype0*
_output_shapes
:
Ã
@resnet_model/batch_normalization_33/gamma/Initializer/ones/ConstConst*<
_class2
0.loc:@resnet_model/batch_normalization_33/gamma*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Æ
:resnet_model/batch_normalization_33/gamma/Initializer/onesFillJresnet_model/batch_normalization_33/gamma/Initializer/ones/shape_as_tensor@resnet_model/batch_normalization_33/gamma/Initializer/ones/Const*<
_class2
0.loc:@resnet_model/batch_normalization_33/gamma*

index_type0*
_output_shapes	
:*
T0
ä
)resnet_model/batch_normalization_33/gamma
VariableV2"/device:CPU:0*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_33/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
½
0resnet_model/batch_normalization_33/gamma/AssignAssign)resnet_model/batch_normalization_33/gamma:resnet_model/batch_normalization_33/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_33/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_33/gamma/readIdentity)resnet_model/batch_normalization_33/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_33/gamma*
_output_shapes	
:
Ò
Jresnet_model/batch_normalization_33/beta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*;
_class1
/-loc:@resnet_model/batch_normalization_33/beta*
valueB:*
dtype0
Â
@resnet_model/batch_normalization_33/beta/Initializer/zeros/ConstConst*;
_class1
/-loc:@resnet_model/batch_normalization_33/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
Å
:resnet_model/batch_normalization_33/beta/Initializer/zerosFillJresnet_model/batch_normalization_33/beta/Initializer/zeros/shape_as_tensor@resnet_model/batch_normalization_33/beta/Initializer/zeros/Const*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_33/beta*

index_type0*
_output_shapes	
:
â
(resnet_model/batch_normalization_33/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_33/beta*
	container *
shape:
º
/resnet_model/batch_normalization_33/beta/AssignAssign(resnet_model/batch_normalization_33/beta:resnet_model/batch_normalization_33/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_33/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_33/beta/readIdentity(resnet_model/batch_normalization_33/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_33/beta
à
Qresnet_model/batch_normalization_33/moving_mean/Initializer/zeros/shape_as_tensorConst*B
_class8
64loc:@resnet_model/batch_normalization_33/moving_mean*
valueB:*
dtype0*
_output_shapes
:
Ð
Gresnet_model/batch_normalization_33/moving_mean/Initializer/zeros/ConstConst*B
_class8
64loc:@resnet_model/batch_normalization_33/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
á
Aresnet_model/batch_normalization_33/moving_mean/Initializer/zerosFillQresnet_model/batch_normalization_33/moving_mean/Initializer/zeros/shape_as_tensorGresnet_model/batch_normalization_33/moving_mean/Initializer/zeros/Const*
T0*B
_class8
64loc:@resnet_model/batch_normalization_33/moving_mean*

index_type0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_33/moving_mean
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_33/moving_mean*
	container *
shape:
Ö
6resnet_model/batch_normalization_33/moving_mean/AssignAssign/resnet_model/batch_normalization_33/moving_meanAresnet_model/batch_normalization_33/moving_mean/Initializer/zeros"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_33/moving_mean*
validate_shape(
ê
4resnet_model/batch_normalization_33/moving_mean/readIdentity/resnet_model/batch_normalization_33/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_33/moving_mean
ç
Tresnet_model/batch_normalization_33/moving_variance/Initializer/ones/shape_as_tensorConst*F
_class<
:8loc:@resnet_model/batch_normalization_33/moving_variance*
valueB:*
dtype0*
_output_shapes
:
×
Jresnet_model/batch_normalization_33/moving_variance/Initializer/ones/ConstConst*F
_class<
:8loc:@resnet_model/batch_normalization_33/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 
î
Dresnet_model/batch_normalization_33/moving_variance/Initializer/onesFillTresnet_model/batch_normalization_33/moving_variance/Initializer/ones/shape_as_tensorJresnet_model/batch_normalization_33/moving_variance/Initializer/ones/Const*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_33/moving_variance*

index_type0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_33/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_33/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_33/moving_variance/AssignAssign3resnet_model/batch_normalization_33/moving_varianceDresnet_model/batch_normalization_33/moving_variance/Initializer/ones"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_33/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
8resnet_model/batch_normalization_33/moving_variance/readIdentity3resnet_model/batch_normalization_33/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_33/moving_variance*
_output_shapes	
:
Á
2resnet_model/batch_normalization_33/FusedBatchNormFusedBatchNormresnet_model/add_10.resnet_model/batch_normalization_33/gamma/read-resnet_model/batch_normalization_33/beta/read4resnet_model/batch_normalization_33/moving_mean/read8resnet_model/batch_normalization_33/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7
}
)resnet_model/batch_normalization_33/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_33Relu2resnet_model/batch_normalization_33/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_37/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_37/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_37/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_37/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_37/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_37/kernel*
valueB
 *   =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_37/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_37/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@resnet_model/conv2d_37/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed 
¹
>resnet_model/conv2d_37/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_37/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_37/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_37/kernel
§
:resnet_model/conv2d_37/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_37/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_37/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_37/kernel
æ
resnet_model/conv2d_37/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_37/kernel*
	container *
shape:
¦
$resnet_model/conv2d_37/kernel/AssignAssignresnet_model/conv2d_37/kernel:resnet_model/conv2d_37/kernel/Initializer/truncated_normal"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_37/kernel
Á
"resnet_model/conv2d_37/kernel/readIdentityresnet_model/conv2d_37/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_37/kernel*(
_output_shapes
:

$resnet_model/conv2d_37/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_37/Conv2DConv2Dresnet_model/Relu_33"resnet_model/conv2d_37/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ç
:resnet_model/batch_normalization_34/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*<
_class2
0.loc:@resnet_model/batch_normalization_34/gamma*
valueB*  ?
ä
)resnet_model/batch_normalization_34/gamma
VariableV2"/device:CPU:0*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_34/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
½
0resnet_model/batch_normalization_34/gamma/AssignAssign)resnet_model/batch_normalization_34/gamma:resnet_model/batch_normalization_34/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_34/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_34/gamma/readIdentity)resnet_model/batch_normalization_34/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_34/gamma
Æ
:resnet_model/batch_normalization_34/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_34/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_34/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_34/beta*
	container *
shape:
º
/resnet_model/batch_normalization_34/beta/AssignAssign(resnet_model/batch_normalization_34/beta:resnet_model/batch_normalization_34/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_34/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_34/beta/readIdentity(resnet_model/batch_normalization_34/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_34/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_34/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_34/moving_mean*
valueB*    
ð
/resnet_model/batch_normalization_34/moving_mean
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_34/moving_mean*
	container *
shape:
Ö
6resnet_model/batch_normalization_34/moving_mean/AssignAssign/resnet_model/batch_normalization_34/moving_meanAresnet_model/batch_normalization_34/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_34/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_34/moving_mean/readIdentity/resnet_model/batch_normalization_34/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_34/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_34/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_34/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_34/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_34/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_34/moving_variance/AssignAssign3resnet_model/batch_normalization_34/moving_varianceDresnet_model/batch_normalization_34/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_34/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_34/moving_variance/readIdentity3resnet_model/batch_normalization_34/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_34/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_34/FusedBatchNormFusedBatchNormresnet_model/conv2d_37/Conv2D.resnet_model/batch_normalization_34/gamma/read-resnet_model/batch_normalization_34/beta/read4resnet_model/batch_normalization_34/moving_mean/read8resnet_model/batch_normalization_34/moving_variance/read"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0
}
)resnet_model/batch_normalization_34/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_34Relu2resnet_model/batch_normalization_34/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_38/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_38/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_38/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_38/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_38/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_38/kernel*
valueB
 *«ªª<*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_38/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_38/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_38/kernel*
seed2 
¹
>resnet_model/conv2d_38/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_38/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_38/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_38/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_38/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_38/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_38/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_38/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_38/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_38/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
¦
$resnet_model/conv2d_38/kernel/AssignAssignresnet_model/conv2d_38/kernel:resnet_model/conv2d_38/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_38/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_38/kernel/readIdentityresnet_model/conv2d_38/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_38/kernel*(
_output_shapes
:

$resnet_model/conv2d_38/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_38/Conv2DConv2Dresnet_model/Relu_34"resnet_model/conv2d_38/kernel/read"/device:GPU:0*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides

Ç
:resnet_model/batch_normalization_35/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_35/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_35/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_35/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_35/gamma/AssignAssign)resnet_model/batch_normalization_35/gamma:resnet_model/batch_normalization_35/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_35/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_35/gamma/readIdentity)resnet_model/batch_normalization_35/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_35/gamma
Æ
:resnet_model/batch_normalization_35/beta/Initializer/zerosConst*
_output_shapes	
:*;
_class1
/-loc:@resnet_model/batch_normalization_35/beta*
valueB*    *
dtype0
â
(resnet_model/batch_normalization_35/beta
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_35/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
º
/resnet_model/batch_normalization_35/beta/AssignAssign(resnet_model/batch_normalization_35/beta:resnet_model/batch_normalization_35/beta/Initializer/zeros"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_35/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Õ
-resnet_model/batch_normalization_35/beta/readIdentity(resnet_model/batch_normalization_35/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_35/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_35/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_35/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_35/moving_mean
VariableV2"/device:CPU:0*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_35/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
6resnet_model/batch_normalization_35/moving_mean/AssignAssign/resnet_model/batch_normalization_35/moving_meanAresnet_model/batch_normalization_35/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_35/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_35/moving_mean/readIdentity/resnet_model/batch_normalization_35/moving_mean"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_35/moving_mean*
_output_shapes	
:*
T0
Û
Dresnet_model/batch_normalization_35/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_35/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_35/moving_variance
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_35/moving_variance*
	container 
å
:resnet_model/batch_normalization_35/moving_variance/AssignAssign3resnet_model/batch_normalization_35/moving_varianceDresnet_model/batch_normalization_35/moving_variance/Initializer/ones"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_35/moving_variance
ö
8resnet_model/batch_normalization_35/moving_variance/readIdentity3resnet_model/batch_normalization_35/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_35/moving_variance
Ë
2resnet_model/batch_normalization_35/FusedBatchNormFusedBatchNormresnet_model/conv2d_38/Conv2D.resnet_model/batch_normalization_35/gamma/read-resnet_model/batch_normalization_35/beta/read4resnet_model/batch_normalization_35/moving_mean/read8resnet_model/batch_normalization_35/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_35/ConstConst"/device:GPU:0*
_output_shapes
: *
valueB
 *d;?*
dtype0

resnet_model/Relu_35Relu2resnet_model/batch_normalization_35/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_39/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*0
_class&
$"loc:@resnet_model/conv2d_39/kernel*%
valueB"            *
dtype0
¶
?resnet_model/conv2d_39/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_39/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_39/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_39/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_39/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_39/kernel/Initializer/truncated_normal/shape*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_39/kernel*
seed2 *
dtype0*(
_output_shapes
:
¹
>resnet_model/conv2d_39/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_39/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_39/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_39/kernel
§
:resnet_model/conv2d_39/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_39/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_39/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_39/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_39/kernel
VariableV2"/device:CPU:0*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_39/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:
¦
$resnet_model/conv2d_39/kernel/AssignAssignresnet_model/conv2d_39/kernel:resnet_model/conv2d_39/kernel/Initializer/truncated_normal"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_39/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
Á
"resnet_model/conv2d_39/kernel/readIdentityresnet_model/conv2d_39/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_39/kernel*(
_output_shapes
:

$resnet_model/conv2d_39/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_39/Conv2DConv2Dresnet_model/Relu_35"resnet_model/conv2d_39/kernel/read"/device:GPU:0*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations


resnet_model/add_11Addresnet_model/conv2d_39/Conv2Dresnet_model/add_10"/device:GPU:0*'
_output_shapes
:@*
T0
Ó
Jresnet_model/batch_normalization_36/gamma/Initializer/ones/shape_as_tensorConst*<
_class2
0.loc:@resnet_model/batch_normalization_36/gamma*
valueB:*
dtype0*
_output_shapes
:
Ã
@resnet_model/batch_normalization_36/gamma/Initializer/ones/ConstConst*<
_class2
0.loc:@resnet_model/batch_normalization_36/gamma*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Æ
:resnet_model/batch_normalization_36/gamma/Initializer/onesFillJresnet_model/batch_normalization_36/gamma/Initializer/ones/shape_as_tensor@resnet_model/batch_normalization_36/gamma/Initializer/ones/Const*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_36/gamma*

index_type0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_36/gamma
VariableV2"/device:CPU:0*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_36/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
½
0resnet_model/batch_normalization_36/gamma/AssignAssign)resnet_model/batch_normalization_36/gamma:resnet_model/batch_normalization_36/gamma/Initializer/ones"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_36/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ø
.resnet_model/batch_normalization_36/gamma/readIdentity)resnet_model/batch_normalization_36/gamma"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_36/gamma*
_output_shapes	
:*
T0
Ò
Jresnet_model/batch_normalization_36/beta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*;
_class1
/-loc:@resnet_model/batch_normalization_36/beta*
valueB:*
dtype0
Â
@resnet_model/batch_normalization_36/beta/Initializer/zeros/ConstConst*;
_class1
/-loc:@resnet_model/batch_normalization_36/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
Å
:resnet_model/batch_normalization_36/beta/Initializer/zerosFillJresnet_model/batch_normalization_36/beta/Initializer/zeros/shape_as_tensor@resnet_model/batch_normalization_36/beta/Initializer/zeros/Const*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_36/beta*

index_type0
â
(resnet_model/batch_normalization_36/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_36/beta*
	container *
shape:
º
/resnet_model/batch_normalization_36/beta/AssignAssign(resnet_model/batch_normalization_36/beta:resnet_model/batch_normalization_36/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_36/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_36/beta/readIdentity(resnet_model/batch_normalization_36/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_36/beta*
_output_shapes	
:
à
Qresnet_model/batch_normalization_36/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@resnet_model/batch_normalization_36/moving_mean*
valueB:
Ð
Gresnet_model/batch_normalization_36/moving_mean/Initializer/zeros/ConstConst*
_output_shapes
: *B
_class8
64loc:@resnet_model/batch_normalization_36/moving_mean*
valueB
 *    *
dtype0
á
Aresnet_model/batch_normalization_36/moving_mean/Initializer/zerosFillQresnet_model/batch_normalization_36/moving_mean/Initializer/zeros/shape_as_tensorGresnet_model/batch_normalization_36/moving_mean/Initializer/zeros/Const*
T0*B
_class8
64loc:@resnet_model/batch_normalization_36/moving_mean*

index_type0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_36/moving_mean
VariableV2"/device:CPU:0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_36/moving_mean*
	container *
shape:*
dtype0
Ö
6resnet_model/batch_normalization_36/moving_mean/AssignAssign/resnet_model/batch_normalization_36/moving_meanAresnet_model/batch_normalization_36/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_36/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_36/moving_mean/readIdentity/resnet_model/batch_normalization_36/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_36/moving_mean
ç
Tresnet_model/batch_normalization_36/moving_variance/Initializer/ones/shape_as_tensorConst*F
_class<
:8loc:@resnet_model/batch_normalization_36/moving_variance*
valueB:*
dtype0*
_output_shapes
:
×
Jresnet_model/batch_normalization_36/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@resnet_model/batch_normalization_36/moving_variance*
valueB
 *  ?
î
Dresnet_model/batch_normalization_36/moving_variance/Initializer/onesFillTresnet_model/batch_normalization_36/moving_variance/Initializer/ones/shape_as_tensorJresnet_model/batch_normalization_36/moving_variance/Initializer/ones/Const*F
_class<
:8loc:@resnet_model/batch_normalization_36/moving_variance*

index_type0*
_output_shapes	
:*
T0
ø
3resnet_model/batch_normalization_36/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_36/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_36/moving_variance/AssignAssign3resnet_model/batch_normalization_36/moving_varianceDresnet_model/batch_normalization_36/moving_variance/Initializer/ones"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_36/moving_variance*
validate_shape(
ö
8resnet_model/batch_normalization_36/moving_variance/readIdentity3resnet_model/batch_normalization_36/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_36/moving_variance
Á
2resnet_model/batch_normalization_36/FusedBatchNormFusedBatchNormresnet_model/add_11.resnet_model/batch_normalization_36/gamma/read-resnet_model/batch_normalization_36/beta/read4resnet_model/batch_normalization_36/moving_mean/read8resnet_model/batch_normalization_36/moving_variance/read"/device:GPU:0*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW
}
)resnet_model/batch_normalization_36/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_36Relu2resnet_model/batch_normalization_36/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_40/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*0
_class&
$"loc:@resnet_model/conv2d_40/kernel*%
valueB"            *
dtype0
¶
?resnet_model/conv2d_40/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_40/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_40/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_40/kernel*
valueB
 *   =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_40/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_40/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_40/kernel*
seed2 
¹
>resnet_model/conv2d_40/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_40/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_40/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_40/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_40/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_40/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_40/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_40/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_40/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_40/kernel*
	container *
shape:
¦
$resnet_model/conv2d_40/kernel/AssignAssignresnet_model/conv2d_40/kernel:resnet_model/conv2d_40/kernel/Initializer/truncated_normal"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_40/kernel*
validate_shape(
Á
"resnet_model/conv2d_40/kernel/readIdentityresnet_model/conv2d_40/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_40/kernel*(
_output_shapes
:

$resnet_model/conv2d_40/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_40/Conv2DConv2Dresnet_model/Relu_36"resnet_model/conv2d_40/kernel/read"/device:GPU:0*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Ç
:resnet_model/batch_normalization_37/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_37/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_37/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_37/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_37/gamma/AssignAssign)resnet_model/batch_normalization_37/gamma:resnet_model/batch_normalization_37/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_37/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_37/gamma/readIdentity)resnet_model/batch_normalization_37/gamma"/device:CPU:0*
_output_shapes	
:*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_37/gamma
Æ
:resnet_model/batch_normalization_37/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_37/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_37/beta
VariableV2"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_37/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
º
/resnet_model/batch_normalization_37/beta/AssignAssign(resnet_model/batch_normalization_37/beta:resnet_model/batch_normalization_37/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_37/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_37/beta/readIdentity(resnet_model/batch_normalization_37/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_37/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_37/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_37/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_37/moving_mean
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_37/moving_mean*
	container 
Ö
6resnet_model/batch_normalization_37/moving_mean/AssignAssign/resnet_model/batch_normalization_37/moving_meanAresnet_model/batch_normalization_37/moving_mean/Initializer/zeros"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_37/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ê
4resnet_model/batch_normalization_37/moving_mean/readIdentity/resnet_model/batch_normalization_37/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_37/moving_mean
Û
Dresnet_model/batch_normalization_37/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_37/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_37/moving_variance
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_37/moving_variance*
	container 
å
:resnet_model/batch_normalization_37/moving_variance/AssignAssign3resnet_model/batch_normalization_37/moving_varianceDresnet_model/batch_normalization_37/moving_variance/Initializer/ones"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_37/moving_variance
ö
8resnet_model/batch_normalization_37/moving_variance/readIdentity3resnet_model/batch_normalization_37/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_37/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_37/FusedBatchNormFusedBatchNormresnet_model/conv2d_40/Conv2D.resnet_model/batch_normalization_37/gamma/read-resnet_model/batch_normalization_37/beta/read4resnet_model/batch_normalization_37/moving_mean/read8resnet_model/batch_normalization_37/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7
}
)resnet_model/batch_normalization_37/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_37Relu2resnet_model/batch_normalization_37/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_41/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_41/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_41/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_41/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_41/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_41/kernel*
valueB
 *«ªª<
ª
Jresnet_model/conv2d_41/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_41/kernel/Initializer/truncated_normal/shape*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_41/kernel*
seed2 *
dtype0*(
_output_shapes
:
¹
>resnet_model/conv2d_41/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_41/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_41/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_41/kernel
§
:resnet_model/conv2d_41/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_41/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_41/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_41/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_41/kernel
VariableV2"/device:CPU:0*
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_41/kernel*
	container 
¦
$resnet_model/conv2d_41/kernel/AssignAssignresnet_model/conv2d_41/kernel:resnet_model/conv2d_41/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_41/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_41/kernel/readIdentityresnet_model/conv2d_41/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_41/kernel*(
_output_shapes
:

$resnet_model/conv2d_41/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_41/Conv2DConv2Dresnet_model/Relu_37"resnet_model/conv2d_41/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Ç
:resnet_model/batch_normalization_38/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_38/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_38/gamma
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_38/gamma*
	container 
½
0resnet_model/batch_normalization_38/gamma/AssignAssign)resnet_model/batch_normalization_38/gamma:resnet_model/batch_normalization_38/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_38/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_38/gamma/readIdentity)resnet_model/batch_normalization_38/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_38/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_38/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_38/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_38/beta
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_38/beta*
	container 
º
/resnet_model/batch_normalization_38/beta/AssignAssign(resnet_model/batch_normalization_38/beta:resnet_model/batch_normalization_38/beta/Initializer/zeros"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_38/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Õ
-resnet_model/batch_normalization_38/beta/readIdentity(resnet_model/batch_normalization_38/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_38/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_38/moving_mean/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_38/moving_mean*
valueB*    *
dtype0
ð
/resnet_model/batch_normalization_38/moving_mean
VariableV2"/device:CPU:0*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_38/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
6resnet_model/batch_normalization_38/moving_mean/AssignAssign/resnet_model/batch_normalization_38/moving_meanAresnet_model/batch_normalization_38/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_38/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_38/moving_mean/readIdentity/resnet_model/batch_normalization_38/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_38/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_38/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_38/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_38/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_38/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_38/moving_variance/AssignAssign3resnet_model/batch_normalization_38/moving_varianceDresnet_model/batch_normalization_38/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_38/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_38/moving_variance/readIdentity3resnet_model/batch_normalization_38/moving_variance"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_38/moving_variance*
_output_shapes	
:*
T0
Ë
2resnet_model/batch_normalization_38/FusedBatchNormFusedBatchNormresnet_model/conv2d_41/Conv2D.resnet_model/batch_normalization_38/gamma/read-resnet_model/batch_normalization_38/beta/read4resnet_model/batch_normalization_38/moving_mean/read8resnet_model/batch_normalization_38/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_38/ConstConst"/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *d;?

resnet_model/Relu_38Relu2resnet_model/batch_normalization_38/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_42/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_42/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_42/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_42/kernel*
valueB
 *    *
dtype0
¸
Aresnet_model/conv2d_42/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_42/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_42/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_42/kernel/Initializer/truncated_normal/shape*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_42/kernel*
seed2 *
dtype0*(
_output_shapes
:
¹
>resnet_model/conv2d_42/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_42/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_42/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_42/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_42/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_42/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_42/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_42/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_42/kernel
VariableV2"/device:CPU:0*
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_42/kernel*
	container 
¦
$resnet_model/conv2d_42/kernel/AssignAssignresnet_model/conv2d_42/kernel:resnet_model/conv2d_42/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_42/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_42/kernel/readIdentityresnet_model/conv2d_42/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_42/kernel*(
_output_shapes
:

$resnet_model/conv2d_42/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_42/Conv2DConv2Dresnet_model/Relu_38"resnet_model/conv2d_42/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME

resnet_model/add_12Addresnet_model/conv2d_42/Conv2Dresnet_model/add_11"/device:GPU:0*
T0*'
_output_shapes
:@
{
resnet_model/block_layer3Identityresnet_model/add_12"/device:GPU:0*
T0*'
_output_shapes
:@
Ó
Jresnet_model/batch_normalization_39/gamma/Initializer/ones/shape_as_tensorConst*<
_class2
0.loc:@resnet_model/batch_normalization_39/gamma*
valueB:*
dtype0*
_output_shapes
:
Ã
@resnet_model/batch_normalization_39/gamma/Initializer/ones/ConstConst*<
_class2
0.loc:@resnet_model/batch_normalization_39/gamma*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Æ
:resnet_model/batch_normalization_39/gamma/Initializer/onesFillJresnet_model/batch_normalization_39/gamma/Initializer/ones/shape_as_tensor@resnet_model/batch_normalization_39/gamma/Initializer/ones/Const*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_39/gamma*

index_type0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_39/gamma
VariableV2"/device:CPU:0*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_39/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
½
0resnet_model/batch_normalization_39/gamma/AssignAssign)resnet_model/batch_normalization_39/gamma:resnet_model/batch_normalization_39/gamma/Initializer/ones"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_39/gamma*
validate_shape(
Ø
.resnet_model/batch_normalization_39/gamma/readIdentity)resnet_model/batch_normalization_39/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_39/gamma*
_output_shapes	
:
Ò
Jresnet_model/batch_normalization_39/beta/Initializer/zeros/shape_as_tensorConst*;
_class1
/-loc:@resnet_model/batch_normalization_39/beta*
valueB:*
dtype0*
_output_shapes
:
Â
@resnet_model/batch_normalization_39/beta/Initializer/zeros/ConstConst*;
_class1
/-loc:@resnet_model/batch_normalization_39/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
Å
:resnet_model/batch_normalization_39/beta/Initializer/zerosFillJresnet_model/batch_normalization_39/beta/Initializer/zeros/shape_as_tensor@resnet_model/batch_normalization_39/beta/Initializer/zeros/Const*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_39/beta*

index_type0*
_output_shapes	
:
â
(resnet_model/batch_normalization_39/beta
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_39/beta*
	container 
º
/resnet_model/batch_normalization_39/beta/AssignAssign(resnet_model/batch_normalization_39/beta:resnet_model/batch_normalization_39/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_39/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_39/beta/readIdentity(resnet_model/batch_normalization_39/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_39/beta
à
Qresnet_model/batch_normalization_39/moving_mean/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*B
_class8
64loc:@resnet_model/batch_normalization_39/moving_mean*
valueB:*
dtype0
Ð
Gresnet_model/batch_normalization_39/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@resnet_model/batch_normalization_39/moving_mean*
valueB
 *    
á
Aresnet_model/batch_normalization_39/moving_mean/Initializer/zerosFillQresnet_model/batch_normalization_39/moving_mean/Initializer/zeros/shape_as_tensorGresnet_model/batch_normalization_39/moving_mean/Initializer/zeros/Const*
T0*B
_class8
64loc:@resnet_model/batch_normalization_39/moving_mean*

index_type0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_39/moving_mean
VariableV2"/device:CPU:0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_39/moving_mean*
	container *
shape:*
dtype0
Ö
6resnet_model/batch_normalization_39/moving_mean/AssignAssign/resnet_model/batch_normalization_39/moving_meanAresnet_model/batch_normalization_39/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_39/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_39/moving_mean/readIdentity/resnet_model/batch_normalization_39/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_39/moving_mean*
_output_shapes	
:
ç
Tresnet_model/batch_normalization_39/moving_variance/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*F
_class<
:8loc:@resnet_model/batch_normalization_39/moving_variance*
valueB:*
dtype0
×
Jresnet_model/batch_normalization_39/moving_variance/Initializer/ones/ConstConst*F
_class<
:8loc:@resnet_model/batch_normalization_39/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 
î
Dresnet_model/batch_normalization_39/moving_variance/Initializer/onesFillTresnet_model/batch_normalization_39/moving_variance/Initializer/ones/shape_as_tensorJresnet_model/batch_normalization_39/moving_variance/Initializer/ones/Const*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_39/moving_variance*

index_type0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_39/moving_variance
VariableV2"/device:CPU:0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_39/moving_variance*
	container *
shape:*
dtype0
å
:resnet_model/batch_normalization_39/moving_variance/AssignAssign3resnet_model/batch_normalization_39/moving_varianceDresnet_model/batch_normalization_39/moving_variance/Initializer/ones"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_39/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
8resnet_model/batch_normalization_39/moving_variance/readIdentity3resnet_model/batch_normalization_39/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_39/moving_variance
Ç
2resnet_model/batch_normalization_39/FusedBatchNormFusedBatchNormresnet_model/block_layer3.resnet_model/batch_normalization_39/gamma/read-resnet_model/batch_normalization_39/beta/read4resnet_model/batch_normalization_39/moving_mean/read8resnet_model/batch_normalization_39/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_39/ConstConst"/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *d;?

resnet_model/Relu_39Relu2resnet_model/batch_normalization_39/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@

resnet_model/Pad_5/paddingsConst"/device:GPU:0*
_output_shapes

:*9
value0B."                                 *
dtype0

resnet_model/Pad_5Padresnet_model/Relu_39resnet_model/Pad_5/paddings"/device:GPU:0*
T0*
	Tpaddings0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_43/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_43/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_43/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_43/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_43/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_43/kernel*
valueB
 *   =
ª
Jresnet_model/conv2d_43/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_43/kernel/Initializer/truncated_normal/shape*0
_class&
$"loc:@resnet_model/conv2d_43/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0
¹
>resnet_model/conv2d_43/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_43/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_43/kernel/Initializer/truncated_normal/stddev*0
_class&
$"loc:@resnet_model/conv2d_43/kernel*(
_output_shapes
:*
T0
§
:resnet_model/conv2d_43/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_43/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_43/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_43/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_43/kernel
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_43/kernel
¦
$resnet_model/conv2d_43/kernel/AssignAssignresnet_model/conv2d_43/kernel:resnet_model/conv2d_43/kernel/Initializer/truncated_normal"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_43/kernel*
validate_shape(
Á
"resnet_model/conv2d_43/kernel/readIdentityresnet_model/conv2d_43/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_43/kernel*(
_output_shapes
:

$resnet_model/conv2d_43/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_43/Conv2DConv2Dresnet_model/Pad_5"resnet_model/conv2d_43/kernel/read"/device:GPU:0*
paddingVALID*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Ë
@resnet_model/conv2d_44/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_44/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_44/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_44/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_44/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_44/kernel*
valueB
 *   =*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_44/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_44/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_44/kernel
¹
>resnet_model/conv2d_44/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_44/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_44/kernel/Initializer/truncated_normal/stddev*0
_class&
$"loc:@resnet_model/conv2d_44/kernel*(
_output_shapes
:*
T0
§
:resnet_model/conv2d_44/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_44/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_44/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_44/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_44/kernel
VariableV2"/device:CPU:0*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_44/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:
¦
$resnet_model/conv2d_44/kernel/AssignAssignresnet_model/conv2d_44/kernel:resnet_model/conv2d_44/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_44/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_44/kernel/readIdentityresnet_model/conv2d_44/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_44/kernel*(
_output_shapes
:

$resnet_model/conv2d_44/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_44/Conv2DConv2Dresnet_model/Relu_39"resnet_model/conv2d_44/kernel/read"/device:GPU:0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0
Ç
:resnet_model/batch_normalization_40/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_40/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_40/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_40/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_40/gamma/AssignAssign)resnet_model/batch_normalization_40/gamma:resnet_model/batch_normalization_40/gamma/Initializer/ones"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_40/gamma
Ø
.resnet_model/batch_normalization_40/gamma/readIdentity)resnet_model/batch_normalization_40/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_40/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_40/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_40/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_40/beta
VariableV2"/device:CPU:0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_40/beta*
	container *
shape:*
dtype0
º
/resnet_model/batch_normalization_40/beta/AssignAssign(resnet_model/batch_normalization_40/beta:resnet_model/batch_normalization_40/beta/Initializer/zeros"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_40/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Õ
-resnet_model/batch_normalization_40/beta/readIdentity(resnet_model/batch_normalization_40/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_40/beta
Ô
Aresnet_model/batch_normalization_40/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_40/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_40/moving_mean
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_40/moving_mean
Ö
6resnet_model/batch_normalization_40/moving_mean/AssignAssign/resnet_model/batch_normalization_40/moving_meanAresnet_model/batch_normalization_40/moving_mean/Initializer/zeros"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_40/moving_mean*
validate_shape(
ê
4resnet_model/batch_normalization_40/moving_mean/readIdentity/resnet_model/batch_normalization_40/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_40/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_40/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_40/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_40/moving_variance
VariableV2"/device:CPU:0*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_40/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
å
:resnet_model/batch_normalization_40/moving_variance/AssignAssign3resnet_model/batch_normalization_40/moving_varianceDresnet_model/batch_normalization_40/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_40/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_40/moving_variance/readIdentity3resnet_model/batch_normalization_40/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_40/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_40/FusedBatchNormFusedBatchNormresnet_model/conv2d_44/Conv2D.resnet_model/batch_normalization_40/gamma/read-resnet_model/batch_normalization_40/beta/read4resnet_model/batch_normalization_40/moving_mean/read8resnet_model/batch_normalization_40/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_40/ConstConst"/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *d;?

resnet_model/Relu_40Relu2resnet_model/batch_normalization_40/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@

resnet_model/Pad_6/paddingsConst"/device:GPU:0*
_output_shapes

:*9
value0B."                             *
dtype0

resnet_model/Pad_6Padresnet_model/Relu_40resnet_model/Pad_6/paddings"/device:GPU:0*'
_output_shapes
:@*
T0*
	Tpaddings0
Ë
@resnet_model/conv2d_45/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_45/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_45/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_45/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_45/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_45/kernel*
valueB
 *ï[q<*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_45/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_45/kernel/Initializer/truncated_normal/shape*0
_class&
$"loc:@resnet_model/conv2d_45/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0
¹
>resnet_model/conv2d_45/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_45/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_45/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_45/kernel
§
:resnet_model/conv2d_45/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_45/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_45/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_45/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_45/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_45/kernel*
	container *
shape:
¦
$resnet_model/conv2d_45/kernel/AssignAssignresnet_model/conv2d_45/kernel:resnet_model/conv2d_45/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_45/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_45/kernel/readIdentityresnet_model/conv2d_45/kernel"/device:CPU:0*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_45/kernel

$resnet_model/conv2d_45/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_45/Conv2DConv2Dresnet_model/Pad_6"resnet_model/conv2d_45/kernel/read"/device:GPU:0*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides

Ç
:resnet_model/batch_normalization_41/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_41/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_41/gamma
VariableV2"/device:CPU:0*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_41/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
½
0resnet_model/batch_normalization_41/gamma/AssignAssign)resnet_model/batch_normalization_41/gamma:resnet_model/batch_normalization_41/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_41/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_41/gamma/readIdentity)resnet_model/batch_normalization_41/gamma"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_41/gamma*
_output_shapes	
:*
T0
Æ
:resnet_model/batch_normalization_41/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_41/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_41/beta
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_41/beta
º
/resnet_model/batch_normalization_41/beta/AssignAssign(resnet_model/batch_normalization_41/beta:resnet_model/batch_normalization_41/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_41/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_41/beta/readIdentity(resnet_model/batch_normalization_41/beta"/device:CPU:0*
_output_shapes	
:*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_41/beta
Ô
Aresnet_model/batch_normalization_41/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_41/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_41/moving_mean
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_41/moving_mean*
	container 
Ö
6resnet_model/batch_normalization_41/moving_mean/AssignAssign/resnet_model/batch_normalization_41/moving_meanAresnet_model/batch_normalization_41/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_41/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_41/moving_mean/readIdentity/resnet_model/batch_normalization_41/moving_mean"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_41/moving_mean*
_output_shapes	
:*
T0
Û
Dresnet_model/batch_normalization_41/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_41/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_41/moving_variance
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_41/moving_variance
å
:resnet_model/batch_normalization_41/moving_variance/AssignAssign3resnet_model/batch_normalization_41/moving_varianceDresnet_model/batch_normalization_41/moving_variance/Initializer/ones"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_41/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
8resnet_model/batch_normalization_41/moving_variance/readIdentity3resnet_model/batch_normalization_41/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_41/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_41/FusedBatchNormFusedBatchNormresnet_model/conv2d_45/Conv2D.resnet_model/batch_normalization_41/gamma/read-resnet_model/batch_normalization_41/beta/read4resnet_model/batch_normalization_41/moving_mean/read8resnet_model/batch_normalization_41/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_41/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_41Relu2resnet_model/batch_normalization_41/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_46/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_46/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_46/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_46/kernel*
valueB
 *    
¸
Aresnet_model/conv2d_46/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_46/kernel*
valueB
 *ó5=
ª
Jresnet_model/conv2d_46/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_46/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_46/kernel*
seed2 
¹
>resnet_model/conv2d_46/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_46/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_46/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_46/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_46/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_46/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_46/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_46/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_46/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_46/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
¦
$resnet_model/conv2d_46/kernel/AssignAssignresnet_model/conv2d_46/kernel:resnet_model/conv2d_46/kernel/Initializer/truncated_normal"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_46/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
Á
"resnet_model/conv2d_46/kernel/readIdentityresnet_model/conv2d_46/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_46/kernel*(
_output_shapes
:

$resnet_model/conv2d_46/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_46/Conv2DConv2Dresnet_model/Relu_41"resnet_model/conv2d_46/kernel/read"/device:GPU:0*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@

resnet_model/add_13Addresnet_model/conv2d_46/Conv2Dresnet_model/conv2d_43/Conv2D"/device:GPU:0*
T0*'
_output_shapes
:@
Ó
Jresnet_model/batch_normalization_42/gamma/Initializer/ones/shape_as_tensorConst*<
_class2
0.loc:@resnet_model/batch_normalization_42/gamma*
valueB:*
dtype0*
_output_shapes
:
Ã
@resnet_model/batch_normalization_42/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *<
_class2
0.loc:@resnet_model/batch_normalization_42/gamma*
valueB
 *  ?
Æ
:resnet_model/batch_normalization_42/gamma/Initializer/onesFillJresnet_model/batch_normalization_42/gamma/Initializer/ones/shape_as_tensor@resnet_model/batch_normalization_42/gamma/Initializer/ones/Const*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_42/gamma*

index_type0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_42/gamma
VariableV2"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_42/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
½
0resnet_model/batch_normalization_42/gamma/AssignAssign)resnet_model/batch_normalization_42/gamma:resnet_model/batch_normalization_42/gamma/Initializer/ones"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_42/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Ø
.resnet_model/batch_normalization_42/gamma/readIdentity)resnet_model/batch_normalization_42/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_42/gamma*
_output_shapes	
:
Ò
Jresnet_model/batch_normalization_42/beta/Initializer/zeros/shape_as_tensorConst*;
_class1
/-loc:@resnet_model/batch_normalization_42/beta*
valueB:*
dtype0*
_output_shapes
:
Â
@resnet_model/batch_normalization_42/beta/Initializer/zeros/ConstConst*;
_class1
/-loc:@resnet_model/batch_normalization_42/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
Å
:resnet_model/batch_normalization_42/beta/Initializer/zerosFillJresnet_model/batch_normalization_42/beta/Initializer/zeros/shape_as_tensor@resnet_model/batch_normalization_42/beta/Initializer/zeros/Const*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_42/beta*

index_type0*
_output_shapes	
:
â
(resnet_model/batch_normalization_42/beta
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_42/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
º
/resnet_model/batch_normalization_42/beta/AssignAssign(resnet_model/batch_normalization_42/beta:resnet_model/batch_normalization_42/beta/Initializer/zeros"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_42/beta
Õ
-resnet_model/batch_normalization_42/beta/readIdentity(resnet_model/batch_normalization_42/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_42/beta*
_output_shapes	
:
à
Qresnet_model/batch_normalization_42/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*B
_class8
64loc:@resnet_model/batch_normalization_42/moving_mean*
valueB:
Ð
Gresnet_model/batch_normalization_42/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@resnet_model/batch_normalization_42/moving_mean*
valueB
 *    
á
Aresnet_model/batch_normalization_42/moving_mean/Initializer/zerosFillQresnet_model/batch_normalization_42/moving_mean/Initializer/zeros/shape_as_tensorGresnet_model/batch_normalization_42/moving_mean/Initializer/zeros/Const*B
_class8
64loc:@resnet_model/batch_normalization_42/moving_mean*

index_type0*
_output_shapes	
:*
T0
ð
/resnet_model/batch_normalization_42/moving_mean
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_42/moving_mean
Ö
6resnet_model/batch_normalization_42/moving_mean/AssignAssign/resnet_model/batch_normalization_42/moving_meanAresnet_model/batch_normalization_42/moving_mean/Initializer/zeros"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_42/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ê
4resnet_model/batch_normalization_42/moving_mean/readIdentity/resnet_model/batch_normalization_42/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_42/moving_mean*
_output_shapes	
:
ç
Tresnet_model/batch_normalization_42/moving_variance/Initializer/ones/shape_as_tensorConst*F
_class<
:8loc:@resnet_model/batch_normalization_42/moving_variance*
valueB:*
dtype0*
_output_shapes
:
×
Jresnet_model/batch_normalization_42/moving_variance/Initializer/ones/ConstConst*F
_class<
:8loc:@resnet_model/batch_normalization_42/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 
î
Dresnet_model/batch_normalization_42/moving_variance/Initializer/onesFillTresnet_model/batch_normalization_42/moving_variance/Initializer/ones/shape_as_tensorJresnet_model/batch_normalization_42/moving_variance/Initializer/ones/Const*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_42/moving_variance*

index_type0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_42/moving_variance
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_42/moving_variance*
	container 
å
:resnet_model/batch_normalization_42/moving_variance/AssignAssign3resnet_model/batch_normalization_42/moving_varianceDresnet_model/batch_normalization_42/moving_variance/Initializer/ones"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_42/moving_variance*
validate_shape(
ö
8resnet_model/batch_normalization_42/moving_variance/readIdentity3resnet_model/batch_normalization_42/moving_variance"/device:CPU:0*
_output_shapes	
:*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_42/moving_variance
Á
2resnet_model/batch_normalization_42/FusedBatchNormFusedBatchNormresnet_model/add_13.resnet_model/batch_normalization_42/gamma/read-resnet_model/batch_normalization_42/beta/read4resnet_model/batch_normalization_42/moving_mean/read8resnet_model/batch_normalization_42/moving_variance/read"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0
}
)resnet_model/batch_normalization_42/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_42Relu2resnet_model/batch_normalization_42/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_47/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_47/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_47/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_47/kernel*
valueB
 *    *
dtype0
¸
Aresnet_model/conv2d_47/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_47/kernel*
valueB
 *óµ<*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_47/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_47/kernel/Initializer/truncated_normal/shape*0
_class&
$"loc:@resnet_model/conv2d_47/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0
¹
>resnet_model/conv2d_47/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_47/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_47/kernel/Initializer/truncated_normal/stddev*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_47/kernel
§
:resnet_model/conv2d_47/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_47/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_47/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_47/kernel
æ
resnet_model/conv2d_47/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_47/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
¦
$resnet_model/conv2d_47/kernel/AssignAssignresnet_model/conv2d_47/kernel:resnet_model/conv2d_47/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_47/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_47/kernel/readIdentityresnet_model/conv2d_47/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_47/kernel*(
_output_shapes
:

$resnet_model/conv2d_47/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_47/Conv2DConv2Dresnet_model/Relu_42"resnet_model/conv2d_47/kernel/read"/device:GPU:0*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Ç
:resnet_model/batch_normalization_43/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_43/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_43/gamma
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_43/gamma*
	container 
½
0resnet_model/batch_normalization_43/gamma/AssignAssign)resnet_model/batch_normalization_43/gamma:resnet_model/batch_normalization_43/gamma/Initializer/ones"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_43/gamma*
validate_shape(
Ø
.resnet_model/batch_normalization_43/gamma/readIdentity)resnet_model/batch_normalization_43/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_43/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_43/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_43/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_43/beta
VariableV2"/device:CPU:0*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_43/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
º
/resnet_model/batch_normalization_43/beta/AssignAssign(resnet_model/batch_normalization_43/beta:resnet_model/batch_normalization_43/beta/Initializer/zeros"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_43/beta*
validate_shape(
Õ
-resnet_model/batch_normalization_43/beta/readIdentity(resnet_model/batch_normalization_43/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_43/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_43/moving_mean/Initializer/zerosConst*B
_class8
64loc:@resnet_model/batch_normalization_43/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_43/moving_mean
VariableV2"/device:CPU:0*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_43/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
6resnet_model/batch_normalization_43/moving_mean/AssignAssign/resnet_model/batch_normalization_43/moving_meanAresnet_model/batch_normalization_43/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_43/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_43/moving_mean/readIdentity/resnet_model/batch_normalization_43/moving_mean"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_43/moving_mean*
_output_shapes	
:*
T0
Û
Dresnet_model/batch_normalization_43/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_43/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_43/moving_variance
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_43/moving_variance*
	container *
shape:
å
:resnet_model/batch_normalization_43/moving_variance/AssignAssign3resnet_model/batch_normalization_43/moving_varianceDresnet_model/batch_normalization_43/moving_variance/Initializer/ones"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_43/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
8resnet_model/batch_normalization_43/moving_variance/readIdentity3resnet_model/batch_normalization_43/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_43/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_43/FusedBatchNormFusedBatchNormresnet_model/conv2d_47/Conv2D.resnet_model/batch_normalization_43/gamma/read-resnet_model/batch_normalization_43/beta/read4resnet_model/batch_normalization_43/moving_mean/read8resnet_model/batch_normalization_43/moving_variance/read"/device:GPU:0*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW
}
)resnet_model/batch_normalization_43/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_43Relu2resnet_model/batch_normalization_43/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_48/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_48/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_48/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_48/kernel*
valueB
 *    
¸
Aresnet_model/conv2d_48/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_48/kernel*
valueB
 *ï[q<*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_48/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_48/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_48/kernel
¹
>resnet_model/conv2d_48/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_48/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_48/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_48/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_48/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_48/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_48/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_48/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_48/kernel
VariableV2"/device:CPU:0*
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_48/kernel*
	container 
¦
$resnet_model/conv2d_48/kernel/AssignAssignresnet_model/conv2d_48/kernel:resnet_model/conv2d_48/kernel/Initializer/truncated_normal"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_48/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Á
"resnet_model/conv2d_48/kernel/readIdentityresnet_model/conv2d_48/kernel"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_48/kernel*(
_output_shapes
:*
T0

$resnet_model/conv2d_48/dilation_rateConst"/device:GPU:0*
_output_shapes
:*
valueB"      *
dtype0

resnet_model/conv2d_48/Conv2DConv2Dresnet_model/Relu_43"resnet_model/conv2d_48/kernel/read"/device:GPU:0*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(
Ç
:resnet_model/batch_normalization_44/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_44/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_44/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_44/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_44/gamma/AssignAssign)resnet_model/batch_normalization_44/gamma:resnet_model/batch_normalization_44/gamma/Initializer/ones"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_44/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Ø
.resnet_model/batch_normalization_44/gamma/readIdentity)resnet_model/batch_normalization_44/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_44/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_44/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_44/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_44/beta
VariableV2"/device:CPU:0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_44/beta*
	container *
shape:*
dtype0
º
/resnet_model/batch_normalization_44/beta/AssignAssign(resnet_model/batch_normalization_44/beta:resnet_model/batch_normalization_44/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_44/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_44/beta/readIdentity(resnet_model/batch_normalization_44/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_44/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_44/moving_mean/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_44/moving_mean*
valueB*    *
dtype0
ð
/resnet_model/batch_normalization_44/moving_mean
VariableV2"/device:CPU:0*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_44/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
6resnet_model/batch_normalization_44/moving_mean/AssignAssign/resnet_model/batch_normalization_44/moving_meanAresnet_model/batch_normalization_44/moving_mean/Initializer/zeros"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_44/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ê
4resnet_model/batch_normalization_44/moving_mean/readIdentity/resnet_model/batch_normalization_44/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_44/moving_mean
Û
Dresnet_model/batch_normalization_44/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_44/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_44/moving_variance
VariableV2"/device:CPU:0*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_44/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
å
:resnet_model/batch_normalization_44/moving_variance/AssignAssign3resnet_model/batch_normalization_44/moving_varianceDresnet_model/batch_normalization_44/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_44/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_44/moving_variance/readIdentity3resnet_model/batch_normalization_44/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_44/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_44/FusedBatchNormFusedBatchNormresnet_model/conv2d_48/Conv2D.resnet_model/batch_normalization_44/gamma/read-resnet_model/batch_normalization_44/beta/read4resnet_model/batch_normalization_44/moving_mean/read8resnet_model/batch_normalization_44/moving_variance/read"/device:GPU:0*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW
}
)resnet_model/batch_normalization_44/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_44Relu2resnet_model/batch_normalization_44/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_49/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_49/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_49/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_49/kernel*
valueB
 *    *
dtype0
¸
Aresnet_model/conv2d_49/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_49/kernel*
valueB
 *ó5=*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_49/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_49/kernel/Initializer/truncated_normal/shape*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_49/kernel*
seed2 *
dtype0*(
_output_shapes
:
¹
>resnet_model/conv2d_49/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_49/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_49/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_49/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_49/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_49/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_49/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_49/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_49/kernel
VariableV2"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_49/kernel*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name 
¦
$resnet_model/conv2d_49/kernel/AssignAssignresnet_model/conv2d_49/kernel:resnet_model/conv2d_49/kernel/Initializer/truncated_normal"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_49/kernel*
validate_shape(*(
_output_shapes
:
Á
"resnet_model/conv2d_49/kernel/readIdentityresnet_model/conv2d_49/kernel"/device:CPU:0*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_49/kernel

$resnet_model/conv2d_49/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_49/Conv2DConv2Dresnet_model/Relu_44"resnet_model/conv2d_49/kernel/read"/device:GPU:0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0

resnet_model/add_14Addresnet_model/conv2d_49/Conv2Dresnet_model/add_13"/device:GPU:0*'
_output_shapes
:@*
T0
Ó
Jresnet_model/batch_normalization_45/gamma/Initializer/ones/shape_as_tensorConst*<
_class2
0.loc:@resnet_model/batch_normalization_45/gamma*
valueB:*
dtype0*
_output_shapes
:
Ã
@resnet_model/batch_normalization_45/gamma/Initializer/ones/ConstConst*<
_class2
0.loc:@resnet_model/batch_normalization_45/gamma*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Æ
:resnet_model/batch_normalization_45/gamma/Initializer/onesFillJresnet_model/batch_normalization_45/gamma/Initializer/ones/shape_as_tensor@resnet_model/batch_normalization_45/gamma/Initializer/ones/Const*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_45/gamma*

index_type0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_45/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_45/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_45/gamma/AssignAssign)resnet_model/batch_normalization_45/gamma:resnet_model/batch_normalization_45/gamma/Initializer/ones"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_45/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ø
.resnet_model/batch_normalization_45/gamma/readIdentity)resnet_model/batch_normalization_45/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_45/gamma*
_output_shapes	
:
Ò
Jresnet_model/batch_normalization_45/beta/Initializer/zeros/shape_as_tensorConst*;
_class1
/-loc:@resnet_model/batch_normalization_45/beta*
valueB:*
dtype0*
_output_shapes
:
Â
@resnet_model/batch_normalization_45/beta/Initializer/zeros/ConstConst*;
_class1
/-loc:@resnet_model/batch_normalization_45/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
Å
:resnet_model/batch_normalization_45/beta/Initializer/zerosFillJresnet_model/batch_normalization_45/beta/Initializer/zeros/shape_as_tensor@resnet_model/batch_normalization_45/beta/Initializer/zeros/Const*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_45/beta*

index_type0*
_output_shapes	
:
â
(resnet_model/batch_normalization_45/beta
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_45/beta
º
/resnet_model/batch_normalization_45/beta/AssignAssign(resnet_model/batch_normalization_45/beta:resnet_model/batch_normalization_45/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_45/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_45/beta/readIdentity(resnet_model/batch_normalization_45/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_45/beta*
_output_shapes	
:
à
Qresnet_model/batch_normalization_45/moving_mean/Initializer/zeros/shape_as_tensorConst*B
_class8
64loc:@resnet_model/batch_normalization_45/moving_mean*
valueB:*
dtype0*
_output_shapes
:
Ð
Gresnet_model/batch_normalization_45/moving_mean/Initializer/zeros/ConstConst*
_output_shapes
: *B
_class8
64loc:@resnet_model/batch_normalization_45/moving_mean*
valueB
 *    *
dtype0
á
Aresnet_model/batch_normalization_45/moving_mean/Initializer/zerosFillQresnet_model/batch_normalization_45/moving_mean/Initializer/zeros/shape_as_tensorGresnet_model/batch_normalization_45/moving_mean/Initializer/zeros/Const*
T0*B
_class8
64loc:@resnet_model/batch_normalization_45/moving_mean*

index_type0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_45/moving_mean
VariableV2"/device:CPU:0*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_45/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
6resnet_model/batch_normalization_45/moving_mean/AssignAssign/resnet_model/batch_normalization_45/moving_meanAresnet_model/batch_normalization_45/moving_mean/Initializer/zeros"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_45/moving_mean*
validate_shape(
ê
4resnet_model/batch_normalization_45/moving_mean/readIdentity/resnet_model/batch_normalization_45/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_45/moving_mean*
_output_shapes	
:
ç
Tresnet_model/batch_normalization_45/moving_variance/Initializer/ones/shape_as_tensorConst*F
_class<
:8loc:@resnet_model/batch_normalization_45/moving_variance*
valueB:*
dtype0*
_output_shapes
:
×
Jresnet_model/batch_normalization_45/moving_variance/Initializer/ones/ConstConst*F
_class<
:8loc:@resnet_model/batch_normalization_45/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 
î
Dresnet_model/batch_normalization_45/moving_variance/Initializer/onesFillTresnet_model/batch_normalization_45/moving_variance/Initializer/ones/shape_as_tensorJresnet_model/batch_normalization_45/moving_variance/Initializer/ones/Const*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_45/moving_variance*

index_type0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_45/moving_variance
VariableV2"/device:CPU:0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_45/moving_variance*
	container *
shape:*
dtype0
å
:resnet_model/batch_normalization_45/moving_variance/AssignAssign3resnet_model/batch_normalization_45/moving_varianceDresnet_model/batch_normalization_45/moving_variance/Initializer/ones"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_45/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
8resnet_model/batch_normalization_45/moving_variance/readIdentity3resnet_model/batch_normalization_45/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_45/moving_variance*
_output_shapes	
:
Á
2resnet_model/batch_normalization_45/FusedBatchNormFusedBatchNormresnet_model/add_14.resnet_model/batch_normalization_45/gamma/read-resnet_model/batch_normalization_45/beta/read4resnet_model/batch_normalization_45/moving_mean/read8resnet_model/batch_normalization_45/moving_variance/read"/device:GPU:0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0
}
)resnet_model/batch_normalization_45/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_45Relu2resnet_model/batch_normalization_45/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_50/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_50/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_50/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_50/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_50/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_50/kernel*
valueB
 *óµ<*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_50/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_50/kernel/Initializer/truncated_normal/shape*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_50/kernel*
seed2 *
dtype0*(
_output_shapes
:
¹
>resnet_model/conv2d_50/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_50/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_50/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_50/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_50/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_50/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_50/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_50/kernel
æ
resnet_model/conv2d_50/kernel
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_50/kernel
¦
$resnet_model/conv2d_50/kernel/AssignAssignresnet_model/conv2d_50/kernel:resnet_model/conv2d_50/kernel/Initializer/truncated_normal"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_50/kernel*
validate_shape(
Á
"resnet_model/conv2d_50/kernel/readIdentityresnet_model/conv2d_50/kernel"/device:CPU:0*(
_output_shapes
:*
T0*0
_class&
$"loc:@resnet_model/conv2d_50/kernel

$resnet_model/conv2d_50/dilation_rateConst"/device:GPU:0*
dtype0*
_output_shapes
:*
valueB"      

resnet_model/conv2d_50/Conv2DConv2Dresnet_model/Relu_45"resnet_model/conv2d_50/kernel/read"/device:GPU:0*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(
Ç
:resnet_model/batch_normalization_46/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_46/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_46/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_46/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_46/gamma/AssignAssign)resnet_model/batch_normalization_46/gamma:resnet_model/batch_normalization_46/gamma/Initializer/ones"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_46/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Ø
.resnet_model/batch_normalization_46/gamma/readIdentity)resnet_model/batch_normalization_46/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_46/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_46/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_46/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_46/beta
VariableV2"/device:CPU:0*
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_46/beta*
	container 
º
/resnet_model/batch_normalization_46/beta/AssignAssign(resnet_model/batch_normalization_46/beta:resnet_model/batch_normalization_46/beta/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_46/beta*
validate_shape(*
_output_shapes	
:
Õ
-resnet_model/batch_normalization_46/beta/readIdentity(resnet_model/batch_normalization_46/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_46/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_46/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_46/moving_mean*
valueB*    
ð
/resnet_model/batch_normalization_46/moving_mean
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_46/moving_mean*
	container *
shape:
Ö
6resnet_model/batch_normalization_46/moving_mean/AssignAssign/resnet_model/batch_normalization_46/moving_meanAresnet_model/batch_normalization_46/moving_mean/Initializer/zeros"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_46/moving_mean*
validate_shape(
ê
4resnet_model/batch_normalization_46/moving_mean/readIdentity/resnet_model/batch_normalization_46/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_46/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_46/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_46/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_46/moving_variance
VariableV2"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_46/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
å
:resnet_model/batch_normalization_46/moving_variance/AssignAssign3resnet_model/batch_normalization_46/moving_varianceDresnet_model/batch_normalization_46/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_46/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_46/moving_variance/readIdentity3resnet_model/batch_normalization_46/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_46/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_46/FusedBatchNormFusedBatchNormresnet_model/conv2d_50/Conv2D.resnet_model/batch_normalization_46/gamma/read-resnet_model/batch_normalization_46/beta/read4resnet_model/batch_normalization_46/moving_mean/read8resnet_model/batch_normalization_46/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_46/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_46Relu2resnet_model/batch_normalization_46/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@
Ë
@resnet_model/conv2d_51/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_51/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_51/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_51/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_51/kernel/Initializer/truncated_normal/stddevConst*0
_class&
$"loc:@resnet_model/conv2d_51/kernel*
valueB
 *ï[q<*
dtype0*
_output_shapes
: 
ª
Jresnet_model/conv2d_51/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_51/kernel/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:*

seed *
T0*0
_class&
$"loc:@resnet_model/conv2d_51/kernel*
seed2 
¹
>resnet_model/conv2d_51/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_51/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_51/kernel/Initializer/truncated_normal/stddev*0
_class&
$"loc:@resnet_model/conv2d_51/kernel*(
_output_shapes
:*
T0
§
:resnet_model/conv2d_51/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_51/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_51/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_51/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_51/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_51/kernel*
	container *
shape:
¦
$resnet_model/conv2d_51/kernel/AssignAssignresnet_model/conv2d_51/kernel:resnet_model/conv2d_51/kernel/Initializer/truncated_normal"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_51/kernel*
validate_shape(
Á
"resnet_model/conv2d_51/kernel/readIdentityresnet_model/conv2d_51/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_51/kernel*(
_output_shapes
:

$resnet_model/conv2d_51/dilation_rateConst"/device:GPU:0*
dtype0*
_output_shapes
:*
valueB"      

resnet_model/conv2d_51/Conv2DConv2Dresnet_model/Relu_46"resnet_model/conv2d_51/kernel/read"/device:GPU:0*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
Ç
:resnet_model/batch_normalization_47/gamma/Initializer/onesConst*<
_class2
0.loc:@resnet_model/batch_normalization_47/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_47/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_47/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_47/gamma/AssignAssign)resnet_model/batch_normalization_47/gamma:resnet_model/batch_normalization_47/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_47/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_47/gamma/readIdentity)resnet_model/batch_normalization_47/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_47/gamma*
_output_shapes	
:
Æ
:resnet_model/batch_normalization_47/beta/Initializer/zerosConst*;
_class1
/-loc:@resnet_model/batch_normalization_47/beta*
valueB*    *
dtype0*
_output_shapes	
:
â
(resnet_model/batch_normalization_47/beta
VariableV2"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_47/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
º
/resnet_model/batch_normalization_47/beta/AssignAssign(resnet_model/batch_normalization_47/beta:resnet_model/batch_normalization_47/beta/Initializer/zeros"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_47/beta*
validate_shape(
Õ
-resnet_model/batch_normalization_47/beta/readIdentity(resnet_model/batch_normalization_47/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_47/beta*
_output_shapes	
:
Ô
Aresnet_model/batch_normalization_47/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@resnet_model/batch_normalization_47/moving_mean*
valueB*    
ð
/resnet_model/batch_normalization_47/moving_mean
VariableV2"/device:CPU:0*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_47/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
6resnet_model/batch_normalization_47/moving_mean/AssignAssign/resnet_model/batch_normalization_47/moving_meanAresnet_model/batch_normalization_47/moving_mean/Initializer/zeros"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_47/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ê
4resnet_model/batch_normalization_47/moving_mean/readIdentity/resnet_model/batch_normalization_47/moving_mean"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_47/moving_mean*
_output_shapes	
:
Û
Dresnet_model/batch_normalization_47/moving_variance/Initializer/onesConst*F
_class<
:8loc:@resnet_model/batch_normalization_47/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_47/moving_variance
VariableV2"/device:CPU:0*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_47/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
å
:resnet_model/batch_normalization_47/moving_variance/AssignAssign3resnet_model/batch_normalization_47/moving_varianceDresnet_model/batch_normalization_47/moving_variance/Initializer/ones"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_47/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
8resnet_model/batch_normalization_47/moving_variance/readIdentity3resnet_model/batch_normalization_47/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_47/moving_variance*
_output_shapes	
:
Ë
2resnet_model/batch_normalization_47/FusedBatchNormFusedBatchNormresnet_model/conv2d_51/Conv2D.resnet_model/batch_normalization_47/gamma/read-resnet_model/batch_normalization_47/beta/read4resnet_model/batch_normalization_47/moving_mean/read8resnet_model/batch_normalization_47/moving_variance/read"/device:GPU:0*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7
}
)resnet_model/batch_normalization_47/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_47Relu2resnet_model/batch_normalization_47/FusedBatchNorm"/device:GPU:0*'
_output_shapes
:@*
T0
Ë
@resnet_model/conv2d_52/kernel/Initializer/truncated_normal/shapeConst*0
_class&
$"loc:@resnet_model/conv2d_52/kernel*%
valueB"            *
dtype0*
_output_shapes
:
¶
?resnet_model/conv2d_52/kernel/Initializer/truncated_normal/meanConst*0
_class&
$"loc:@resnet_model/conv2d_52/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¸
Aresnet_model/conv2d_52/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *0
_class&
$"loc:@resnet_model/conv2d_52/kernel*
valueB
 *ó5=*
dtype0
ª
Jresnet_model/conv2d_52/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@resnet_model/conv2d_52/kernel/Initializer/truncated_normal/shape*
T0*0
_class&
$"loc:@resnet_model/conv2d_52/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed 
¹
>resnet_model/conv2d_52/kernel/Initializer/truncated_normal/mulMulJresnet_model/conv2d_52/kernel/Initializer/truncated_normal/TruncatedNormalAresnet_model/conv2d_52/kernel/Initializer/truncated_normal/stddev*
T0*0
_class&
$"loc:@resnet_model/conv2d_52/kernel*(
_output_shapes
:
§
:resnet_model/conv2d_52/kernel/Initializer/truncated_normalAdd>resnet_model/conv2d_52/kernel/Initializer/truncated_normal/mul?resnet_model/conv2d_52/kernel/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@resnet_model/conv2d_52/kernel*(
_output_shapes
:
æ
resnet_model/conv2d_52/kernel
VariableV2"/device:CPU:0*
dtype0*(
_output_shapes
:*
shared_name *0
_class&
$"loc:@resnet_model/conv2d_52/kernel*
	container *
shape:
¦
$resnet_model/conv2d_52/kernel/AssignAssignresnet_model/conv2d_52/kernel:resnet_model/conv2d_52/kernel/Initializer/truncated_normal"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_52/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
Á
"resnet_model/conv2d_52/kernel/readIdentityresnet_model/conv2d_52/kernel"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_52/kernel*(
_output_shapes
:

$resnet_model/conv2d_52/dilation_rateConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:

resnet_model/conv2d_52/Conv2DConv2Dresnet_model/Relu_47"resnet_model/conv2d_52/kernel/read"/device:GPU:0*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME

resnet_model/add_15Addresnet_model/conv2d_52/Conv2Dresnet_model/add_14"/device:GPU:0*
T0*'
_output_shapes
:@
{
resnet_model/block_layer4Identityresnet_model/add_15"/device:GPU:0*
T0*'
_output_shapes
:@
Ó
Jresnet_model/batch_normalization_48/gamma/Initializer/ones/shape_as_tensorConst*<
_class2
0.loc:@resnet_model/batch_normalization_48/gamma*
valueB:*
dtype0*
_output_shapes
:
Ã
@resnet_model/batch_normalization_48/gamma/Initializer/ones/ConstConst*<
_class2
0.loc:@resnet_model/batch_normalization_48/gamma*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Æ
:resnet_model/batch_normalization_48/gamma/Initializer/onesFillJresnet_model/batch_normalization_48/gamma/Initializer/ones/shape_as_tensor@resnet_model/batch_normalization_48/gamma/Initializer/ones/Const*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_48/gamma*

index_type0*
_output_shapes	
:
ä
)resnet_model/batch_normalization_48/gamma
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@resnet_model/batch_normalization_48/gamma*
	container *
shape:
½
0resnet_model/batch_normalization_48/gamma/AssignAssign)resnet_model/batch_normalization_48/gamma:resnet_model/batch_normalization_48/gamma/Initializer/ones"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_48/gamma*
validate_shape(*
_output_shapes	
:
Ø
.resnet_model/batch_normalization_48/gamma/readIdentity)resnet_model/batch_normalization_48/gamma"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_48/gamma*
_output_shapes	
:
Ò
Jresnet_model/batch_normalization_48/beta/Initializer/zeros/shape_as_tensorConst*;
_class1
/-loc:@resnet_model/batch_normalization_48/beta*
valueB:*
dtype0*
_output_shapes
:
Â
@resnet_model/batch_normalization_48/beta/Initializer/zeros/ConstConst*
_output_shapes
: *;
_class1
/-loc:@resnet_model/batch_normalization_48/beta*
valueB
 *    *
dtype0
Å
:resnet_model/batch_normalization_48/beta/Initializer/zerosFillJresnet_model/batch_normalization_48/beta/Initializer/zeros/shape_as_tensor@resnet_model/batch_normalization_48/beta/Initializer/zeros/Const*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_48/beta*

index_type0*
_output_shapes	
:
â
(resnet_model/batch_normalization_48/beta
VariableV2"/device:CPU:0*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@resnet_model/batch_normalization_48/beta*
	container *
shape:
º
/resnet_model/batch_normalization_48/beta/AssignAssign(resnet_model/batch_normalization_48/beta:resnet_model/batch_normalization_48/beta/Initializer/zeros"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_48/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Õ
-resnet_model/batch_normalization_48/beta/readIdentity(resnet_model/batch_normalization_48/beta"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_48/beta*
_output_shapes	
:
à
Qresnet_model/batch_normalization_48/moving_mean/Initializer/zeros/shape_as_tensorConst*B
_class8
64loc:@resnet_model/batch_normalization_48/moving_mean*
valueB:*
dtype0*
_output_shapes
:
Ð
Gresnet_model/batch_normalization_48/moving_mean/Initializer/zeros/ConstConst*
_output_shapes
: *B
_class8
64loc:@resnet_model/batch_normalization_48/moving_mean*
valueB
 *    *
dtype0
á
Aresnet_model/batch_normalization_48/moving_mean/Initializer/zerosFillQresnet_model/batch_normalization_48/moving_mean/Initializer/zeros/shape_as_tensorGresnet_model/batch_normalization_48/moving_mean/Initializer/zeros/Const*
T0*B
_class8
64loc:@resnet_model/batch_normalization_48/moving_mean*

index_type0*
_output_shapes	
:
ð
/resnet_model/batch_normalization_48/moving_mean
VariableV2"/device:CPU:0*
shared_name *B
_class8
64loc:@resnet_model/batch_normalization_48/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
6resnet_model/batch_normalization_48/moving_mean/AssignAssign/resnet_model/batch_normalization_48/moving_meanAresnet_model/batch_normalization_48/moving_mean/Initializer/zeros"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_48/moving_mean*
validate_shape(*
_output_shapes	
:
ê
4resnet_model/batch_normalization_48/moving_mean/readIdentity/resnet_model/batch_normalization_48/moving_mean"/device:CPU:0*
_output_shapes	
:*
T0*B
_class8
64loc:@resnet_model/batch_normalization_48/moving_mean
ç
Tresnet_model/batch_normalization_48/moving_variance/Initializer/ones/shape_as_tensorConst*F
_class<
:8loc:@resnet_model/batch_normalization_48/moving_variance*
valueB:*
dtype0*
_output_shapes
:
×
Jresnet_model/batch_normalization_48/moving_variance/Initializer/ones/ConstConst*F
_class<
:8loc:@resnet_model/batch_normalization_48/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 
î
Dresnet_model/batch_normalization_48/moving_variance/Initializer/onesFillTresnet_model/batch_normalization_48/moving_variance/Initializer/ones/shape_as_tensorJresnet_model/batch_normalization_48/moving_variance/Initializer/ones/Const*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_48/moving_variance*

index_type0*
_output_shapes	
:
ø
3resnet_model/batch_normalization_48/moving_variance
VariableV2"/device:CPU:0*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@resnet_model/batch_normalization_48/moving_variance
å
:resnet_model/batch_normalization_48/moving_variance/AssignAssign3resnet_model/batch_normalization_48/moving_varianceDresnet_model/batch_normalization_48/moving_variance/Initializer/ones"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_48/moving_variance*
validate_shape(*
_output_shapes	
:
ö
8resnet_model/batch_normalization_48/moving_variance/readIdentity3resnet_model/batch_normalization_48/moving_variance"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_48/moving_variance*
_output_shapes	
:
Ç
2resnet_model/batch_normalization_48/FusedBatchNormFusedBatchNormresnet_model/block_layer4.resnet_model/batch_normalization_48/gamma/read-resnet_model/batch_normalization_48/beta/read4resnet_model/batch_normalization_48/moving_mean/read8resnet_model/batch_normalization_48/moving_variance/read"/device:GPU:0*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 
}
)resnet_model/batch_normalization_48/ConstConst"/device:GPU:0*
valueB
 *d;?*
dtype0*
_output_shapes
: 

resnet_model/Relu_48Relu2resnet_model/batch_normalization_48/FusedBatchNorm"/device:GPU:0*
T0*'
_output_shapes
:@

#resnet_model/Mean/reduction_indicesConst"/device:GPU:0*
valueB"      *
dtype0*
_output_shapes
:
²
resnet_model/MeanMeanresnet_model/Relu_48#resnet_model/Mean/reduction_indices"/device:GPU:0*'
_output_shapes
:@*
	keep_dims(*

Tidx0*
T0
~
resnet_model/final_reduce_meanIdentityresnet_model/Mean"/device:GPU:0*
T0*'
_output_shapes
:@
z
resnet_model/Reshape/shapeConst"/device:GPU:0*
dtype0*
_output_shapes
:*
valueB"ÿÿÿÿ   
¢
resnet_model/ReshapeReshaperesnet_model/final_reduce_meanresnet_model/Reshape/shape"/device:GPU:0*
T0*
Tshape0*
_output_shapes
:	@
¹
:resnet_model/dense/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@resnet_model/dense/kernel*
valueB"   é  *
dtype0*
_output_shapes
:
«
8resnet_model/dense/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@resnet_model/dense/kernel*
valueB
 *h³5½*
dtype0*
_output_shapes
: 
«
8resnet_model/dense/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@resnet_model/dense/kernel*
valueB
 *h³5=*
dtype0*
_output_shapes
: 

Bresnet_model/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform:resnet_model/dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
é*

seed *
T0*,
_class"
 loc:@resnet_model/dense/kernel*
seed2 *
dtype0

8resnet_model/dense/kernel/Initializer/random_uniform/subSub8resnet_model/dense/kernel/Initializer/random_uniform/max8resnet_model/dense/kernel/Initializer/random_uniform/min*,
_class"
 loc:@resnet_model/dense/kernel*
_output_shapes
: *
T0

8resnet_model/dense/kernel/Initializer/random_uniform/mulMulBresnet_model/dense/kernel/Initializer/random_uniform/RandomUniform8resnet_model/dense/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@resnet_model/dense/kernel* 
_output_shapes
:
é

4resnet_model/dense/kernel/Initializer/random_uniformAdd8resnet_model/dense/kernel/Initializer/random_uniform/mul8resnet_model/dense/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@resnet_model/dense/kernel* 
_output_shapes
:
é
Î
resnet_model/dense/kernel
VariableV2"/device:CPU:0*,
_class"
 loc:@resnet_model/dense/kernel*
	container *
shape:
é*
dtype0* 
_output_shapes
:
é*
shared_name 

 resnet_model/dense/kernel/AssignAssignresnet_model/dense/kernel4resnet_model/dense/kernel/Initializer/random_uniform"/device:CPU:0*
use_locking(*
T0*,
_class"
 loc:@resnet_model/dense/kernel*
validate_shape(* 
_output_shapes
:
é
­
resnet_model/dense/kernel/readIdentityresnet_model/dense/kernel"/device:CPU:0*
T0*,
_class"
 loc:@resnet_model/dense/kernel* 
_output_shapes
:
é
°
9resnet_model/dense/bias/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@resnet_model/dense/bias*
valueB:é*
dtype0*
_output_shapes
:
 
/resnet_model/dense/bias/Initializer/zeros/ConstConst**
_class 
loc:@resnet_model/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 

)resnet_model/dense/bias/Initializer/zerosFill9resnet_model/dense/bias/Initializer/zeros/shape_as_tensor/resnet_model/dense/bias/Initializer/zeros/Const*
_output_shapes	
:é*
T0**
_class 
loc:@resnet_model/dense/bias*

index_type0
À
resnet_model/dense/bias
VariableV2"/device:CPU:0*
	container *
shape:é*
dtype0*
_output_shapes	
:é*
shared_name **
_class 
loc:@resnet_model/dense/bias
ö
resnet_model/dense/bias/AssignAssignresnet_model/dense/bias)resnet_model/dense/bias/Initializer/zeros"/device:CPU:0*
use_locking(*
T0**
_class 
loc:@resnet_model/dense/bias*
validate_shape(*
_output_shapes	
:é
¢
resnet_model/dense/bias/readIdentityresnet_model/dense/bias"/device:CPU:0*
_output_shapes	
:é*
T0**
_class 
loc:@resnet_model/dense/bias
¸
resnet_model/dense/MatMulMatMulresnet_model/Reshaperesnet_model/dense/kernel/read"/device:GPU:0*
T0*
_output_shapes
:	@é*
transpose_a( *
transpose_b( 
®
resnet_model/dense/BiasAddBiasAddresnet_model/dense/MatMulresnet_model/dense/bias/read"/device:GPU:0*
T0*
data_formatNHWC*
_output_shapes
:	@é
y
resnet_model/final_denseIdentityresnet_model/dense/BiasAdd"/device:GPU:0*
T0*
_output_shapes
:	@é
a
ArgMax/dimensionConst"/device:GPU:0*
value	B :*
dtype0*
_output_shapes
: 

ArgMaxArgMaxresnet_model/final_denseArgMax/dimension"/device:GPU:0*
T0*
output_type0	*
_output_shapes
:@*

Tidx0
l
softmax_tensorSoftmaxresnet_model/final_dense"/device:GPU:0*
_output_shapes
:	@é*
T0
p
tower_1/images/tagConst"/device:GPU:1*
valueB Btower_1/images*
dtype0*
_output_shapes
: 
«
tower_1/imagesImageSummarytower_1/images/tagsplit_inputs/split:1"/device:GPU:1*
_output_shapes
: *

max_images*
T0*
	bad_colorB:ÿ  ÿ

#tower_1/resnet_model/transpose/permConst"/device:GPU:1*%
valueB"             *
dtype0*
_output_shapes
:
µ
tower_1/resnet_model/transpose	Transposesplit_inputs/split:1#tower_1/resnet_model/transpose/perm"/device:GPU:1*(
_output_shapes
:@àà*
Tperm0*
T0
¡
!tower_1/resnet_model/Pad/paddingsConst"/device:GPU:1*9
value0B."                             *
dtype0*
_output_shapes

:
µ
tower_1/resnet_model/PadPadtower_1/resnet_model/transpose!tower_1/resnet_model/Pad/paddings"/device:GPU:1*
T0*
	Tpaddings0*(
_output_shapes
:@ææ

)tower_1/resnet_model/conv2d/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

"tower_1/resnet_model/conv2d/Conv2DConv2Dtower_1/resnet_model/Padresnet_model/conv2d/kernel/read"/device:GPU:1*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:@@pp*
	dilations
*
T0*
data_formatNCHW*
strides


!tower_1/resnet_model/initial_convIdentity"tower_1/resnet_model/conv2d/Conv2D"/device:GPU:1*&
_output_shapes
:@@pp*
T0
ê
*tower_1/resnet_model/max_pooling2d/MaxPoolMaxPool!tower_1/resnet_model/initial_conv"/device:GPU:1*&
_output_shapes
:@@88*
T0*
data_formatNCHW*
strides
*
ksize
*
paddingSAME

%tower_1/resnet_model/initial_max_poolIdentity*tower_1/resnet_model/max_pooling2d/MaxPool"/device:GPU:1*
T0*&
_output_shapes
:@@88
Ç
7tower_1/resnet_model/batch_normalization/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/initial_max_pool+resnet_model/batch_normalization/gamma/read*resnet_model/batch_normalization/beta/read1resnet_model/batch_normalization/moving_mean/read5resnet_model/batch_normalization/moving_variance/read"/device:GPU:1*
T0*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( *
epsilon%ð'7

.tower_1/resnet_model/batch_normalization/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 

tower_1/resnet_model/ReluRelu7tower_1/resnet_model/batch_normalization/FusedBatchNorm"/device:GPU:1*&
_output_shapes
:@@88*
T0

+tower_1/resnet_model/conv2d_1/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

$tower_1/resnet_model/conv2d_1/Conv2DConv2Dtower_1/resnet_model/Relu!resnet_model/conv2d_1/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@88*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(

+tower_1/resnet_model/conv2d_2/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

$tower_1/resnet_model/conv2d_2/Conv2DConv2Dtower_1/resnet_model/Relu!resnet_model/conv2d_2/kernel/read"/device:GPU:1*
paddingSAME*&
_output_shapes
:@@88*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Ð
9tower_1/resnet_model/batch_normalization_1/FusedBatchNormFusedBatchNorm$tower_1/resnet_model/conv2d_2/Conv2D-resnet_model/batch_normalization_1/gamma/read,resnet_model/batch_normalization_1/beta/read3resnet_model/batch_normalization_1/moving_mean/read7resnet_model/batch_normalization_1/moving_variance/read"/device:GPU:1*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW

0tower_1/resnet_model/batch_normalization_1/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *d;?*
dtype0

tower_1/resnet_model/Relu_1Relu9tower_1/resnet_model/batch_normalization_1/FusedBatchNorm"/device:GPU:1*
T0*&
_output_shapes
:@@88

+tower_1/resnet_model/conv2d_3/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

$tower_1/resnet_model/conv2d_3/Conv2DConv2Dtower_1/resnet_model/Relu_1!resnet_model/conv2d_3/kernel/read"/device:GPU:1*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@88*
	dilations
*
T0
Ð
9tower_1/resnet_model/batch_normalization_2/FusedBatchNormFusedBatchNorm$tower_1/resnet_model/conv2d_3/Conv2D-resnet_model/batch_normalization_2/gamma/read,resnet_model/batch_normalization_2/beta/read3resnet_model/batch_normalization_2/moving_mean/read7resnet_model/batch_normalization_2/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( 

0tower_1/resnet_model/batch_normalization_2/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 

tower_1/resnet_model/Relu_2Relu9tower_1/resnet_model/batch_normalization_2/FusedBatchNorm"/device:GPU:1*&
_output_shapes
:@@88*
T0

+tower_1/resnet_model/conv2d_4/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

$tower_1/resnet_model/conv2d_4/Conv2DConv2Dtower_1/resnet_model/Relu_2!resnet_model/conv2d_4/kernel/read"/device:GPU:1*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@88*
	dilations
*
T0
¬
tower_1/resnet_model/addAdd$tower_1/resnet_model/conv2d_4/Conv2D$tower_1/resnet_model/conv2d_1/Conv2D"/device:GPU:1*
T0*'
_output_shapes
:@88
É
9tower_1/resnet_model/batch_normalization_3/FusedBatchNormFusedBatchNormtower_1/resnet_model/add-resnet_model/batch_normalization_3/gamma/read,resnet_model/batch_normalization_3/beta/read3resnet_model/batch_normalization_3/moving_mean/read7resnet_model/batch_normalization_3/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@88::::*
is_training( 

0tower_1/resnet_model/batch_normalization_3/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 

tower_1/resnet_model/Relu_3Relu9tower_1/resnet_model/batch_normalization_3/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@88

+tower_1/resnet_model/conv2d_5/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

$tower_1/resnet_model/conv2d_5/Conv2DConv2Dtower_1/resnet_model/Relu_3!resnet_model/conv2d_5/kernel/read"/device:GPU:1*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@88*
	dilations
*
T0
Ð
9tower_1/resnet_model/batch_normalization_4/FusedBatchNormFusedBatchNorm$tower_1/resnet_model/conv2d_5/Conv2D-resnet_model/batch_normalization_4/gamma/read,resnet_model/batch_normalization_4/beta/read3resnet_model/batch_normalization_4/moving_mean/read7resnet_model/batch_normalization_4/moving_variance/read"/device:GPU:1*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( *
epsilon%ð'7*
T0

0tower_1/resnet_model/batch_normalization_4/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 

tower_1/resnet_model/Relu_4Relu9tower_1/resnet_model/batch_normalization_4/FusedBatchNorm"/device:GPU:1*&
_output_shapes
:@@88*
T0

+tower_1/resnet_model/conv2d_6/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

$tower_1/resnet_model/conv2d_6/Conv2DConv2Dtower_1/resnet_model/Relu_4!resnet_model/conv2d_6/kernel/read"/device:GPU:1*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@88
Ð
9tower_1/resnet_model/batch_normalization_5/FusedBatchNormFusedBatchNorm$tower_1/resnet_model/conv2d_6/Conv2D-resnet_model/batch_normalization_5/gamma/read,resnet_model/batch_normalization_5/beta/read3resnet_model/batch_normalization_5/moving_mean/read7resnet_model/batch_normalization_5/moving_variance/read"/device:GPU:1*
T0*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( *
epsilon%ð'7

0tower_1/resnet_model/batch_normalization_5/ConstConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *d;?

tower_1/resnet_model/Relu_5Relu9tower_1/resnet_model/batch_normalization_5/FusedBatchNorm"/device:GPU:1*&
_output_shapes
:@@88*
T0

+tower_1/resnet_model/conv2d_7/dilation_rateConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB"      

$tower_1/resnet_model/conv2d_7/Conv2DConv2Dtower_1/resnet_model/Relu_5!resnet_model/conv2d_7/kernel/read"/device:GPU:1*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@88*
	dilations
*
T0
¢
tower_1/resnet_model/add_1Add$tower_1/resnet_model/conv2d_7/Conv2Dtower_1/resnet_model/add"/device:GPU:1*'
_output_shapes
:@88*
T0
Ë
9tower_1/resnet_model/batch_normalization_6/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_1-resnet_model/batch_normalization_6/gamma/read,resnet_model/batch_normalization_6/beta/read3resnet_model/batch_normalization_6/moving_mean/read7resnet_model/batch_normalization_6/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@88::::*
is_training( 

0tower_1/resnet_model/batch_normalization_6/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 

tower_1/resnet_model/Relu_6Relu9tower_1/resnet_model/batch_normalization_6/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@88

+tower_1/resnet_model/conv2d_8/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

$tower_1/resnet_model/conv2d_8/Conv2DConv2Dtower_1/resnet_model/Relu_6!resnet_model/conv2d_8/kernel/read"/device:GPU:1*&
_output_shapes
:@@88*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME
Ð
9tower_1/resnet_model/batch_normalization_7/FusedBatchNormFusedBatchNorm$tower_1/resnet_model/conv2d_8/Conv2D-resnet_model/batch_normalization_7/gamma/read,resnet_model/batch_normalization_7/beta/read3resnet_model/batch_normalization_7/moving_mean/read7resnet_model/batch_normalization_7/moving_variance/read"/device:GPU:1*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( *
epsilon%ð'7*
T0

0tower_1/resnet_model/batch_normalization_7/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 

tower_1/resnet_model/Relu_7Relu9tower_1/resnet_model/batch_normalization_7/FusedBatchNorm"/device:GPU:1*&
_output_shapes
:@@88*
T0

+tower_1/resnet_model/conv2d_9/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

$tower_1/resnet_model/conv2d_9/Conv2DConv2Dtower_1/resnet_model/Relu_7!resnet_model/conv2d_9/kernel/read"/device:GPU:1*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@@88*
	dilations
*
T0
Ð
9tower_1/resnet_model/batch_normalization_8/FusedBatchNormFusedBatchNorm$tower_1/resnet_model/conv2d_9/Conv2D-resnet_model/batch_normalization_8/gamma/read,resnet_model/batch_normalization_8/beta/read3resnet_model/batch_normalization_8/moving_mean/read7resnet_model/batch_normalization_8/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*>
_output_shapes,
*:@@88:@:@:@:@*
is_training( 

0tower_1/resnet_model/batch_normalization_8/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 

tower_1/resnet_model/Relu_8Relu9tower_1/resnet_model/batch_normalization_8/FusedBatchNorm"/device:GPU:1*
T0*&
_output_shapes
:@@88

,tower_1/resnet_model/conv2d_10/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

%tower_1/resnet_model/conv2d_10/Conv2DConv2Dtower_1/resnet_model/Relu_8"resnet_model/conv2d_10/kernel/read"/device:GPU:1*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@88*
	dilations
*
T0
¥
tower_1/resnet_model/add_2Add%tower_1/resnet_model/conv2d_10/Conv2Dtower_1/resnet_model/add_1"/device:GPU:1*
T0*'
_output_shapes
:@88

!tower_1/resnet_model/block_layer1Identitytower_1/resnet_model/add_2"/device:GPU:1*
T0*'
_output_shapes
:@88
Ò
9tower_1/resnet_model/batch_normalization_9/FusedBatchNormFusedBatchNorm!tower_1/resnet_model/block_layer1-resnet_model/batch_normalization_9/gamma/read,resnet_model/batch_normalization_9/beta/read3resnet_model/batch_normalization_9/moving_mean/read7resnet_model/batch_normalization_9/moving_variance/read"/device:GPU:1*C
_output_shapes1
/:@88::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW

0tower_1/resnet_model/batch_normalization_9/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 

tower_1/resnet_model/Relu_9Relu9tower_1/resnet_model/batch_normalization_9/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@88
£
#tower_1/resnet_model/Pad_1/paddingsConst"/device:GPU:1*9
value0B."                                 *
dtype0*
_output_shapes

:
µ
tower_1/resnet_model/Pad_1Padtower_1/resnet_model/Relu_9#tower_1/resnet_model/Pad_1/paddings"/device:GPU:1*
T0*
	Tpaddings0*'
_output_shapes
:@88

,tower_1/resnet_model/conv2d_11/dilation_rateConst"/device:GPU:1*
_output_shapes
:*
valueB"      *
dtype0

%tower_1/resnet_model/conv2d_11/Conv2DConv2Dtower_1/resnet_model/Pad_1"resnet_model/conv2d_11/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingVALID

,tower_1/resnet_model/conv2d_12/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

%tower_1/resnet_model/conv2d_12/Conv2DConv2Dtower_1/resnet_model/Relu_9"resnet_model/conv2d_12/kernel/read"/device:GPU:1*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@88
Û
:tower_1/resnet_model/batch_normalization_10/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_12/Conv2D.resnet_model/batch_normalization_10/gamma/read-resnet_model/batch_normalization_10/beta/read4resnet_model/batch_normalization_10/moving_mean/read8resnet_model/batch_normalization_10/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@88::::*
is_training( 

1tower_1/resnet_model/batch_normalization_10/ConstConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *d;?
¡
tower_1/resnet_model/Relu_10Relu:tower_1/resnet_model/batch_normalization_10/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@88*
T0
£
#tower_1/resnet_model/Pad_2/paddingsConst"/device:GPU:1*9
value0B."                             *
dtype0*
_output_shapes

:
¶
tower_1/resnet_model/Pad_2Padtower_1/resnet_model/Relu_10#tower_1/resnet_model/Pad_2/paddings"/device:GPU:1*
T0*
	Tpaddings0*'
_output_shapes
:@::

,tower_1/resnet_model/conv2d_13/dilation_rateConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB"      

%tower_1/resnet_model/conv2d_13/Conv2DConv2Dtower_1/resnet_model/Pad_2"resnet_model/conv2d_13/kernel/read"/device:GPU:1*
paddingVALID*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Û
:tower_1/resnet_model/batch_normalization_11/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_13/Conv2D.resnet_model/batch_normalization_11/gamma/read-resnet_model/batch_normalization_11/beta/read4resnet_model/batch_normalization_11/moving_mean/read8resnet_model/batch_normalization_11/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_11/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_11Relu:tower_1/resnet_model/batch_normalization_11/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_14/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_14/Conv2DConv2Dtower_1/resnet_model/Relu_11"resnet_model/conv2d_14/kernel/read"/device:GPU:1*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
°
tower_1/resnet_model/add_3Add%tower_1/resnet_model/conv2d_14/Conv2D%tower_1/resnet_model/conv2d_11/Conv2D"/device:GPU:1*'
_output_shapes
:@*
T0
Ð
:tower_1/resnet_model/batch_normalization_12/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_3.resnet_model/batch_normalization_12/gamma/read-resnet_model/batch_normalization_12/beta/read4resnet_model/batch_normalization_12/moving_mean/read8resnet_model/batch_normalization_12/moving_variance/read"/device:GPU:1*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7

1tower_1/resnet_model/batch_normalization_12/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_12Relu:tower_1/resnet_model/batch_normalization_12/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_15/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_15/Conv2DConv2Dtower_1/resnet_model/Relu_12"resnet_model/conv2d_15/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Û
:tower_1/resnet_model/batch_normalization_13/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_15/Conv2D.resnet_model/batch_normalization_13/gamma/read-resnet_model/batch_normalization_13/beta/read4resnet_model/batch_normalization_13/moving_mean/read8resnet_model/batch_normalization_13/moving_variance/read"/device:GPU:1*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7

1tower_1/resnet_model/batch_normalization_13/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *d;?*
dtype0
¡
tower_1/resnet_model/Relu_13Relu:tower_1/resnet_model/batch_normalization_13/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_16/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_16/Conv2DConv2Dtower_1/resnet_model/Relu_13"resnet_model/conv2d_16/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Û
:tower_1/resnet_model/batch_normalization_14/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_16/Conv2D.resnet_model/batch_normalization_14/gamma/read-resnet_model/batch_normalization_14/beta/read4resnet_model/batch_normalization_14/moving_mean/read8resnet_model/batch_normalization_14/moving_variance/read"/device:GPU:1*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW

1tower_1/resnet_model/batch_normalization_14/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *d;?*
dtype0
¡
tower_1/resnet_model/Relu_14Relu:tower_1/resnet_model/batch_normalization_14/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_17/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_17/Conv2DConv2Dtower_1/resnet_model/Relu_14"resnet_model/conv2d_17/kernel/read"/device:GPU:1*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
¥
tower_1/resnet_model/add_4Add%tower_1/resnet_model/conv2d_17/Conv2Dtower_1/resnet_model/add_3"/device:GPU:1*
T0*'
_output_shapes
:@
Ð
:tower_1/resnet_model/batch_normalization_15/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_4.resnet_model/batch_normalization_15/gamma/read-resnet_model/batch_normalization_15/beta/read4resnet_model/batch_normalization_15/moving_mean/read8resnet_model/batch_normalization_15/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_15/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_15Relu:tower_1/resnet_model/batch_normalization_15/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_18/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_18/Conv2DConv2Dtower_1/resnet_model/Relu_15"resnet_model/conv2d_18/kernel/read"/device:GPU:1*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
Û
:tower_1/resnet_model/batch_normalization_16/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_18/Conv2D.resnet_model/batch_normalization_16/gamma/read-resnet_model/batch_normalization_16/beta/read4resnet_model/batch_normalization_16/moving_mean/read8resnet_model/batch_normalization_16/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_16/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_16Relu:tower_1/resnet_model/batch_normalization_16/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_19/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_19/Conv2DConv2Dtower_1/resnet_model/Relu_16"resnet_model/conv2d_19/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Û
:tower_1/resnet_model/batch_normalization_17/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_19/Conv2D.resnet_model/batch_normalization_17/gamma/read-resnet_model/batch_normalization_17/beta/read4resnet_model/batch_normalization_17/moving_mean/read8resnet_model/batch_normalization_17/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_17/ConstConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *d;?
¡
tower_1/resnet_model/Relu_17Relu:tower_1/resnet_model/batch_normalization_17/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_20/dilation_rateConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB"      
 
%tower_1/resnet_model/conv2d_20/Conv2DConv2Dtower_1/resnet_model/Relu_17"resnet_model/conv2d_20/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
¥
tower_1/resnet_model/add_5Add%tower_1/resnet_model/conv2d_20/Conv2Dtower_1/resnet_model/add_4"/device:GPU:1*'
_output_shapes
:@*
T0
Ð
:tower_1/resnet_model/batch_normalization_18/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_5.resnet_model/batch_normalization_18/gamma/read-resnet_model/batch_normalization_18/beta/read4resnet_model/batch_normalization_18/moving_mean/read8resnet_model/batch_normalization_18/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_18/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *d;?*
dtype0
¡
tower_1/resnet_model/Relu_18Relu:tower_1/resnet_model/batch_normalization_18/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_21/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_21/Conv2DConv2Dtower_1/resnet_model/Relu_18"resnet_model/conv2d_21/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Û
:tower_1/resnet_model/batch_normalization_19/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_21/Conv2D.resnet_model/batch_normalization_19/gamma/read-resnet_model/batch_normalization_19/beta/read4resnet_model/batch_normalization_19/moving_mean/read8resnet_model/batch_normalization_19/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_19/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_19Relu:tower_1/resnet_model/batch_normalization_19/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_22/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_22/Conv2DConv2Dtower_1/resnet_model/Relu_19"resnet_model/conv2d_22/kernel/read"/device:GPU:1*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
Û
:tower_1/resnet_model/batch_normalization_20/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_22/Conv2D.resnet_model/batch_normalization_20/gamma/read-resnet_model/batch_normalization_20/beta/read4resnet_model/batch_normalization_20/moving_mean/read8resnet_model/batch_normalization_20/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_20/ConstConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *d;?
¡
tower_1/resnet_model/Relu_20Relu:tower_1/resnet_model/batch_normalization_20/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_23/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_23/Conv2DConv2Dtower_1/resnet_model/Relu_20"resnet_model/conv2d_23/kernel/read"/device:GPU:1*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations

¥
tower_1/resnet_model/add_6Add%tower_1/resnet_model/conv2d_23/Conv2Dtower_1/resnet_model/add_5"/device:GPU:1*
T0*'
_output_shapes
:@

!tower_1/resnet_model/block_layer2Identitytower_1/resnet_model/add_6"/device:GPU:1*
T0*'
_output_shapes
:@
×
:tower_1/resnet_model/batch_normalization_21/FusedBatchNormFusedBatchNorm!tower_1/resnet_model/block_layer2.resnet_model/batch_normalization_21/gamma/read-resnet_model/batch_normalization_21/beta/read4resnet_model/batch_normalization_21/moving_mean/read8resnet_model/batch_normalization_21/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_21/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_21Relu:tower_1/resnet_model/batch_normalization_21/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@
£
#tower_1/resnet_model/Pad_3/paddingsConst"/device:GPU:1*9
value0B."                                 *
dtype0*
_output_shapes

:
¶
tower_1/resnet_model/Pad_3Padtower_1/resnet_model/Relu_21#tower_1/resnet_model/Pad_3/paddings"/device:GPU:1*
T0*
	Tpaddings0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_24/dilation_rateConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB"      

%tower_1/resnet_model/conv2d_24/Conv2DConv2Dtower_1/resnet_model/Pad_3"resnet_model/conv2d_24/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingVALID

,tower_1/resnet_model/conv2d_25/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_25/Conv2DConv2Dtower_1/resnet_model/Relu_21"resnet_model/conv2d_25/kernel/read"/device:GPU:1*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
Û
:tower_1/resnet_model/batch_normalization_22/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_25/Conv2D.resnet_model/batch_normalization_22/gamma/read-resnet_model/batch_normalization_22/beta/read4resnet_model/batch_normalization_22/moving_mean/read8resnet_model/batch_normalization_22/moving_variance/read"/device:GPU:1*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7

1tower_1/resnet_model/batch_normalization_22/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_22Relu:tower_1/resnet_model/batch_normalization_22/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@
£
#tower_1/resnet_model/Pad_4/paddingsConst"/device:GPU:1*9
value0B."                             *
dtype0*
_output_shapes

:
¶
tower_1/resnet_model/Pad_4Padtower_1/resnet_model/Relu_22#tower_1/resnet_model/Pad_4/paddings"/device:GPU:1*
	Tpaddings0*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_26/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

%tower_1/resnet_model/conv2d_26/Conv2DConv2Dtower_1/resnet_model/Pad_4"resnet_model/conv2d_26/kernel/read"/device:GPU:1*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:@*
	dilations
*
T0
Û
:tower_1/resnet_model/batch_normalization_23/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_26/Conv2D.resnet_model/batch_normalization_23/gamma/read-resnet_model/batch_normalization_23/beta/read4resnet_model/batch_normalization_23/moving_mean/read8resnet_model/batch_normalization_23/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_23/ConstConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *d;?
¡
tower_1/resnet_model/Relu_23Relu:tower_1/resnet_model/batch_normalization_23/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_27/dilation_rateConst"/device:GPU:1*
dtype0*
_output_shapes
:*
valueB"      
 
%tower_1/resnet_model/conv2d_27/Conv2DConv2Dtower_1/resnet_model/Relu_23"resnet_model/conv2d_27/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
°
tower_1/resnet_model/add_7Add%tower_1/resnet_model/conv2d_27/Conv2D%tower_1/resnet_model/conv2d_24/Conv2D"/device:GPU:1*'
_output_shapes
:@*
T0
Ð
:tower_1/resnet_model/batch_normalization_24/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_7.resnet_model/batch_normalization_24/gamma/read-resnet_model/batch_normalization_24/beta/read4resnet_model/batch_normalization_24/moving_mean/read8resnet_model/batch_normalization_24/moving_variance/read"/device:GPU:1*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7

1tower_1/resnet_model/batch_normalization_24/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *d;?*
dtype0
¡
tower_1/resnet_model/Relu_24Relu:tower_1/resnet_model/batch_normalization_24/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_28/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_28/Conv2DConv2Dtower_1/resnet_model/Relu_24"resnet_model/conv2d_28/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Û
:tower_1/resnet_model/batch_normalization_25/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_28/Conv2D.resnet_model/batch_normalization_25/gamma/read-resnet_model/batch_normalization_25/beta/read4resnet_model/batch_normalization_25/moving_mean/read8resnet_model/batch_normalization_25/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_25/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_25Relu:tower_1/resnet_model/batch_normalization_25/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_29/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_29/Conv2DConv2Dtower_1/resnet_model/Relu_25"resnet_model/conv2d_29/kernel/read"/device:GPU:1*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0
Û
:tower_1/resnet_model/batch_normalization_26/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_29/Conv2D.resnet_model/batch_normalization_26/gamma/read-resnet_model/batch_normalization_26/beta/read4resnet_model/batch_normalization_26/moving_mean/read8resnet_model/batch_normalization_26/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_26/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_26Relu:tower_1/resnet_model/batch_normalization_26/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_30/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_30/Conv2DConv2Dtower_1/resnet_model/Relu_26"resnet_model/conv2d_30/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
¥
tower_1/resnet_model/add_8Add%tower_1/resnet_model/conv2d_30/Conv2Dtower_1/resnet_model/add_7"/device:GPU:1*
T0*'
_output_shapes
:@
Ð
:tower_1/resnet_model/batch_normalization_27/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_8.resnet_model/batch_normalization_27/gamma/read-resnet_model/batch_normalization_27/beta/read4resnet_model/batch_normalization_27/moving_mean/read8resnet_model/batch_normalization_27/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_27/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_27Relu:tower_1/resnet_model/batch_normalization_27/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_31/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_31/Conv2DConv2Dtower_1/resnet_model/Relu_27"resnet_model/conv2d_31/kernel/read"/device:GPU:1*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
Û
:tower_1/resnet_model/batch_normalization_28/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_31/Conv2D.resnet_model/batch_normalization_28/gamma/read-resnet_model/batch_normalization_28/beta/read4resnet_model/batch_normalization_28/moving_mean/read8resnet_model/batch_normalization_28/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_28/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *d;?*
dtype0
¡
tower_1/resnet_model/Relu_28Relu:tower_1/resnet_model/batch_normalization_28/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_32/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_32/Conv2DConv2Dtower_1/resnet_model/Relu_28"resnet_model/conv2d_32/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Û
:tower_1/resnet_model/batch_normalization_29/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_32/Conv2D.resnet_model/batch_normalization_29/gamma/read-resnet_model/batch_normalization_29/beta/read4resnet_model/batch_normalization_29/moving_mean/read8resnet_model/batch_normalization_29/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_29/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_29Relu:tower_1/resnet_model/batch_normalization_29/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_33/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_33/Conv2DConv2Dtower_1/resnet_model/Relu_29"resnet_model/conv2d_33/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
¥
tower_1/resnet_model/add_9Add%tower_1/resnet_model/conv2d_33/Conv2Dtower_1/resnet_model/add_8"/device:GPU:1*
T0*'
_output_shapes
:@
Ð
:tower_1/resnet_model/batch_normalization_30/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_9.resnet_model/batch_normalization_30/gamma/read-resnet_model/batch_normalization_30/beta/read4resnet_model/batch_normalization_30/moving_mean/read8resnet_model/batch_normalization_30/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_30/ConstConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *d;?
¡
tower_1/resnet_model/Relu_30Relu:tower_1/resnet_model/batch_normalization_30/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_34/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_34/Conv2DConv2Dtower_1/resnet_model/Relu_30"resnet_model/conv2d_34/kernel/read"/device:GPU:1*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
Û
:tower_1/resnet_model/batch_normalization_31/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_34/Conv2D.resnet_model/batch_normalization_31/gamma/read-resnet_model/batch_normalization_31/beta/read4resnet_model/batch_normalization_31/moving_mean/read8resnet_model/batch_normalization_31/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_31/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_31Relu:tower_1/resnet_model/batch_normalization_31/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_35/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_35/Conv2DConv2Dtower_1/resnet_model/Relu_31"resnet_model/conv2d_35/kernel/read"/device:GPU:1*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
Û
:tower_1/resnet_model/batch_normalization_32/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_35/Conv2D.resnet_model/batch_normalization_32/gamma/read-resnet_model/batch_normalization_32/beta/read4resnet_model/batch_normalization_32/moving_mean/read8resnet_model/batch_normalization_32/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_32/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_32Relu:tower_1/resnet_model/batch_normalization_32/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_36/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_36/Conv2DConv2Dtower_1/resnet_model/Relu_32"resnet_model/conv2d_36/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
¦
tower_1/resnet_model/add_10Add%tower_1/resnet_model/conv2d_36/Conv2Dtower_1/resnet_model/add_9"/device:GPU:1*'
_output_shapes
:@*
T0
Ñ
:tower_1/resnet_model/batch_normalization_33/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_10.resnet_model/batch_normalization_33/gamma/read-resnet_model/batch_normalization_33/beta/read4resnet_model/batch_normalization_33/moving_mean/read8resnet_model/batch_normalization_33/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_33/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_33Relu:tower_1/resnet_model/batch_normalization_33/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_37/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_37/Conv2DConv2Dtower_1/resnet_model/Relu_33"resnet_model/conv2d_37/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME
Û
:tower_1/resnet_model/batch_normalization_34/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_37/Conv2D.resnet_model/batch_normalization_34/gamma/read-resnet_model/batch_normalization_34/beta/read4resnet_model/batch_normalization_34/moving_mean/read8resnet_model/batch_normalization_34/moving_variance/read"/device:GPU:1*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7

1tower_1/resnet_model/batch_normalization_34/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_34Relu:tower_1/resnet_model/batch_normalization_34/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_38/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_38/Conv2DConv2Dtower_1/resnet_model/Relu_34"resnet_model/conv2d_38/kernel/read"/device:GPU:1*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
Û
:tower_1/resnet_model/batch_normalization_35/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_38/Conv2D.resnet_model/batch_normalization_35/gamma/read-resnet_model/batch_normalization_35/beta/read4resnet_model/batch_normalization_35/moving_mean/read8resnet_model/batch_normalization_35/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_35/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *d;?*
dtype0
¡
tower_1/resnet_model/Relu_35Relu:tower_1/resnet_model/batch_normalization_35/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_39/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_39/Conv2DConv2Dtower_1/resnet_model/Relu_35"resnet_model/conv2d_39/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME
§
tower_1/resnet_model/add_11Add%tower_1/resnet_model/conv2d_39/Conv2Dtower_1/resnet_model/add_10"/device:GPU:1*'
_output_shapes
:@*
T0
Ñ
:tower_1/resnet_model/batch_normalization_36/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_11.resnet_model/batch_normalization_36/gamma/read-resnet_model/batch_normalization_36/beta/read4resnet_model/batch_normalization_36/moving_mean/read8resnet_model/batch_normalization_36/moving_variance/read"/device:GPU:1*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW

1tower_1/resnet_model/batch_normalization_36/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_36Relu:tower_1/resnet_model/batch_normalization_36/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_40/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_40/Conv2DConv2Dtower_1/resnet_model/Relu_36"resnet_model/conv2d_40/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
Û
:tower_1/resnet_model/batch_normalization_37/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_40/Conv2D.resnet_model/batch_normalization_37/gamma/read-resnet_model/batch_normalization_37/beta/read4resnet_model/batch_normalization_37/moving_mean/read8resnet_model/batch_normalization_37/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_37/ConstConst"/device:GPU:1*
dtype0*
_output_shapes
: *
valueB
 *d;?
¡
tower_1/resnet_model/Relu_37Relu:tower_1/resnet_model/batch_normalization_37/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_41/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_41/Conv2DConv2Dtower_1/resnet_model/Relu_37"resnet_model/conv2d_41/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Û
:tower_1/resnet_model/batch_normalization_38/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_41/Conv2D.resnet_model/batch_normalization_38/gamma/read-resnet_model/batch_normalization_38/beta/read4resnet_model/batch_normalization_38/moving_mean/read8resnet_model/batch_normalization_38/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_38/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_38Relu:tower_1/resnet_model/batch_normalization_38/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_42/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_42/Conv2DConv2Dtower_1/resnet_model/Relu_38"resnet_model/conv2d_42/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(
§
tower_1/resnet_model/add_12Add%tower_1/resnet_model/conv2d_42/Conv2Dtower_1/resnet_model/add_11"/device:GPU:1*
T0*'
_output_shapes
:@

!tower_1/resnet_model/block_layer3Identitytower_1/resnet_model/add_12"/device:GPU:1*'
_output_shapes
:@*
T0
×
:tower_1/resnet_model/batch_normalization_39/FusedBatchNormFusedBatchNorm!tower_1/resnet_model/block_layer3.resnet_model/batch_normalization_39/gamma/read-resnet_model/batch_normalization_39/beta/read4resnet_model/batch_normalization_39/moving_mean/read8resnet_model/batch_normalization_39/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_39/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *d;?*
dtype0
¡
tower_1/resnet_model/Relu_39Relu:tower_1/resnet_model/batch_normalization_39/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0
£
#tower_1/resnet_model/Pad_5/paddingsConst"/device:GPU:1*
_output_shapes

:*9
value0B."                                 *
dtype0
¶
tower_1/resnet_model/Pad_5Padtower_1/resnet_model/Relu_39#tower_1/resnet_model/Pad_5/paddings"/device:GPU:1*
	Tpaddings0*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_43/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

%tower_1/resnet_model/conv2d_43/Conv2DConv2Dtower_1/resnet_model/Pad_5"resnet_model/conv2d_43/kernel/read"/device:GPU:1*
paddingVALID*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(

,tower_1/resnet_model/conv2d_44/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_44/Conv2DConv2Dtower_1/resnet_model/Relu_39"resnet_model/conv2d_44/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(
Û
:tower_1/resnet_model/batch_normalization_40/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_44/Conv2D.resnet_model/batch_normalization_40/gamma/read-resnet_model/batch_normalization_40/beta/read4resnet_model/batch_normalization_40/moving_mean/read8resnet_model/batch_normalization_40/moving_variance/read"/device:GPU:1*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW

1tower_1/resnet_model/batch_normalization_40/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_40Relu:tower_1/resnet_model/batch_normalization_40/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0
£
#tower_1/resnet_model/Pad_6/paddingsConst"/device:GPU:1*9
value0B."                             *
dtype0*
_output_shapes

:
¶
tower_1/resnet_model/Pad_6Padtower_1/resnet_model/Relu_40#tower_1/resnet_model/Pad_6/paddings"/device:GPU:1*
T0*
	Tpaddings0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_45/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:

%tower_1/resnet_model/conv2d_45/Conv2DConv2Dtower_1/resnet_model/Pad_6"resnet_model/conv2d_45/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingVALID
Û
:tower_1/resnet_model/batch_normalization_41/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_45/Conv2D.resnet_model/batch_normalization_41/gamma/read-resnet_model/batch_normalization_41/beta/read4resnet_model/batch_normalization_41/moving_mean/read8resnet_model/batch_normalization_41/moving_variance/read"/device:GPU:1*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW

1tower_1/resnet_model/batch_normalization_41/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_41Relu:tower_1/resnet_model/batch_normalization_41/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_46/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_46/Conv2DConv2Dtower_1/resnet_model/Relu_41"resnet_model/conv2d_46/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
±
tower_1/resnet_model/add_13Add%tower_1/resnet_model/conv2d_46/Conv2D%tower_1/resnet_model/conv2d_43/Conv2D"/device:GPU:1*
T0*'
_output_shapes
:@
Ñ
:tower_1/resnet_model/batch_normalization_42/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_13.resnet_model/batch_normalization_42/gamma/read-resnet_model/batch_normalization_42/beta/read4resnet_model/batch_normalization_42/moving_mean/read8resnet_model/batch_normalization_42/moving_variance/read"/device:GPU:1*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7

1tower_1/resnet_model/batch_normalization_42/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_42Relu:tower_1/resnet_model/batch_normalization_42/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_47/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_47/Conv2DConv2Dtower_1/resnet_model/Relu_42"resnet_model/conv2d_47/kernel/read"/device:GPU:1*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(*
paddingSAME
Û
:tower_1/resnet_model/batch_normalization_43/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_47/Conv2D.resnet_model/batch_normalization_43/gamma/read-resnet_model/batch_normalization_43/beta/read4resnet_model/batch_normalization_43/moving_mean/read8resnet_model/batch_normalization_43/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_43/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_43Relu:tower_1/resnet_model/batch_normalization_43/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_48/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_48/Conv2DConv2Dtower_1/resnet_model/Relu_43"resnet_model/conv2d_48/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
strides
*
data_formatNCHW*
use_cudnn_on_gpu(
Û
:tower_1/resnet_model/batch_normalization_44/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_48/Conv2D.resnet_model/batch_normalization_44/gamma/read-resnet_model/batch_normalization_44/beta/read4resnet_model/batch_normalization_44/moving_mean/read8resnet_model/batch_normalization_44/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_44/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_44Relu:tower_1/resnet_model/batch_normalization_44/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_49/dilation_rateConst"/device:GPU:1*
_output_shapes
:*
valueB"      *
dtype0
 
%tower_1/resnet_model/conv2d_49/Conv2DConv2Dtower_1/resnet_model/Relu_44"resnet_model/conv2d_49/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
§
tower_1/resnet_model/add_14Add%tower_1/resnet_model/conv2d_49/Conv2Dtower_1/resnet_model/add_13"/device:GPU:1*
T0*'
_output_shapes
:@
Ñ
:tower_1/resnet_model/batch_normalization_45/FusedBatchNormFusedBatchNormtower_1/resnet_model/add_14.resnet_model/batch_normalization_45/gamma/read-resnet_model/batch_normalization_45/beta/read4resnet_model/batch_normalization_45/moving_mean/read8resnet_model/batch_normalization_45/moving_variance/read"/device:GPU:1*
epsilon%ð'7*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( 

1tower_1/resnet_model/batch_normalization_45/ConstConst"/device:GPU:1*
_output_shapes
: *
valueB
 *d;?*
dtype0
¡
tower_1/resnet_model/Relu_45Relu:tower_1/resnet_model/batch_normalization_45/FusedBatchNorm"/device:GPU:1*'
_output_shapes
:@*
T0

,tower_1/resnet_model/conv2d_50/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_50/Conv2DConv2Dtower_1/resnet_model/Relu_45"resnet_model/conv2d_50/kernel/read"/device:GPU:1*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides

Û
:tower_1/resnet_model/batch_normalization_46/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_50/Conv2D.resnet_model/batch_normalization_46/gamma/read-resnet_model/batch_normalization_46/beta/read4resnet_model/batch_normalization_46/moving_mean/read8resnet_model/batch_normalization_46/moving_variance/read"/device:GPU:1*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0

1tower_1/resnet_model/batch_normalization_46/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_46Relu:tower_1/resnet_model/batch_normalization_46/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_51/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_51/Conv2DConv2Dtower_1/resnet_model/Relu_46"resnet_model/conv2d_51/kernel/read"/device:GPU:1*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@
Û
:tower_1/resnet_model/batch_normalization_47/FusedBatchNormFusedBatchNorm%tower_1/resnet_model/conv2d_51/Conv2D.resnet_model/batch_normalization_47/gamma/read-resnet_model/batch_normalization_47/beta/read4resnet_model/batch_normalization_47/moving_mean/read8resnet_model/batch_normalization_47/moving_variance/read"/device:GPU:1*
T0*
data_formatNCHW*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7

1tower_1/resnet_model/batch_normalization_47/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_47Relu:tower_1/resnet_model/batch_normalization_47/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

,tower_1/resnet_model/conv2d_52/dilation_rateConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
 
%tower_1/resnet_model/conv2d_52/Conv2DConv2Dtower_1/resnet_model/Relu_47"resnet_model/conv2d_52/kernel/read"/device:GPU:1*
paddingSAME*'
_output_shapes
:@*
	dilations
*
T0*
data_formatNCHW*
strides
*
use_cudnn_on_gpu(
§
tower_1/resnet_model/add_15Add%tower_1/resnet_model/conv2d_52/Conv2Dtower_1/resnet_model/add_14"/device:GPU:1*
T0*'
_output_shapes
:@

!tower_1/resnet_model/block_layer4Identitytower_1/resnet_model/add_15"/device:GPU:1*
T0*'
_output_shapes
:@
×
:tower_1/resnet_model/batch_normalization_48/FusedBatchNormFusedBatchNorm!tower_1/resnet_model/block_layer4.resnet_model/batch_normalization_48/gamma/read-resnet_model/batch_normalization_48/beta/read4resnet_model/batch_normalization_48/moving_mean/read8resnet_model/batch_normalization_48/moving_variance/read"/device:GPU:1*C
_output_shapes1
/:@::::*
is_training( *
epsilon%ð'7*
T0*
data_formatNCHW

1tower_1/resnet_model/batch_normalization_48/ConstConst"/device:GPU:1*
valueB
 *d;?*
dtype0*
_output_shapes
: 
¡
tower_1/resnet_model/Relu_48Relu:tower_1/resnet_model/batch_normalization_48/FusedBatchNorm"/device:GPU:1*
T0*'
_output_shapes
:@

+tower_1/resnet_model/Mean/reduction_indicesConst"/device:GPU:1*
valueB"      *
dtype0*
_output_shapes
:
Ê
tower_1/resnet_model/MeanMeantower_1/resnet_model/Relu_48+tower_1/resnet_model/Mean/reduction_indices"/device:GPU:1*
T0*'
_output_shapes
:@*
	keep_dims(*

Tidx0

&tower_1/resnet_model/final_reduce_meanIdentitytower_1/resnet_model/Mean"/device:GPU:1*
T0*'
_output_shapes
:@

"tower_1/resnet_model/Reshape/shapeConst"/device:GPU:1*
valueB"ÿÿÿÿ   *
dtype0*
_output_shapes
:
º
tower_1/resnet_model/ReshapeReshape&tower_1/resnet_model/final_reduce_mean"tower_1/resnet_model/Reshape/shape"/device:GPU:1*
T0*
Tshape0*
_output_shapes
:	@
È
!tower_1/resnet_model/dense/MatMulMatMultower_1/resnet_model/Reshaperesnet_model/dense/kernel/read"/device:GPU:1*
T0*
_output_shapes
:	@é*
transpose_a( *
transpose_b( 
¾
"tower_1/resnet_model/dense/BiasAddBiasAdd!tower_1/resnet_model/dense/MatMulresnet_model/dense/bias/read"/device:GPU:1*
T0*
data_formatNHWC*
_output_shapes
:	@é

 tower_1/resnet_model/final_denseIdentity"tower_1/resnet_model/dense/BiasAdd"/device:GPU:1*
T0*
_output_shapes
:	@é
i
tower_1/ArgMax/dimensionConst"/device:GPU:1*
value	B :*
dtype0*
_output_shapes
: 
§
tower_1/ArgMaxArgMax tower_1/resnet_model/final_densetower_1/ArgMax/dimension"/device:GPU:1*

Tidx0*
T0*
output_type0	*
_output_shapes
:@
|
tower_1/softmax_tensorSoftmax tower_1/resnet_model/final_dense"/device:GPU:1*
_output_shapes
:	@é*
T0
]
classes/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

classesConcatV2ArgMaxtower_1/ArgMaxclasses/axis"/device:CPU:0*
T0	*
N*
_output_shapes	
:*

Tidx0
c
probabilities/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
¤
probabilitiesConcatV2softmax_tensortower_1/softmax_tensorprobabilities/axis"/device:CPU:0*
T0*
N* 
_output_shapes
:
é*

Tidx0
_
classes_1/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

	classes_1ConcatV2ArgMaxtower_1/ArgMaxclasses_1/axis"/device:CPU:0*
N*
_output_shapes	
:*

Tidx0*
T0	
e
probabilities_1/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
¨
probabilities_1ConcatV2softmax_tensortower_1/softmax_tensorprobabilities_1/axis"/device:CPU:0*
N* 
_output_shapes
:
é*

Tidx0*
T0
_
classes_2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

	classes_2ConcatV2ArgMaxtower_1/ArgMaxclasses_2/axis"/device:CPU:0*
T0	*
N*
_output_shapes	
:*

Tidx0
e
probabilities_2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
¨
probabilities_2ConcatV2softmax_tensortower_1/softmax_tensorprobabilities_2/axis"/device:CPU:0*

Tidx0*
T0*
N* 
_output_shapes
:
é
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_96d84af9c8ca4623b6c257cd16865bfe/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
{
save/SaveV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
dtype0*
_output_shapes
:
t
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 

save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
£U
save/SaveV2_1/tensor_namesConst"/device:CPU:0*ÄT
valueºTB·TûB%resnet_model/batch_normalization/betaB&resnet_model/batch_normalization/gammaB,resnet_model/batch_normalization/moving_meanB0resnet_model/batch_normalization/moving_varianceB'resnet_model/batch_normalization_1/betaB(resnet_model/batch_normalization_1/gammaB.resnet_model/batch_normalization_1/moving_meanB2resnet_model/batch_normalization_1/moving_varianceB(resnet_model/batch_normalization_10/betaB)resnet_model/batch_normalization_10/gammaB/resnet_model/batch_normalization_10/moving_meanB3resnet_model/batch_normalization_10/moving_varianceB(resnet_model/batch_normalization_11/betaB)resnet_model/batch_normalization_11/gammaB/resnet_model/batch_normalization_11/moving_meanB3resnet_model/batch_normalization_11/moving_varianceB(resnet_model/batch_normalization_12/betaB)resnet_model/batch_normalization_12/gammaB/resnet_model/batch_normalization_12/moving_meanB3resnet_model/batch_normalization_12/moving_varianceB(resnet_model/batch_normalization_13/betaB)resnet_model/batch_normalization_13/gammaB/resnet_model/batch_normalization_13/moving_meanB3resnet_model/batch_normalization_13/moving_varianceB(resnet_model/batch_normalization_14/betaB)resnet_model/batch_normalization_14/gammaB/resnet_model/batch_normalization_14/moving_meanB3resnet_model/batch_normalization_14/moving_varianceB(resnet_model/batch_normalization_15/betaB)resnet_model/batch_normalization_15/gammaB/resnet_model/batch_normalization_15/moving_meanB3resnet_model/batch_normalization_15/moving_varianceB(resnet_model/batch_normalization_16/betaB)resnet_model/batch_normalization_16/gammaB/resnet_model/batch_normalization_16/moving_meanB3resnet_model/batch_normalization_16/moving_varianceB(resnet_model/batch_normalization_17/betaB)resnet_model/batch_normalization_17/gammaB/resnet_model/batch_normalization_17/moving_meanB3resnet_model/batch_normalization_17/moving_varianceB(resnet_model/batch_normalization_18/betaB)resnet_model/batch_normalization_18/gammaB/resnet_model/batch_normalization_18/moving_meanB3resnet_model/batch_normalization_18/moving_varianceB(resnet_model/batch_normalization_19/betaB)resnet_model/batch_normalization_19/gammaB/resnet_model/batch_normalization_19/moving_meanB3resnet_model/batch_normalization_19/moving_varianceB'resnet_model/batch_normalization_2/betaB(resnet_model/batch_normalization_2/gammaB.resnet_model/batch_normalization_2/moving_meanB2resnet_model/batch_normalization_2/moving_varianceB(resnet_model/batch_normalization_20/betaB)resnet_model/batch_normalization_20/gammaB/resnet_model/batch_normalization_20/moving_meanB3resnet_model/batch_normalization_20/moving_varianceB(resnet_model/batch_normalization_21/betaB)resnet_model/batch_normalization_21/gammaB/resnet_model/batch_normalization_21/moving_meanB3resnet_model/batch_normalization_21/moving_varianceB(resnet_model/batch_normalization_22/betaB)resnet_model/batch_normalization_22/gammaB/resnet_model/batch_normalization_22/moving_meanB3resnet_model/batch_normalization_22/moving_varianceB(resnet_model/batch_normalization_23/betaB)resnet_model/batch_normalization_23/gammaB/resnet_model/batch_normalization_23/moving_meanB3resnet_model/batch_normalization_23/moving_varianceB(resnet_model/batch_normalization_24/betaB)resnet_model/batch_normalization_24/gammaB/resnet_model/batch_normalization_24/moving_meanB3resnet_model/batch_normalization_24/moving_varianceB(resnet_model/batch_normalization_25/betaB)resnet_model/batch_normalization_25/gammaB/resnet_model/batch_normalization_25/moving_meanB3resnet_model/batch_normalization_25/moving_varianceB(resnet_model/batch_normalization_26/betaB)resnet_model/batch_normalization_26/gammaB/resnet_model/batch_normalization_26/moving_meanB3resnet_model/batch_normalization_26/moving_varianceB(resnet_model/batch_normalization_27/betaB)resnet_model/batch_normalization_27/gammaB/resnet_model/batch_normalization_27/moving_meanB3resnet_model/batch_normalization_27/moving_varianceB(resnet_model/batch_normalization_28/betaB)resnet_model/batch_normalization_28/gammaB/resnet_model/batch_normalization_28/moving_meanB3resnet_model/batch_normalization_28/moving_varianceB(resnet_model/batch_normalization_29/betaB)resnet_model/batch_normalization_29/gammaB/resnet_model/batch_normalization_29/moving_meanB3resnet_model/batch_normalization_29/moving_varianceB'resnet_model/batch_normalization_3/betaB(resnet_model/batch_normalization_3/gammaB.resnet_model/batch_normalization_3/moving_meanB2resnet_model/batch_normalization_3/moving_varianceB(resnet_model/batch_normalization_30/betaB)resnet_model/batch_normalization_30/gammaB/resnet_model/batch_normalization_30/moving_meanB3resnet_model/batch_normalization_30/moving_varianceB(resnet_model/batch_normalization_31/betaB)resnet_model/batch_normalization_31/gammaB/resnet_model/batch_normalization_31/moving_meanB3resnet_model/batch_normalization_31/moving_varianceB(resnet_model/batch_normalization_32/betaB)resnet_model/batch_normalization_32/gammaB/resnet_model/batch_normalization_32/moving_meanB3resnet_model/batch_normalization_32/moving_varianceB(resnet_model/batch_normalization_33/betaB)resnet_model/batch_normalization_33/gammaB/resnet_model/batch_normalization_33/moving_meanB3resnet_model/batch_normalization_33/moving_varianceB(resnet_model/batch_normalization_34/betaB)resnet_model/batch_normalization_34/gammaB/resnet_model/batch_normalization_34/moving_meanB3resnet_model/batch_normalization_34/moving_varianceB(resnet_model/batch_normalization_35/betaB)resnet_model/batch_normalization_35/gammaB/resnet_model/batch_normalization_35/moving_meanB3resnet_model/batch_normalization_35/moving_varianceB(resnet_model/batch_normalization_36/betaB)resnet_model/batch_normalization_36/gammaB/resnet_model/batch_normalization_36/moving_meanB3resnet_model/batch_normalization_36/moving_varianceB(resnet_model/batch_normalization_37/betaB)resnet_model/batch_normalization_37/gammaB/resnet_model/batch_normalization_37/moving_meanB3resnet_model/batch_normalization_37/moving_varianceB(resnet_model/batch_normalization_38/betaB)resnet_model/batch_normalization_38/gammaB/resnet_model/batch_normalization_38/moving_meanB3resnet_model/batch_normalization_38/moving_varianceB(resnet_model/batch_normalization_39/betaB)resnet_model/batch_normalization_39/gammaB/resnet_model/batch_normalization_39/moving_meanB3resnet_model/batch_normalization_39/moving_varianceB'resnet_model/batch_normalization_4/betaB(resnet_model/batch_normalization_4/gammaB.resnet_model/batch_normalization_4/moving_meanB2resnet_model/batch_normalization_4/moving_varianceB(resnet_model/batch_normalization_40/betaB)resnet_model/batch_normalization_40/gammaB/resnet_model/batch_normalization_40/moving_meanB3resnet_model/batch_normalization_40/moving_varianceB(resnet_model/batch_normalization_41/betaB)resnet_model/batch_normalization_41/gammaB/resnet_model/batch_normalization_41/moving_meanB3resnet_model/batch_normalization_41/moving_varianceB(resnet_model/batch_normalization_42/betaB)resnet_model/batch_normalization_42/gammaB/resnet_model/batch_normalization_42/moving_meanB3resnet_model/batch_normalization_42/moving_varianceB(resnet_model/batch_normalization_43/betaB)resnet_model/batch_normalization_43/gammaB/resnet_model/batch_normalization_43/moving_meanB3resnet_model/batch_normalization_43/moving_varianceB(resnet_model/batch_normalization_44/betaB)resnet_model/batch_normalization_44/gammaB/resnet_model/batch_normalization_44/moving_meanB3resnet_model/batch_normalization_44/moving_varianceB(resnet_model/batch_normalization_45/betaB)resnet_model/batch_normalization_45/gammaB/resnet_model/batch_normalization_45/moving_meanB3resnet_model/batch_normalization_45/moving_varianceB(resnet_model/batch_normalization_46/betaB)resnet_model/batch_normalization_46/gammaB/resnet_model/batch_normalization_46/moving_meanB3resnet_model/batch_normalization_46/moving_varianceB(resnet_model/batch_normalization_47/betaB)resnet_model/batch_normalization_47/gammaB/resnet_model/batch_normalization_47/moving_meanB3resnet_model/batch_normalization_47/moving_varianceB(resnet_model/batch_normalization_48/betaB)resnet_model/batch_normalization_48/gammaB/resnet_model/batch_normalization_48/moving_meanB3resnet_model/batch_normalization_48/moving_varianceB'resnet_model/batch_normalization_5/betaB(resnet_model/batch_normalization_5/gammaB.resnet_model/batch_normalization_5/moving_meanB2resnet_model/batch_normalization_5/moving_varianceB'resnet_model/batch_normalization_6/betaB(resnet_model/batch_normalization_6/gammaB.resnet_model/batch_normalization_6/moving_meanB2resnet_model/batch_normalization_6/moving_varianceB'resnet_model/batch_normalization_7/betaB(resnet_model/batch_normalization_7/gammaB.resnet_model/batch_normalization_7/moving_meanB2resnet_model/batch_normalization_7/moving_varianceB'resnet_model/batch_normalization_8/betaB(resnet_model/batch_normalization_8/gammaB.resnet_model/batch_normalization_8/moving_meanB2resnet_model/batch_normalization_8/moving_varianceB'resnet_model/batch_normalization_9/betaB(resnet_model/batch_normalization_9/gammaB.resnet_model/batch_normalization_9/moving_meanB2resnet_model/batch_normalization_9/moving_varianceBresnet_model/conv2d/kernelBresnet_model/conv2d_1/kernelBresnet_model/conv2d_10/kernelBresnet_model/conv2d_11/kernelBresnet_model/conv2d_12/kernelBresnet_model/conv2d_13/kernelBresnet_model/conv2d_14/kernelBresnet_model/conv2d_15/kernelBresnet_model/conv2d_16/kernelBresnet_model/conv2d_17/kernelBresnet_model/conv2d_18/kernelBresnet_model/conv2d_19/kernelBresnet_model/conv2d_2/kernelBresnet_model/conv2d_20/kernelBresnet_model/conv2d_21/kernelBresnet_model/conv2d_22/kernelBresnet_model/conv2d_23/kernelBresnet_model/conv2d_24/kernelBresnet_model/conv2d_25/kernelBresnet_model/conv2d_26/kernelBresnet_model/conv2d_27/kernelBresnet_model/conv2d_28/kernelBresnet_model/conv2d_29/kernelBresnet_model/conv2d_3/kernelBresnet_model/conv2d_30/kernelBresnet_model/conv2d_31/kernelBresnet_model/conv2d_32/kernelBresnet_model/conv2d_33/kernelBresnet_model/conv2d_34/kernelBresnet_model/conv2d_35/kernelBresnet_model/conv2d_36/kernelBresnet_model/conv2d_37/kernelBresnet_model/conv2d_38/kernelBresnet_model/conv2d_39/kernelBresnet_model/conv2d_4/kernelBresnet_model/conv2d_40/kernelBresnet_model/conv2d_41/kernelBresnet_model/conv2d_42/kernelBresnet_model/conv2d_43/kernelBresnet_model/conv2d_44/kernelBresnet_model/conv2d_45/kernelBresnet_model/conv2d_46/kernelBresnet_model/conv2d_47/kernelBresnet_model/conv2d_48/kernelBresnet_model/conv2d_49/kernelBresnet_model/conv2d_5/kernelBresnet_model/conv2d_50/kernelBresnet_model/conv2d_51/kernelBresnet_model/conv2d_52/kernelBresnet_model/conv2d_6/kernelBresnet_model/conv2d_7/kernelBresnet_model/conv2d_8/kernelBresnet_model/conv2d_9/kernelBresnet_model/dense/biasBresnet_model/dense/kernel*
dtype0*
_output_shapes	
:û
ï
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueBÿûB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:û
·W
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slices%resnet_model/batch_normalization/beta&resnet_model/batch_normalization/gamma,resnet_model/batch_normalization/moving_mean0resnet_model/batch_normalization/moving_variance'resnet_model/batch_normalization_1/beta(resnet_model/batch_normalization_1/gamma.resnet_model/batch_normalization_1/moving_mean2resnet_model/batch_normalization_1/moving_variance(resnet_model/batch_normalization_10/beta)resnet_model/batch_normalization_10/gamma/resnet_model/batch_normalization_10/moving_mean3resnet_model/batch_normalization_10/moving_variance(resnet_model/batch_normalization_11/beta)resnet_model/batch_normalization_11/gamma/resnet_model/batch_normalization_11/moving_mean3resnet_model/batch_normalization_11/moving_variance(resnet_model/batch_normalization_12/beta)resnet_model/batch_normalization_12/gamma/resnet_model/batch_normalization_12/moving_mean3resnet_model/batch_normalization_12/moving_variance(resnet_model/batch_normalization_13/beta)resnet_model/batch_normalization_13/gamma/resnet_model/batch_normalization_13/moving_mean3resnet_model/batch_normalization_13/moving_variance(resnet_model/batch_normalization_14/beta)resnet_model/batch_normalization_14/gamma/resnet_model/batch_normalization_14/moving_mean3resnet_model/batch_normalization_14/moving_variance(resnet_model/batch_normalization_15/beta)resnet_model/batch_normalization_15/gamma/resnet_model/batch_normalization_15/moving_mean3resnet_model/batch_normalization_15/moving_variance(resnet_model/batch_normalization_16/beta)resnet_model/batch_normalization_16/gamma/resnet_model/batch_normalization_16/moving_mean3resnet_model/batch_normalization_16/moving_variance(resnet_model/batch_normalization_17/beta)resnet_model/batch_normalization_17/gamma/resnet_model/batch_normalization_17/moving_mean3resnet_model/batch_normalization_17/moving_variance(resnet_model/batch_normalization_18/beta)resnet_model/batch_normalization_18/gamma/resnet_model/batch_normalization_18/moving_mean3resnet_model/batch_normalization_18/moving_variance(resnet_model/batch_normalization_19/beta)resnet_model/batch_normalization_19/gamma/resnet_model/batch_normalization_19/moving_mean3resnet_model/batch_normalization_19/moving_variance'resnet_model/batch_normalization_2/beta(resnet_model/batch_normalization_2/gamma.resnet_model/batch_normalization_2/moving_mean2resnet_model/batch_normalization_2/moving_variance(resnet_model/batch_normalization_20/beta)resnet_model/batch_normalization_20/gamma/resnet_model/batch_normalization_20/moving_mean3resnet_model/batch_normalization_20/moving_variance(resnet_model/batch_normalization_21/beta)resnet_model/batch_normalization_21/gamma/resnet_model/batch_normalization_21/moving_mean3resnet_model/batch_normalization_21/moving_variance(resnet_model/batch_normalization_22/beta)resnet_model/batch_normalization_22/gamma/resnet_model/batch_normalization_22/moving_mean3resnet_model/batch_normalization_22/moving_variance(resnet_model/batch_normalization_23/beta)resnet_model/batch_normalization_23/gamma/resnet_model/batch_normalization_23/moving_mean3resnet_model/batch_normalization_23/moving_variance(resnet_model/batch_normalization_24/beta)resnet_model/batch_normalization_24/gamma/resnet_model/batch_normalization_24/moving_mean3resnet_model/batch_normalization_24/moving_variance(resnet_model/batch_normalization_25/beta)resnet_model/batch_normalization_25/gamma/resnet_model/batch_normalization_25/moving_mean3resnet_model/batch_normalization_25/moving_variance(resnet_model/batch_normalization_26/beta)resnet_model/batch_normalization_26/gamma/resnet_model/batch_normalization_26/moving_mean3resnet_model/batch_normalization_26/moving_variance(resnet_model/batch_normalization_27/beta)resnet_model/batch_normalization_27/gamma/resnet_model/batch_normalization_27/moving_mean3resnet_model/batch_normalization_27/moving_variance(resnet_model/batch_normalization_28/beta)resnet_model/batch_normalization_28/gamma/resnet_model/batch_normalization_28/moving_mean3resnet_model/batch_normalization_28/moving_variance(resnet_model/batch_normalization_29/beta)resnet_model/batch_normalization_29/gamma/resnet_model/batch_normalization_29/moving_mean3resnet_model/batch_normalization_29/moving_variance'resnet_model/batch_normalization_3/beta(resnet_model/batch_normalization_3/gamma.resnet_model/batch_normalization_3/moving_mean2resnet_model/batch_normalization_3/moving_variance(resnet_model/batch_normalization_30/beta)resnet_model/batch_normalization_30/gamma/resnet_model/batch_normalization_30/moving_mean3resnet_model/batch_normalization_30/moving_variance(resnet_model/batch_normalization_31/beta)resnet_model/batch_normalization_31/gamma/resnet_model/batch_normalization_31/moving_mean3resnet_model/batch_normalization_31/moving_variance(resnet_model/batch_normalization_32/beta)resnet_model/batch_normalization_32/gamma/resnet_model/batch_normalization_32/moving_mean3resnet_model/batch_normalization_32/moving_variance(resnet_model/batch_normalization_33/beta)resnet_model/batch_normalization_33/gamma/resnet_model/batch_normalization_33/moving_mean3resnet_model/batch_normalization_33/moving_variance(resnet_model/batch_normalization_34/beta)resnet_model/batch_normalization_34/gamma/resnet_model/batch_normalization_34/moving_mean3resnet_model/batch_normalization_34/moving_variance(resnet_model/batch_normalization_35/beta)resnet_model/batch_normalization_35/gamma/resnet_model/batch_normalization_35/moving_mean3resnet_model/batch_normalization_35/moving_variance(resnet_model/batch_normalization_36/beta)resnet_model/batch_normalization_36/gamma/resnet_model/batch_normalization_36/moving_mean3resnet_model/batch_normalization_36/moving_variance(resnet_model/batch_normalization_37/beta)resnet_model/batch_normalization_37/gamma/resnet_model/batch_normalization_37/moving_mean3resnet_model/batch_normalization_37/moving_variance(resnet_model/batch_normalization_38/beta)resnet_model/batch_normalization_38/gamma/resnet_model/batch_normalization_38/moving_mean3resnet_model/batch_normalization_38/moving_variance(resnet_model/batch_normalization_39/beta)resnet_model/batch_normalization_39/gamma/resnet_model/batch_normalization_39/moving_mean3resnet_model/batch_normalization_39/moving_variance'resnet_model/batch_normalization_4/beta(resnet_model/batch_normalization_4/gamma.resnet_model/batch_normalization_4/moving_mean2resnet_model/batch_normalization_4/moving_variance(resnet_model/batch_normalization_40/beta)resnet_model/batch_normalization_40/gamma/resnet_model/batch_normalization_40/moving_mean3resnet_model/batch_normalization_40/moving_variance(resnet_model/batch_normalization_41/beta)resnet_model/batch_normalization_41/gamma/resnet_model/batch_normalization_41/moving_mean3resnet_model/batch_normalization_41/moving_variance(resnet_model/batch_normalization_42/beta)resnet_model/batch_normalization_42/gamma/resnet_model/batch_normalization_42/moving_mean3resnet_model/batch_normalization_42/moving_variance(resnet_model/batch_normalization_43/beta)resnet_model/batch_normalization_43/gamma/resnet_model/batch_normalization_43/moving_mean3resnet_model/batch_normalization_43/moving_variance(resnet_model/batch_normalization_44/beta)resnet_model/batch_normalization_44/gamma/resnet_model/batch_normalization_44/moving_mean3resnet_model/batch_normalization_44/moving_variance(resnet_model/batch_normalization_45/beta)resnet_model/batch_normalization_45/gamma/resnet_model/batch_normalization_45/moving_mean3resnet_model/batch_normalization_45/moving_variance(resnet_model/batch_normalization_46/beta)resnet_model/batch_normalization_46/gamma/resnet_model/batch_normalization_46/moving_mean3resnet_model/batch_normalization_46/moving_variance(resnet_model/batch_normalization_47/beta)resnet_model/batch_normalization_47/gamma/resnet_model/batch_normalization_47/moving_mean3resnet_model/batch_normalization_47/moving_variance(resnet_model/batch_normalization_48/beta)resnet_model/batch_normalization_48/gamma/resnet_model/batch_normalization_48/moving_mean3resnet_model/batch_normalization_48/moving_variance'resnet_model/batch_normalization_5/beta(resnet_model/batch_normalization_5/gamma.resnet_model/batch_normalization_5/moving_mean2resnet_model/batch_normalization_5/moving_variance'resnet_model/batch_normalization_6/beta(resnet_model/batch_normalization_6/gamma.resnet_model/batch_normalization_6/moving_mean2resnet_model/batch_normalization_6/moving_variance'resnet_model/batch_normalization_7/beta(resnet_model/batch_normalization_7/gamma.resnet_model/batch_normalization_7/moving_mean2resnet_model/batch_normalization_7/moving_variance'resnet_model/batch_normalization_8/beta(resnet_model/batch_normalization_8/gamma.resnet_model/batch_normalization_8/moving_mean2resnet_model/batch_normalization_8/moving_variance'resnet_model/batch_normalization_9/beta(resnet_model/batch_normalization_9/gamma.resnet_model/batch_normalization_9/moving_mean2resnet_model/batch_normalization_9/moving_varianceresnet_model/conv2d/kernelresnet_model/conv2d_1/kernelresnet_model/conv2d_10/kernelresnet_model/conv2d_11/kernelresnet_model/conv2d_12/kernelresnet_model/conv2d_13/kernelresnet_model/conv2d_14/kernelresnet_model/conv2d_15/kernelresnet_model/conv2d_16/kernelresnet_model/conv2d_17/kernelresnet_model/conv2d_18/kernelresnet_model/conv2d_19/kernelresnet_model/conv2d_2/kernelresnet_model/conv2d_20/kernelresnet_model/conv2d_21/kernelresnet_model/conv2d_22/kernelresnet_model/conv2d_23/kernelresnet_model/conv2d_24/kernelresnet_model/conv2d_25/kernelresnet_model/conv2d_26/kernelresnet_model/conv2d_27/kernelresnet_model/conv2d_28/kernelresnet_model/conv2d_29/kernelresnet_model/conv2d_3/kernelresnet_model/conv2d_30/kernelresnet_model/conv2d_31/kernelresnet_model/conv2d_32/kernelresnet_model/conv2d_33/kernelresnet_model/conv2d_34/kernelresnet_model/conv2d_35/kernelresnet_model/conv2d_36/kernelresnet_model/conv2d_37/kernelresnet_model/conv2d_38/kernelresnet_model/conv2d_39/kernelresnet_model/conv2d_4/kernelresnet_model/conv2d_40/kernelresnet_model/conv2d_41/kernelresnet_model/conv2d_42/kernelresnet_model/conv2d_43/kernelresnet_model/conv2d_44/kernelresnet_model/conv2d_45/kernelresnet_model/conv2d_46/kernelresnet_model/conv2d_47/kernelresnet_model/conv2d_48/kernelresnet_model/conv2d_49/kernelresnet_model/conv2d_5/kernelresnet_model/conv2d_50/kernelresnet_model/conv2d_51/kernelresnet_model/conv2d_52/kernelresnet_model/conv2d_6/kernelresnet_model/conv2d_7/kernelresnet_model/conv2d_8/kernelresnet_model/conv2d_9/kernelresnet_model/dense/biasresnet_model/dense/kernel"/device:CPU:0*
dtypes
þ2û
¨
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
à
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
¥
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
_output_shapes
: *
T0
~
save/RestoreV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
dtype0*
_output_shapes
:
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*
_output_shapes
:

save/AssignAssignglobal_stepsave/RestoreV2*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(
(
save/restore_shardNoOp^save/Assign
¦U
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*ÄT
valueºTB·TûB%resnet_model/batch_normalization/betaB&resnet_model/batch_normalization/gammaB,resnet_model/batch_normalization/moving_meanB0resnet_model/batch_normalization/moving_varianceB'resnet_model/batch_normalization_1/betaB(resnet_model/batch_normalization_1/gammaB.resnet_model/batch_normalization_1/moving_meanB2resnet_model/batch_normalization_1/moving_varianceB(resnet_model/batch_normalization_10/betaB)resnet_model/batch_normalization_10/gammaB/resnet_model/batch_normalization_10/moving_meanB3resnet_model/batch_normalization_10/moving_varianceB(resnet_model/batch_normalization_11/betaB)resnet_model/batch_normalization_11/gammaB/resnet_model/batch_normalization_11/moving_meanB3resnet_model/batch_normalization_11/moving_varianceB(resnet_model/batch_normalization_12/betaB)resnet_model/batch_normalization_12/gammaB/resnet_model/batch_normalization_12/moving_meanB3resnet_model/batch_normalization_12/moving_varianceB(resnet_model/batch_normalization_13/betaB)resnet_model/batch_normalization_13/gammaB/resnet_model/batch_normalization_13/moving_meanB3resnet_model/batch_normalization_13/moving_varianceB(resnet_model/batch_normalization_14/betaB)resnet_model/batch_normalization_14/gammaB/resnet_model/batch_normalization_14/moving_meanB3resnet_model/batch_normalization_14/moving_varianceB(resnet_model/batch_normalization_15/betaB)resnet_model/batch_normalization_15/gammaB/resnet_model/batch_normalization_15/moving_meanB3resnet_model/batch_normalization_15/moving_varianceB(resnet_model/batch_normalization_16/betaB)resnet_model/batch_normalization_16/gammaB/resnet_model/batch_normalization_16/moving_meanB3resnet_model/batch_normalization_16/moving_varianceB(resnet_model/batch_normalization_17/betaB)resnet_model/batch_normalization_17/gammaB/resnet_model/batch_normalization_17/moving_meanB3resnet_model/batch_normalization_17/moving_varianceB(resnet_model/batch_normalization_18/betaB)resnet_model/batch_normalization_18/gammaB/resnet_model/batch_normalization_18/moving_meanB3resnet_model/batch_normalization_18/moving_varianceB(resnet_model/batch_normalization_19/betaB)resnet_model/batch_normalization_19/gammaB/resnet_model/batch_normalization_19/moving_meanB3resnet_model/batch_normalization_19/moving_varianceB'resnet_model/batch_normalization_2/betaB(resnet_model/batch_normalization_2/gammaB.resnet_model/batch_normalization_2/moving_meanB2resnet_model/batch_normalization_2/moving_varianceB(resnet_model/batch_normalization_20/betaB)resnet_model/batch_normalization_20/gammaB/resnet_model/batch_normalization_20/moving_meanB3resnet_model/batch_normalization_20/moving_varianceB(resnet_model/batch_normalization_21/betaB)resnet_model/batch_normalization_21/gammaB/resnet_model/batch_normalization_21/moving_meanB3resnet_model/batch_normalization_21/moving_varianceB(resnet_model/batch_normalization_22/betaB)resnet_model/batch_normalization_22/gammaB/resnet_model/batch_normalization_22/moving_meanB3resnet_model/batch_normalization_22/moving_varianceB(resnet_model/batch_normalization_23/betaB)resnet_model/batch_normalization_23/gammaB/resnet_model/batch_normalization_23/moving_meanB3resnet_model/batch_normalization_23/moving_varianceB(resnet_model/batch_normalization_24/betaB)resnet_model/batch_normalization_24/gammaB/resnet_model/batch_normalization_24/moving_meanB3resnet_model/batch_normalization_24/moving_varianceB(resnet_model/batch_normalization_25/betaB)resnet_model/batch_normalization_25/gammaB/resnet_model/batch_normalization_25/moving_meanB3resnet_model/batch_normalization_25/moving_varianceB(resnet_model/batch_normalization_26/betaB)resnet_model/batch_normalization_26/gammaB/resnet_model/batch_normalization_26/moving_meanB3resnet_model/batch_normalization_26/moving_varianceB(resnet_model/batch_normalization_27/betaB)resnet_model/batch_normalization_27/gammaB/resnet_model/batch_normalization_27/moving_meanB3resnet_model/batch_normalization_27/moving_varianceB(resnet_model/batch_normalization_28/betaB)resnet_model/batch_normalization_28/gammaB/resnet_model/batch_normalization_28/moving_meanB3resnet_model/batch_normalization_28/moving_varianceB(resnet_model/batch_normalization_29/betaB)resnet_model/batch_normalization_29/gammaB/resnet_model/batch_normalization_29/moving_meanB3resnet_model/batch_normalization_29/moving_varianceB'resnet_model/batch_normalization_3/betaB(resnet_model/batch_normalization_3/gammaB.resnet_model/batch_normalization_3/moving_meanB2resnet_model/batch_normalization_3/moving_varianceB(resnet_model/batch_normalization_30/betaB)resnet_model/batch_normalization_30/gammaB/resnet_model/batch_normalization_30/moving_meanB3resnet_model/batch_normalization_30/moving_varianceB(resnet_model/batch_normalization_31/betaB)resnet_model/batch_normalization_31/gammaB/resnet_model/batch_normalization_31/moving_meanB3resnet_model/batch_normalization_31/moving_varianceB(resnet_model/batch_normalization_32/betaB)resnet_model/batch_normalization_32/gammaB/resnet_model/batch_normalization_32/moving_meanB3resnet_model/batch_normalization_32/moving_varianceB(resnet_model/batch_normalization_33/betaB)resnet_model/batch_normalization_33/gammaB/resnet_model/batch_normalization_33/moving_meanB3resnet_model/batch_normalization_33/moving_varianceB(resnet_model/batch_normalization_34/betaB)resnet_model/batch_normalization_34/gammaB/resnet_model/batch_normalization_34/moving_meanB3resnet_model/batch_normalization_34/moving_varianceB(resnet_model/batch_normalization_35/betaB)resnet_model/batch_normalization_35/gammaB/resnet_model/batch_normalization_35/moving_meanB3resnet_model/batch_normalization_35/moving_varianceB(resnet_model/batch_normalization_36/betaB)resnet_model/batch_normalization_36/gammaB/resnet_model/batch_normalization_36/moving_meanB3resnet_model/batch_normalization_36/moving_varianceB(resnet_model/batch_normalization_37/betaB)resnet_model/batch_normalization_37/gammaB/resnet_model/batch_normalization_37/moving_meanB3resnet_model/batch_normalization_37/moving_varianceB(resnet_model/batch_normalization_38/betaB)resnet_model/batch_normalization_38/gammaB/resnet_model/batch_normalization_38/moving_meanB3resnet_model/batch_normalization_38/moving_varianceB(resnet_model/batch_normalization_39/betaB)resnet_model/batch_normalization_39/gammaB/resnet_model/batch_normalization_39/moving_meanB3resnet_model/batch_normalization_39/moving_varianceB'resnet_model/batch_normalization_4/betaB(resnet_model/batch_normalization_4/gammaB.resnet_model/batch_normalization_4/moving_meanB2resnet_model/batch_normalization_4/moving_varianceB(resnet_model/batch_normalization_40/betaB)resnet_model/batch_normalization_40/gammaB/resnet_model/batch_normalization_40/moving_meanB3resnet_model/batch_normalization_40/moving_varianceB(resnet_model/batch_normalization_41/betaB)resnet_model/batch_normalization_41/gammaB/resnet_model/batch_normalization_41/moving_meanB3resnet_model/batch_normalization_41/moving_varianceB(resnet_model/batch_normalization_42/betaB)resnet_model/batch_normalization_42/gammaB/resnet_model/batch_normalization_42/moving_meanB3resnet_model/batch_normalization_42/moving_varianceB(resnet_model/batch_normalization_43/betaB)resnet_model/batch_normalization_43/gammaB/resnet_model/batch_normalization_43/moving_meanB3resnet_model/batch_normalization_43/moving_varianceB(resnet_model/batch_normalization_44/betaB)resnet_model/batch_normalization_44/gammaB/resnet_model/batch_normalization_44/moving_meanB3resnet_model/batch_normalization_44/moving_varianceB(resnet_model/batch_normalization_45/betaB)resnet_model/batch_normalization_45/gammaB/resnet_model/batch_normalization_45/moving_meanB3resnet_model/batch_normalization_45/moving_varianceB(resnet_model/batch_normalization_46/betaB)resnet_model/batch_normalization_46/gammaB/resnet_model/batch_normalization_46/moving_meanB3resnet_model/batch_normalization_46/moving_varianceB(resnet_model/batch_normalization_47/betaB)resnet_model/batch_normalization_47/gammaB/resnet_model/batch_normalization_47/moving_meanB3resnet_model/batch_normalization_47/moving_varianceB(resnet_model/batch_normalization_48/betaB)resnet_model/batch_normalization_48/gammaB/resnet_model/batch_normalization_48/moving_meanB3resnet_model/batch_normalization_48/moving_varianceB'resnet_model/batch_normalization_5/betaB(resnet_model/batch_normalization_5/gammaB.resnet_model/batch_normalization_5/moving_meanB2resnet_model/batch_normalization_5/moving_varianceB'resnet_model/batch_normalization_6/betaB(resnet_model/batch_normalization_6/gammaB.resnet_model/batch_normalization_6/moving_meanB2resnet_model/batch_normalization_6/moving_varianceB'resnet_model/batch_normalization_7/betaB(resnet_model/batch_normalization_7/gammaB.resnet_model/batch_normalization_7/moving_meanB2resnet_model/batch_normalization_7/moving_varianceB'resnet_model/batch_normalization_8/betaB(resnet_model/batch_normalization_8/gammaB.resnet_model/batch_normalization_8/moving_meanB2resnet_model/batch_normalization_8/moving_varianceB'resnet_model/batch_normalization_9/betaB(resnet_model/batch_normalization_9/gammaB.resnet_model/batch_normalization_9/moving_meanB2resnet_model/batch_normalization_9/moving_varianceBresnet_model/conv2d/kernelBresnet_model/conv2d_1/kernelBresnet_model/conv2d_10/kernelBresnet_model/conv2d_11/kernelBresnet_model/conv2d_12/kernelBresnet_model/conv2d_13/kernelBresnet_model/conv2d_14/kernelBresnet_model/conv2d_15/kernelBresnet_model/conv2d_16/kernelBresnet_model/conv2d_17/kernelBresnet_model/conv2d_18/kernelBresnet_model/conv2d_19/kernelBresnet_model/conv2d_2/kernelBresnet_model/conv2d_20/kernelBresnet_model/conv2d_21/kernelBresnet_model/conv2d_22/kernelBresnet_model/conv2d_23/kernelBresnet_model/conv2d_24/kernelBresnet_model/conv2d_25/kernelBresnet_model/conv2d_26/kernelBresnet_model/conv2d_27/kernelBresnet_model/conv2d_28/kernelBresnet_model/conv2d_29/kernelBresnet_model/conv2d_3/kernelBresnet_model/conv2d_30/kernelBresnet_model/conv2d_31/kernelBresnet_model/conv2d_32/kernelBresnet_model/conv2d_33/kernelBresnet_model/conv2d_34/kernelBresnet_model/conv2d_35/kernelBresnet_model/conv2d_36/kernelBresnet_model/conv2d_37/kernelBresnet_model/conv2d_38/kernelBresnet_model/conv2d_39/kernelBresnet_model/conv2d_4/kernelBresnet_model/conv2d_40/kernelBresnet_model/conv2d_41/kernelBresnet_model/conv2d_42/kernelBresnet_model/conv2d_43/kernelBresnet_model/conv2d_44/kernelBresnet_model/conv2d_45/kernelBresnet_model/conv2d_46/kernelBresnet_model/conv2d_47/kernelBresnet_model/conv2d_48/kernelBresnet_model/conv2d_49/kernelBresnet_model/conv2d_5/kernelBresnet_model/conv2d_50/kernelBresnet_model/conv2d_51/kernelBresnet_model/conv2d_52/kernelBresnet_model/conv2d_6/kernelBresnet_model/conv2d_7/kernelBresnet_model/conv2d_8/kernelBresnet_model/conv2d_9/kernelBresnet_model/dense/biasBresnet_model/dense/kernel*
dtype0*
_output_shapes	
:û
ò
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueBÿûB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:û

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*
_output_shapesï
ì:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
þ2û
ç
save/Assign_1Assign%resnet_model/batch_normalization/betasave/RestoreV2_1"/device:CPU:0*
T0*8
_class.
,*loc:@resnet_model/batch_normalization/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
ë
save/Assign_2Assign&resnet_model/batch_normalization/gammasave/RestoreV2_1:1"/device:CPU:0*
use_locking(*
T0*9
_class/
-+loc:@resnet_model/batch_normalization/gamma*
validate_shape(*
_output_shapes
:@
÷
save/Assign_3Assign,resnet_model/batch_normalization/moving_meansave/RestoreV2_1:2"/device:CPU:0*
T0*?
_class5
31loc:@resnet_model/batch_normalization/moving_mean*
validate_shape(*
_output_shapes
:@*
use_locking(
ÿ
save/Assign_4Assign0resnet_model/batch_normalization/moving_variancesave/RestoreV2_1:3"/device:CPU:0*
use_locking(*
T0*C
_class9
75loc:@resnet_model/batch_normalization/moving_variance*
validate_shape(*
_output_shapes
:@
í
save/Assign_5Assign'resnet_model/batch_normalization_1/betasave/RestoreV2_1:4"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_1/beta*
validate_shape(*
_output_shapes
:@
ï
save/Assign_6Assign(resnet_model/batch_normalization_1/gammasave/RestoreV2_1:5"/device:CPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_1/gamma
û
save/Assign_7Assign.resnet_model/batch_normalization_1/moving_meansave/RestoreV2_1:6"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
:@

save/Assign_8Assign2resnet_model/batch_normalization_1/moving_variancesave/RestoreV2_1:7"/device:CPU:0*E
_class;
97loc:@resnet_model/batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
ð
save/Assign_9Assign(resnet_model/batch_normalization_10/betasave/RestoreV2_1:8"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_10/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ó
save/Assign_10Assign)resnet_model/batch_normalization_10/gammasave/RestoreV2_1:9"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_10/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_11Assign/resnet_model/batch_normalization_10/moving_meansave/RestoreV2_1:10"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_10/moving_mean*
validate_shape(

save/Assign_12Assign3resnet_model/batch_normalization_10/moving_variancesave/RestoreV2_1:11"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_10/moving_variance*
validate_shape(*
_output_shapes	
:
ò
save/Assign_13Assign(resnet_model/batch_normalization_11/betasave/RestoreV2_1:12"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_11/beta*
validate_shape(*
_output_shapes	
:
ô
save/Assign_14Assign)resnet_model/batch_normalization_11/gammasave/RestoreV2_1:13"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_11/gamma*
validate_shape(

save/Assign_15Assign/resnet_model/batch_normalization_11/moving_meansave/RestoreV2_1:14"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_11/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_16Assign3resnet_model/batch_normalization_11/moving_variancesave/RestoreV2_1:15"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_11/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ò
save/Assign_17Assign(resnet_model/batch_normalization_12/betasave/RestoreV2_1:16"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_12/beta*
validate_shape(
ô
save/Assign_18Assign)resnet_model/batch_normalization_12/gammasave/RestoreV2_1:17"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_12/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_19Assign/resnet_model/batch_normalization_12/moving_meansave/RestoreV2_1:18"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_12/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_20Assign3resnet_model/batch_normalization_12/moving_variancesave/RestoreV2_1:19"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_12/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ò
save/Assign_21Assign(resnet_model/batch_normalization_13/betasave/RestoreV2_1:20"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_13/beta*
validate_shape(
ô
save/Assign_22Assign)resnet_model/batch_normalization_13/gammasave/RestoreV2_1:21"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_13/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_23Assign/resnet_model/batch_normalization_13/moving_meansave/RestoreV2_1:22"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_13/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_24Assign3resnet_model/batch_normalization_13/moving_variancesave/RestoreV2_1:23"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_13/moving_variance
ò
save/Assign_25Assign(resnet_model/batch_normalization_14/betasave/RestoreV2_1:24"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_14/beta*
validate_shape(*
_output_shapes	
:
ô
save/Assign_26Assign)resnet_model/batch_normalization_14/gammasave/RestoreV2_1:25"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_14/gamma

save/Assign_27Assign/resnet_model/batch_normalization_14/moving_meansave/RestoreV2_1:26"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_14/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_28Assign3resnet_model/batch_normalization_14/moving_variancesave/RestoreV2_1:27"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_14/moving_variance*
validate_shape(*
_output_shapes	
:
ò
save/Assign_29Assign(resnet_model/batch_normalization_15/betasave/RestoreV2_1:28"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_15/beta*
validate_shape(*
_output_shapes	
:
ô
save/Assign_30Assign)resnet_model/batch_normalization_15/gammasave/RestoreV2_1:29"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_15/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_31Assign/resnet_model/batch_normalization_15/moving_meansave/RestoreV2_1:30"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_15/moving_mean*
validate_shape(

save/Assign_32Assign3resnet_model/batch_normalization_15/moving_variancesave/RestoreV2_1:31"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_15/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ò
save/Assign_33Assign(resnet_model/batch_normalization_16/betasave/RestoreV2_1:32"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_16/beta*
validate_shape(*
_output_shapes	
:
ô
save/Assign_34Assign)resnet_model/batch_normalization_16/gammasave/RestoreV2_1:33"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_16/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_35Assign/resnet_model/batch_normalization_16/moving_meansave/RestoreV2_1:34"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_16/moving_mean

save/Assign_36Assign3resnet_model/batch_normalization_16/moving_variancesave/RestoreV2_1:35"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_16/moving_variance*
validate_shape(*
_output_shapes	
:
ò
save/Assign_37Assign(resnet_model/batch_normalization_17/betasave/RestoreV2_1:36"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_17/beta
ô
save/Assign_38Assign)resnet_model/batch_normalization_17/gammasave/RestoreV2_1:37"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_17/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_39Assign/resnet_model/batch_normalization_17/moving_meansave/RestoreV2_1:38"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_17/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_40Assign3resnet_model/batch_normalization_17/moving_variancesave/RestoreV2_1:39"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_17/moving_variance
ò
save/Assign_41Assign(resnet_model/batch_normalization_18/betasave/RestoreV2_1:40"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_18/beta*
validate_shape(
ô
save/Assign_42Assign)resnet_model/batch_normalization_18/gammasave/RestoreV2_1:41"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_18/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_43Assign/resnet_model/batch_normalization_18/moving_meansave/RestoreV2_1:42"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_18/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_44Assign3resnet_model/batch_normalization_18/moving_variancesave/RestoreV2_1:43"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_18/moving_variance*
validate_shape(
ò
save/Assign_45Assign(resnet_model/batch_normalization_19/betasave/RestoreV2_1:44"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_19/beta*
validate_shape(
ô
save/Assign_46Assign)resnet_model/batch_normalization_19/gammasave/RestoreV2_1:45"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_19/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_47Assign/resnet_model/batch_normalization_19/moving_meansave/RestoreV2_1:46"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_19/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_48Assign3resnet_model/batch_normalization_19/moving_variancesave/RestoreV2_1:47"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_19/moving_variance*
validate_shape(*
_output_shapes	
:
ï
save/Assign_49Assign'resnet_model/batch_normalization_2/betasave/RestoreV2_1:48"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_2/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
ñ
save/Assign_50Assign(resnet_model/batch_normalization_2/gammasave/RestoreV2_1:49"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_2/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
ý
save/Assign_51Assign.resnet_model/batch_normalization_2/moving_meansave/RestoreV2_1:50"/device:CPU:0*
T0*A
_class7
53loc:@resnet_model/batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes
:@*
use_locking(

save/Assign_52Assign2resnet_model/batch_normalization_2/moving_variancesave/RestoreV2_1:51"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_2/moving_variance*
validate_shape(*
_output_shapes
:@
ò
save/Assign_53Assign(resnet_model/batch_normalization_20/betasave/RestoreV2_1:52"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_20/beta
ô
save/Assign_54Assign)resnet_model/batch_normalization_20/gammasave/RestoreV2_1:53"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_20/gamma*
validate_shape(

save/Assign_55Assign/resnet_model/batch_normalization_20/moving_meansave/RestoreV2_1:54"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_20/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_56Assign3resnet_model/batch_normalization_20/moving_variancesave/RestoreV2_1:55"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_20/moving_variance*
validate_shape(*
_output_shapes	
:
ò
save/Assign_57Assign(resnet_model/batch_normalization_21/betasave/RestoreV2_1:56"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_21/beta
ô
save/Assign_58Assign)resnet_model/batch_normalization_21/gammasave/RestoreV2_1:57"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_21/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_59Assign/resnet_model/batch_normalization_21/moving_meansave/RestoreV2_1:58"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_21/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_60Assign3resnet_model/batch_normalization_21/moving_variancesave/RestoreV2_1:59"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_21/moving_variance*
validate_shape(*
_output_shapes	
:
ò
save/Assign_61Assign(resnet_model/batch_normalization_22/betasave/RestoreV2_1:60"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_22/beta*
validate_shape(
ô
save/Assign_62Assign)resnet_model/batch_normalization_22/gammasave/RestoreV2_1:61"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_22/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_63Assign/resnet_model/batch_normalization_22/moving_meansave/RestoreV2_1:62"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_22/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_64Assign3resnet_model/batch_normalization_22/moving_variancesave/RestoreV2_1:63"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_22/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ò
save/Assign_65Assign(resnet_model/batch_normalization_23/betasave/RestoreV2_1:64"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_23/beta*
validate_shape(*
_output_shapes	
:
ô
save/Assign_66Assign)resnet_model/batch_normalization_23/gammasave/RestoreV2_1:65"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_23/gamma*
validate_shape(

save/Assign_67Assign/resnet_model/batch_normalization_23/moving_meansave/RestoreV2_1:66"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_23/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_68Assign3resnet_model/batch_normalization_23/moving_variancesave/RestoreV2_1:67"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_23/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ò
save/Assign_69Assign(resnet_model/batch_normalization_24/betasave/RestoreV2_1:68"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_24/beta*
validate_shape(
ô
save/Assign_70Assign)resnet_model/batch_normalization_24/gammasave/RestoreV2_1:69"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_24/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_71Assign/resnet_model/batch_normalization_24/moving_meansave/RestoreV2_1:70"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_24/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_72Assign3resnet_model/batch_normalization_24/moving_variancesave/RestoreV2_1:71"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_24/moving_variance*
validate_shape(*
_output_shapes	
:
ò
save/Assign_73Assign(resnet_model/batch_normalization_25/betasave/RestoreV2_1:72"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_25/beta
ô
save/Assign_74Assign)resnet_model/batch_normalization_25/gammasave/RestoreV2_1:73"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_25/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_75Assign/resnet_model/batch_normalization_25/moving_meansave/RestoreV2_1:74"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_25/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_76Assign3resnet_model/batch_normalization_25/moving_variancesave/RestoreV2_1:75"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_25/moving_variance
ò
save/Assign_77Assign(resnet_model/batch_normalization_26/betasave/RestoreV2_1:76"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_26/beta*
validate_shape(*
_output_shapes	
:
ô
save/Assign_78Assign)resnet_model/batch_normalization_26/gammasave/RestoreV2_1:77"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_26/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_79Assign/resnet_model/batch_normalization_26/moving_meansave/RestoreV2_1:78"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_26/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_80Assign3resnet_model/batch_normalization_26/moving_variancesave/RestoreV2_1:79"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_26/moving_variance*
validate_shape(*
_output_shapes	
:
ò
save/Assign_81Assign(resnet_model/batch_normalization_27/betasave/RestoreV2_1:80"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_27/beta*
validate_shape(*
_output_shapes	
:
ô
save/Assign_82Assign)resnet_model/batch_normalization_27/gammasave/RestoreV2_1:81"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_27/gamma*
validate_shape(

save/Assign_83Assign/resnet_model/batch_normalization_27/moving_meansave/RestoreV2_1:82"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_27/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_84Assign3resnet_model/batch_normalization_27/moving_variancesave/RestoreV2_1:83"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_27/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ò
save/Assign_85Assign(resnet_model/batch_normalization_28/betasave/RestoreV2_1:84"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_28/beta*
validate_shape(*
_output_shapes	
:
ô
save/Assign_86Assign)resnet_model/batch_normalization_28/gammasave/RestoreV2_1:85"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_28/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_87Assign/resnet_model/batch_normalization_28/moving_meansave/RestoreV2_1:86"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_28/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_88Assign3resnet_model/batch_normalization_28/moving_variancesave/RestoreV2_1:87"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_28/moving_variance*
validate_shape(*
_output_shapes	
:
ò
save/Assign_89Assign(resnet_model/batch_normalization_29/betasave/RestoreV2_1:88"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_29/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ô
save/Assign_90Assign)resnet_model/batch_normalization_29/gammasave/RestoreV2_1:89"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_29/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_91Assign/resnet_model/batch_normalization_29/moving_meansave/RestoreV2_1:90"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_29/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_92Assign3resnet_model/batch_normalization_29/moving_variancesave/RestoreV2_1:91"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_29/moving_variance*
validate_shape(
ð
save/Assign_93Assign'resnet_model/batch_normalization_3/betasave/RestoreV2_1:92"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_3/beta
ò
save/Assign_94Assign(resnet_model/batch_normalization_3/gammasave/RestoreV2_1:93"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_3/gamma
þ
save/Assign_95Assign.resnet_model/batch_normalization_3/moving_meansave/RestoreV2_1:94"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_3/moving_mean*
validate_shape(

save/Assign_96Assign2resnet_model/batch_normalization_3/moving_variancesave/RestoreV2_1:95"/device:CPU:0*E
_class;
97loc:@resnet_model/batch_normalization_3/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ò
save/Assign_97Assign(resnet_model/batch_normalization_30/betasave/RestoreV2_1:96"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_30/beta*
validate_shape(*
_output_shapes	
:
ô
save/Assign_98Assign)resnet_model/batch_normalization_30/gammasave/RestoreV2_1:97"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_30/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_99Assign/resnet_model/batch_normalization_30/moving_meansave/RestoreV2_1:98"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_30/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_100Assign3resnet_model/batch_normalization_30/moving_variancesave/RestoreV2_1:99"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_30/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ô
save/Assign_101Assign(resnet_model/batch_normalization_31/betasave/RestoreV2_1:100"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_31/beta*
validate_shape(
ö
save/Assign_102Assign)resnet_model/batch_normalization_31/gammasave/RestoreV2_1:101"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_31/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_103Assign/resnet_model/batch_normalization_31/moving_meansave/RestoreV2_1:102"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_31/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_104Assign3resnet_model/batch_normalization_31/moving_variancesave/RestoreV2_1:103"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_31/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ô
save/Assign_105Assign(resnet_model/batch_normalization_32/betasave/RestoreV2_1:104"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_32/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
save/Assign_106Assign)resnet_model/batch_normalization_32/gammasave/RestoreV2_1:105"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_32/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_107Assign/resnet_model/batch_normalization_32/moving_meansave/RestoreV2_1:106"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_32/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_108Assign3resnet_model/batch_normalization_32/moving_variancesave/RestoreV2_1:107"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_32/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ô
save/Assign_109Assign(resnet_model/batch_normalization_33/betasave/RestoreV2_1:108"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_33/beta*
validate_shape(*
_output_shapes	
:
ö
save/Assign_110Assign)resnet_model/batch_normalization_33/gammasave/RestoreV2_1:109"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_33/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_111Assign/resnet_model/batch_normalization_33/moving_meansave/RestoreV2_1:110"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_33/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_112Assign3resnet_model/batch_normalization_33/moving_variancesave/RestoreV2_1:111"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_33/moving_variance
ô
save/Assign_113Assign(resnet_model/batch_normalization_34/betasave/RestoreV2_1:112"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_34/beta*
validate_shape(*
_output_shapes	
:
ö
save/Assign_114Assign)resnet_model/batch_normalization_34/gammasave/RestoreV2_1:113"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_34/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_115Assign/resnet_model/batch_normalization_34/moving_meansave/RestoreV2_1:114"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_34/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_116Assign3resnet_model/batch_normalization_34/moving_variancesave/RestoreV2_1:115"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_34/moving_variance*
validate_shape(*
_output_shapes	
:
ô
save/Assign_117Assign(resnet_model/batch_normalization_35/betasave/RestoreV2_1:116"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_35/beta*
validate_shape(*
_output_shapes	
:
ö
save/Assign_118Assign)resnet_model/batch_normalization_35/gammasave/RestoreV2_1:117"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_35/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_119Assign/resnet_model/batch_normalization_35/moving_meansave/RestoreV2_1:118"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_35/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_120Assign3resnet_model/batch_normalization_35/moving_variancesave/RestoreV2_1:119"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_35/moving_variance*
validate_shape(*
_output_shapes	
:
ô
save/Assign_121Assign(resnet_model/batch_normalization_36/betasave/RestoreV2_1:120"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_36/beta*
validate_shape(*
_output_shapes	
:
ö
save/Assign_122Assign)resnet_model/batch_normalization_36/gammasave/RestoreV2_1:121"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_36/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_123Assign/resnet_model/batch_normalization_36/moving_meansave/RestoreV2_1:122"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_36/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_124Assign3resnet_model/batch_normalization_36/moving_variancesave/RestoreV2_1:123"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_36/moving_variance
ô
save/Assign_125Assign(resnet_model/batch_normalization_37/betasave/RestoreV2_1:124"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_37/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
save/Assign_126Assign)resnet_model/batch_normalization_37/gammasave/RestoreV2_1:125"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_37/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_127Assign/resnet_model/batch_normalization_37/moving_meansave/RestoreV2_1:126"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_37/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_128Assign3resnet_model/batch_normalization_37/moving_variancesave/RestoreV2_1:127"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_37/moving_variance*
validate_shape(*
_output_shapes	
:
ô
save/Assign_129Assign(resnet_model/batch_normalization_38/betasave/RestoreV2_1:128"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_38/beta*
validate_shape(
ö
save/Assign_130Assign)resnet_model/batch_normalization_38/gammasave/RestoreV2_1:129"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_38/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_131Assign/resnet_model/batch_normalization_38/moving_meansave/RestoreV2_1:130"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_38/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_132Assign3resnet_model/batch_normalization_38/moving_variancesave/RestoreV2_1:131"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_38/moving_variance*
validate_shape(*
_output_shapes	
:
ô
save/Assign_133Assign(resnet_model/batch_normalization_39/betasave/RestoreV2_1:132"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_39/beta
ö
save/Assign_134Assign)resnet_model/batch_normalization_39/gammasave/RestoreV2_1:133"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_39/gamma*
validate_shape(

save/Assign_135Assign/resnet_model/batch_normalization_39/moving_meansave/RestoreV2_1:134"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_39/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_136Assign3resnet_model/batch_normalization_39/moving_variancesave/RestoreV2_1:135"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_39/moving_variance*
validate_shape(*
_output_shapes	
:
ñ
save/Assign_137Assign'resnet_model/batch_normalization_4/betasave/RestoreV2_1:136"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_4/beta*
validate_shape(*
_output_shapes
:@
ó
save/Assign_138Assign(resnet_model/batch_normalization_4/gammasave/RestoreV2_1:137"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_4/gamma*
validate_shape(
ÿ
save/Assign_139Assign.resnet_model/batch_normalization_4/moving_meansave/RestoreV2_1:138"/device:CPU:0*
T0*A
_class7
53loc:@resnet_model/batch_normalization_4/moving_mean*
validate_shape(*
_output_shapes
:@*
use_locking(

save/Assign_140Assign2resnet_model/batch_normalization_4/moving_variancesave/RestoreV2_1:139"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_4/moving_variance*
validate_shape(*
_output_shapes
:@
ô
save/Assign_141Assign(resnet_model/batch_normalization_40/betasave/RestoreV2_1:140"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_40/beta*
validate_shape(*
_output_shapes	
:
ö
save/Assign_142Assign)resnet_model/batch_normalization_40/gammasave/RestoreV2_1:141"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_40/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_143Assign/resnet_model/batch_normalization_40/moving_meansave/RestoreV2_1:142"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_40/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_144Assign3resnet_model/batch_normalization_40/moving_variancesave/RestoreV2_1:143"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_40/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ô
save/Assign_145Assign(resnet_model/batch_normalization_41/betasave/RestoreV2_1:144"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_41/beta*
validate_shape(*
_output_shapes	
:
ö
save/Assign_146Assign)resnet_model/batch_normalization_41/gammasave/RestoreV2_1:145"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_41/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_147Assign/resnet_model/batch_normalization_41/moving_meansave/RestoreV2_1:146"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_41/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_148Assign3resnet_model/batch_normalization_41/moving_variancesave/RestoreV2_1:147"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_41/moving_variance*
validate_shape(
ô
save/Assign_149Assign(resnet_model/batch_normalization_42/betasave/RestoreV2_1:148"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_42/beta*
validate_shape(*
_output_shapes	
:
ö
save/Assign_150Assign)resnet_model/batch_normalization_42/gammasave/RestoreV2_1:149"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_42/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_151Assign/resnet_model/batch_normalization_42/moving_meansave/RestoreV2_1:150"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_42/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_152Assign3resnet_model/batch_normalization_42/moving_variancesave/RestoreV2_1:151"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_42/moving_variance*
validate_shape(
ô
save/Assign_153Assign(resnet_model/batch_normalization_43/betasave/RestoreV2_1:152"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_43/beta*
validate_shape(*
_output_shapes	
:
ö
save/Assign_154Assign)resnet_model/batch_normalization_43/gammasave/RestoreV2_1:153"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_43/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_155Assign/resnet_model/batch_normalization_43/moving_meansave/RestoreV2_1:154"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_43/moving_mean*
validate_shape(

save/Assign_156Assign3resnet_model/batch_normalization_43/moving_variancesave/RestoreV2_1:155"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_43/moving_variance
ô
save/Assign_157Assign(resnet_model/batch_normalization_44/betasave/RestoreV2_1:156"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_44/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
save/Assign_158Assign)resnet_model/batch_normalization_44/gammasave/RestoreV2_1:157"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_44/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_159Assign/resnet_model/batch_normalization_44/moving_meansave/RestoreV2_1:158"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_44/moving_mean*
validate_shape(

save/Assign_160Assign3resnet_model/batch_normalization_44/moving_variancesave/RestoreV2_1:159"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_44/moving_variance*
validate_shape(*
_output_shapes	
:
ô
save/Assign_161Assign(resnet_model/batch_normalization_45/betasave/RestoreV2_1:160"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_45/beta*
validate_shape(*
_output_shapes	
:
ö
save/Assign_162Assign)resnet_model/batch_normalization_45/gammasave/RestoreV2_1:161"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_45/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_163Assign/resnet_model/batch_normalization_45/moving_meansave/RestoreV2_1:162"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_45/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_164Assign3resnet_model/batch_normalization_45/moving_variancesave/RestoreV2_1:163"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_45/moving_variance*
validate_shape(*
_output_shapes	
:
ô
save/Assign_165Assign(resnet_model/batch_normalization_46/betasave/RestoreV2_1:164"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_46/beta
ö
save/Assign_166Assign)resnet_model/batch_normalization_46/gammasave/RestoreV2_1:165"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_46/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_167Assign/resnet_model/batch_normalization_46/moving_meansave/RestoreV2_1:166"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_46/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_168Assign3resnet_model/batch_normalization_46/moving_variancesave/RestoreV2_1:167"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_46/moving_variance*
validate_shape(*
_output_shapes	
:
ô
save/Assign_169Assign(resnet_model/batch_normalization_47/betasave/RestoreV2_1:168"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_47/beta*
validate_shape(
ö
save/Assign_170Assign)resnet_model/batch_normalization_47/gammasave/RestoreV2_1:169"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_47/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_171Assign/resnet_model/batch_normalization_47/moving_meansave/RestoreV2_1:170"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_47/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save/Assign_172Assign3resnet_model/batch_normalization_47/moving_variancesave/RestoreV2_1:171"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_47/moving_variance*
validate_shape(
ô
save/Assign_173Assign(resnet_model/batch_normalization_48/betasave/RestoreV2_1:172"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_48/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
save/Assign_174Assign)resnet_model/batch_normalization_48/gammasave/RestoreV2_1:173"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_48/gamma*
validate_shape(

save/Assign_175Assign/resnet_model/batch_normalization_48/moving_meansave/RestoreV2_1:174"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_48/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save/Assign_176Assign3resnet_model/batch_normalization_48/moving_variancesave/RestoreV2_1:175"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_48/moving_variance*
validate_shape(*
_output_shapes	
:
ñ
save/Assign_177Assign'resnet_model/batch_normalization_5/betasave/RestoreV2_1:176"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_5/beta*
validate_shape(*
_output_shapes
:@
ó
save/Assign_178Assign(resnet_model/batch_normalization_5/gammasave/RestoreV2_1:177"/device:CPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_5/gamma
ÿ
save/Assign_179Assign.resnet_model/batch_normalization_5/moving_meansave/RestoreV2_1:178"/device:CPU:0*
T0*A
_class7
53loc:@resnet_model/batch_normalization_5/moving_mean*
validate_shape(*
_output_shapes
:@*
use_locking(

save/Assign_180Assign2resnet_model/batch_normalization_5/moving_variancesave/RestoreV2_1:179"/device:CPU:0*
T0*E
_class;
97loc:@resnet_model/batch_normalization_5/moving_variance*
validate_shape(*
_output_shapes
:@*
use_locking(
ò
save/Assign_181Assign'resnet_model/batch_normalization_6/betasave/RestoreV2_1:180"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_6/beta*
validate_shape(*
_output_shapes	
:
ô
save/Assign_182Assign(resnet_model/batch_normalization_6/gammasave/RestoreV2_1:181"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_6/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_183Assign.resnet_model/batch_normalization_6/moving_meansave/RestoreV2_1:182"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_6/moving_mean

save/Assign_184Assign2resnet_model/batch_normalization_6/moving_variancesave/RestoreV2_1:183"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_6/moving_variance*
validate_shape(*
_output_shapes	
:
ñ
save/Assign_185Assign'resnet_model/batch_normalization_7/betasave/RestoreV2_1:184"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_7/beta*
validate_shape(
ó
save/Assign_186Assign(resnet_model/batch_normalization_7/gammasave/RestoreV2_1:185"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_7/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(
ÿ
save/Assign_187Assign.resnet_model/batch_normalization_7/moving_meansave/RestoreV2_1:186"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_7/moving_mean*
validate_shape(*
_output_shapes
:@

save/Assign_188Assign2resnet_model/batch_normalization_7/moving_variancesave/RestoreV2_1:187"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_7/moving_variance*
validate_shape(*
_output_shapes
:@
ñ
save/Assign_189Assign'resnet_model/batch_normalization_8/betasave/RestoreV2_1:188"/device:CPU:0*:
_class0
.,loc:@resnet_model/batch_normalization_8/beta*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
ó
save/Assign_190Assign(resnet_model/batch_normalization_8/gammasave/RestoreV2_1:189"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_8/gamma*
validate_shape(
ÿ
save/Assign_191Assign.resnet_model/batch_normalization_8/moving_meansave/RestoreV2_1:190"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_8/moving_mean*
validate_shape(*
_output_shapes
:@

save/Assign_192Assign2resnet_model/batch_normalization_8/moving_variancesave/RestoreV2_1:191"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_8/moving_variance*
validate_shape(*
_output_shapes
:@
ò
save/Assign_193Assign'resnet_model/batch_normalization_9/betasave/RestoreV2_1:192"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_9/beta*
validate_shape(
ô
save/Assign_194Assign(resnet_model/batch_normalization_9/gammasave/RestoreV2_1:193"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_9/gamma*
validate_shape(*
_output_shapes	
:

save/Assign_195Assign.resnet_model/batch_normalization_9/moving_meansave/RestoreV2_1:194"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_9/moving_mean*
validate_shape(*
_output_shapes	
:

save/Assign_196Assign2resnet_model/batch_normalization_9/moving_variancesave/RestoreV2_1:195"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_9/moving_variance*
validate_shape(*
_output_shapes	
:
ã
save/Assign_197Assignresnet_model/conv2d/kernelsave/RestoreV2_1:196"/device:CPU:0*&
_output_shapes
:@*
use_locking(*
T0*-
_class#
!loc:@resnet_model/conv2d/kernel*
validate_shape(
è
save/Assign_198Assignresnet_model/conv2d_1/kernelsave/RestoreV2_1:197"/device:CPU:0*'
_output_shapes
:@*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_1/kernel*
validate_shape(
ê
save/Assign_199Assignresnet_model/conv2d_10/kernelsave/RestoreV2_1:198"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_10/kernel*
validate_shape(*'
_output_shapes
:@
ë
save/Assign_200Assignresnet_model/conv2d_11/kernelsave/RestoreV2_1:199"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_201Assignresnet_model/conv2d_12/kernelsave/RestoreV2_1:200"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_12/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_202Assignresnet_model/conv2d_13/kernelsave/RestoreV2_1:201"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_13/kernel
ë
save/Assign_203Assignresnet_model/conv2d_14/kernelsave/RestoreV2_1:202"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_14/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_204Assignresnet_model/conv2d_15/kernelsave/RestoreV2_1:203"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_15/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_205Assignresnet_model/conv2d_16/kernelsave/RestoreV2_1:204"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_16/kernel*
validate_shape(
ë
save/Assign_206Assignresnet_model/conv2d_17/kernelsave/RestoreV2_1:205"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_17/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ë
save/Assign_207Assignresnet_model/conv2d_18/kernelsave/RestoreV2_1:206"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_18/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_208Assignresnet_model/conv2d_19/kernelsave/RestoreV2_1:207"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_19/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
ç
save/Assign_209Assignresnet_model/conv2d_2/kernelsave/RestoreV2_1:208"/device:CPU:0*&
_output_shapes
:@@*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_2/kernel*
validate_shape(
ë
save/Assign_210Assignresnet_model/conv2d_20/kernelsave/RestoreV2_1:209"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_20/kernel
ë
save/Assign_211Assignresnet_model/conv2d_21/kernelsave/RestoreV2_1:210"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_21/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_212Assignresnet_model/conv2d_22/kernelsave/RestoreV2_1:211"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_22/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_213Assignresnet_model/conv2d_23/kernelsave/RestoreV2_1:212"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_23/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_214Assignresnet_model/conv2d_24/kernelsave/RestoreV2_1:213"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_24/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ë
save/Assign_215Assignresnet_model/conv2d_25/kernelsave/RestoreV2_1:214"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_25/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_216Assignresnet_model/conv2d_26/kernelsave/RestoreV2_1:215"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_26/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_217Assignresnet_model/conv2d_27/kernelsave/RestoreV2_1:216"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_27/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_218Assignresnet_model/conv2d_28/kernelsave/RestoreV2_1:217"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_28/kernel*
validate_shape(
ë
save/Assign_219Assignresnet_model/conv2d_29/kernelsave/RestoreV2_1:218"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_29/kernel*
validate_shape(*(
_output_shapes
:
ç
save/Assign_220Assignresnet_model/conv2d_3/kernelsave/RestoreV2_1:219"/device:CPU:0*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_3/kernel
ë
save/Assign_221Assignresnet_model/conv2d_30/kernelsave/RestoreV2_1:220"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_30/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_222Assignresnet_model/conv2d_31/kernelsave/RestoreV2_1:221"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_31/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
ë
save/Assign_223Assignresnet_model/conv2d_32/kernelsave/RestoreV2_1:222"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_32/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ë
save/Assign_224Assignresnet_model/conv2d_33/kernelsave/RestoreV2_1:223"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_33/kernel
ë
save/Assign_225Assignresnet_model/conv2d_34/kernelsave/RestoreV2_1:224"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_34/kernel
ë
save/Assign_226Assignresnet_model/conv2d_35/kernelsave/RestoreV2_1:225"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_35/kernel
ë
save/Assign_227Assignresnet_model/conv2d_36/kernelsave/RestoreV2_1:226"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_36/kernel*
validate_shape(
ë
save/Assign_228Assignresnet_model/conv2d_37/kernelsave/RestoreV2_1:227"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_37/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_229Assignresnet_model/conv2d_38/kernelsave/RestoreV2_1:228"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_38/kernel*
validate_shape(
ë
save/Assign_230Assignresnet_model/conv2d_39/kernelsave/RestoreV2_1:229"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_39/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
è
save/Assign_231Assignresnet_model/conv2d_4/kernelsave/RestoreV2_1:230"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_4/kernel*
validate_shape(*'
_output_shapes
:@*
use_locking(
ë
save/Assign_232Assignresnet_model/conv2d_40/kernelsave/RestoreV2_1:231"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_40/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_233Assignresnet_model/conv2d_41/kernelsave/RestoreV2_1:232"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_41/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_234Assignresnet_model/conv2d_42/kernelsave/RestoreV2_1:233"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_42/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_235Assignresnet_model/conv2d_43/kernelsave/RestoreV2_1:234"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_43/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_236Assignresnet_model/conv2d_44/kernelsave/RestoreV2_1:235"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_44/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ë
save/Assign_237Assignresnet_model/conv2d_45/kernelsave/RestoreV2_1:236"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_45/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_238Assignresnet_model/conv2d_46/kernelsave/RestoreV2_1:237"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_46/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_239Assignresnet_model/conv2d_47/kernelsave/RestoreV2_1:238"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_47/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
ë
save/Assign_240Assignresnet_model/conv2d_48/kernelsave/RestoreV2_1:239"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_48/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ë
save/Assign_241Assignresnet_model/conv2d_49/kernelsave/RestoreV2_1:240"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_49/kernel*
validate_shape(*(
_output_shapes
:
è
save/Assign_242Assignresnet_model/conv2d_5/kernelsave/RestoreV2_1:241"/device:CPU:0*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_5/kernel
ë
save/Assign_243Assignresnet_model/conv2d_50/kernelsave/RestoreV2_1:242"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_50/kernel*
validate_shape(*(
_output_shapes
:
ë
save/Assign_244Assignresnet_model/conv2d_51/kernelsave/RestoreV2_1:243"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_51/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ë
save/Assign_245Assignresnet_model/conv2d_52/kernelsave/RestoreV2_1:244"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_52/kernel*
validate_shape(*(
_output_shapes
:
ç
save/Assign_246Assignresnet_model/conv2d_6/kernelsave/RestoreV2_1:245"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_6/kernel*
validate_shape(*&
_output_shapes
:@@
è
save/Assign_247Assignresnet_model/conv2d_7/kernelsave/RestoreV2_1:246"/device:CPU:0*/
_class%
#!loc:@resnet_model/conv2d_7/kernel*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0
è
save/Assign_248Assignresnet_model/conv2d_8/kernelsave/RestoreV2_1:247"/device:CPU:0*
validate_shape(*'
_output_shapes
:@*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_8/kernel
ç
save/Assign_249Assignresnet_model/conv2d_9/kernelsave/RestoreV2_1:248"/device:CPU:0*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_9/kernel
Ò
save/Assign_250Assignresnet_model/dense/biassave/RestoreV2_1:249"/device:CPU:0*
use_locking(*
T0**
_class 
loc:@resnet_model/dense/bias*
validate_shape(*
_output_shapes	
:é
Û
save/Assign_251Assignresnet_model/dense/kernelsave/RestoreV2_1:250"/device:CPU:0*,
_class"
 loc:@resnet_model/dense/kernel*
validate_shape(* 
_output_shapes
:
é*
use_locking(*
T0
å"
save/restore_shard_1NoOp^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_11^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_12^save/Assign_120^save/Assign_121^save/Assign_122^save/Assign_123^save/Assign_124^save/Assign_125^save/Assign_126^save/Assign_127^save/Assign_128^save/Assign_129^save/Assign_13^save/Assign_130^save/Assign_131^save/Assign_132^save/Assign_133^save/Assign_134^save/Assign_135^save/Assign_136^save/Assign_137^save/Assign_138^save/Assign_139^save/Assign_14^save/Assign_140^save/Assign_141^save/Assign_142^save/Assign_143^save/Assign_144^save/Assign_145^save/Assign_146^save/Assign_147^save/Assign_148^save/Assign_149^save/Assign_15^save/Assign_150^save/Assign_151^save/Assign_152^save/Assign_153^save/Assign_154^save/Assign_155^save/Assign_156^save/Assign_157^save/Assign_158^save/Assign_159^save/Assign_16^save/Assign_160^save/Assign_161^save/Assign_162^save/Assign_163^save/Assign_164^save/Assign_165^save/Assign_166^save/Assign_167^save/Assign_168^save/Assign_169^save/Assign_17^save/Assign_170^save/Assign_171^save/Assign_172^save/Assign_173^save/Assign_174^save/Assign_175^save/Assign_176^save/Assign_177^save/Assign_178^save/Assign_179^save/Assign_18^save/Assign_180^save/Assign_181^save/Assign_182^save/Assign_183^save/Assign_184^save/Assign_185^save/Assign_186^save/Assign_187^save/Assign_188^save/Assign_189^save/Assign_19^save/Assign_190^save/Assign_191^save/Assign_192^save/Assign_193^save/Assign_194^save/Assign_195^save/Assign_196^save/Assign_197^save/Assign_198^save/Assign_199^save/Assign_2^save/Assign_20^save/Assign_200^save/Assign_201^save/Assign_202^save/Assign_203^save/Assign_204^save/Assign_205^save/Assign_206^save/Assign_207^save/Assign_208^save/Assign_209^save/Assign_21^save/Assign_210^save/Assign_211^save/Assign_212^save/Assign_213^save/Assign_214^save/Assign_215^save/Assign_216^save/Assign_217^save/Assign_218^save/Assign_219^save/Assign_22^save/Assign_220^save/Assign_221^save/Assign_222^save/Assign_223^save/Assign_224^save/Assign_225^save/Assign_226^save/Assign_227^save/Assign_228^save/Assign_229^save/Assign_23^save/Assign_230^save/Assign_231^save/Assign_232^save/Assign_233^save/Assign_234^save/Assign_235^save/Assign_236^save/Assign_237^save/Assign_238^save/Assign_239^save/Assign_24^save/Assign_240^save/Assign_241^save/Assign_242^save/Assign_243^save/Assign_244^save/Assign_245^save/Assign_246^save/Assign_247^save/Assign_248^save/Assign_249^save/Assign_25^save/Assign_250^save/Assign_251^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_5bd604d52bcc4e2d8de7ddcffdaf134d/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
}
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:* 
valueBBglobal_step
v
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesglobal_step"/device:CPU:0*
dtypes
2	
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
o
save_1/ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 

save_1/ShardedFilename_1ShardedFilenamesave_1/StringJoinsave_1/ShardedFilename_1/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
¥U
save_1/SaveV2_1/tensor_namesConst"/device:CPU:0*ÄT
valueºTB·TûB%resnet_model/batch_normalization/betaB&resnet_model/batch_normalization/gammaB,resnet_model/batch_normalization/moving_meanB0resnet_model/batch_normalization/moving_varianceB'resnet_model/batch_normalization_1/betaB(resnet_model/batch_normalization_1/gammaB.resnet_model/batch_normalization_1/moving_meanB2resnet_model/batch_normalization_1/moving_varianceB(resnet_model/batch_normalization_10/betaB)resnet_model/batch_normalization_10/gammaB/resnet_model/batch_normalization_10/moving_meanB3resnet_model/batch_normalization_10/moving_varianceB(resnet_model/batch_normalization_11/betaB)resnet_model/batch_normalization_11/gammaB/resnet_model/batch_normalization_11/moving_meanB3resnet_model/batch_normalization_11/moving_varianceB(resnet_model/batch_normalization_12/betaB)resnet_model/batch_normalization_12/gammaB/resnet_model/batch_normalization_12/moving_meanB3resnet_model/batch_normalization_12/moving_varianceB(resnet_model/batch_normalization_13/betaB)resnet_model/batch_normalization_13/gammaB/resnet_model/batch_normalization_13/moving_meanB3resnet_model/batch_normalization_13/moving_varianceB(resnet_model/batch_normalization_14/betaB)resnet_model/batch_normalization_14/gammaB/resnet_model/batch_normalization_14/moving_meanB3resnet_model/batch_normalization_14/moving_varianceB(resnet_model/batch_normalization_15/betaB)resnet_model/batch_normalization_15/gammaB/resnet_model/batch_normalization_15/moving_meanB3resnet_model/batch_normalization_15/moving_varianceB(resnet_model/batch_normalization_16/betaB)resnet_model/batch_normalization_16/gammaB/resnet_model/batch_normalization_16/moving_meanB3resnet_model/batch_normalization_16/moving_varianceB(resnet_model/batch_normalization_17/betaB)resnet_model/batch_normalization_17/gammaB/resnet_model/batch_normalization_17/moving_meanB3resnet_model/batch_normalization_17/moving_varianceB(resnet_model/batch_normalization_18/betaB)resnet_model/batch_normalization_18/gammaB/resnet_model/batch_normalization_18/moving_meanB3resnet_model/batch_normalization_18/moving_varianceB(resnet_model/batch_normalization_19/betaB)resnet_model/batch_normalization_19/gammaB/resnet_model/batch_normalization_19/moving_meanB3resnet_model/batch_normalization_19/moving_varianceB'resnet_model/batch_normalization_2/betaB(resnet_model/batch_normalization_2/gammaB.resnet_model/batch_normalization_2/moving_meanB2resnet_model/batch_normalization_2/moving_varianceB(resnet_model/batch_normalization_20/betaB)resnet_model/batch_normalization_20/gammaB/resnet_model/batch_normalization_20/moving_meanB3resnet_model/batch_normalization_20/moving_varianceB(resnet_model/batch_normalization_21/betaB)resnet_model/batch_normalization_21/gammaB/resnet_model/batch_normalization_21/moving_meanB3resnet_model/batch_normalization_21/moving_varianceB(resnet_model/batch_normalization_22/betaB)resnet_model/batch_normalization_22/gammaB/resnet_model/batch_normalization_22/moving_meanB3resnet_model/batch_normalization_22/moving_varianceB(resnet_model/batch_normalization_23/betaB)resnet_model/batch_normalization_23/gammaB/resnet_model/batch_normalization_23/moving_meanB3resnet_model/batch_normalization_23/moving_varianceB(resnet_model/batch_normalization_24/betaB)resnet_model/batch_normalization_24/gammaB/resnet_model/batch_normalization_24/moving_meanB3resnet_model/batch_normalization_24/moving_varianceB(resnet_model/batch_normalization_25/betaB)resnet_model/batch_normalization_25/gammaB/resnet_model/batch_normalization_25/moving_meanB3resnet_model/batch_normalization_25/moving_varianceB(resnet_model/batch_normalization_26/betaB)resnet_model/batch_normalization_26/gammaB/resnet_model/batch_normalization_26/moving_meanB3resnet_model/batch_normalization_26/moving_varianceB(resnet_model/batch_normalization_27/betaB)resnet_model/batch_normalization_27/gammaB/resnet_model/batch_normalization_27/moving_meanB3resnet_model/batch_normalization_27/moving_varianceB(resnet_model/batch_normalization_28/betaB)resnet_model/batch_normalization_28/gammaB/resnet_model/batch_normalization_28/moving_meanB3resnet_model/batch_normalization_28/moving_varianceB(resnet_model/batch_normalization_29/betaB)resnet_model/batch_normalization_29/gammaB/resnet_model/batch_normalization_29/moving_meanB3resnet_model/batch_normalization_29/moving_varianceB'resnet_model/batch_normalization_3/betaB(resnet_model/batch_normalization_3/gammaB.resnet_model/batch_normalization_3/moving_meanB2resnet_model/batch_normalization_3/moving_varianceB(resnet_model/batch_normalization_30/betaB)resnet_model/batch_normalization_30/gammaB/resnet_model/batch_normalization_30/moving_meanB3resnet_model/batch_normalization_30/moving_varianceB(resnet_model/batch_normalization_31/betaB)resnet_model/batch_normalization_31/gammaB/resnet_model/batch_normalization_31/moving_meanB3resnet_model/batch_normalization_31/moving_varianceB(resnet_model/batch_normalization_32/betaB)resnet_model/batch_normalization_32/gammaB/resnet_model/batch_normalization_32/moving_meanB3resnet_model/batch_normalization_32/moving_varianceB(resnet_model/batch_normalization_33/betaB)resnet_model/batch_normalization_33/gammaB/resnet_model/batch_normalization_33/moving_meanB3resnet_model/batch_normalization_33/moving_varianceB(resnet_model/batch_normalization_34/betaB)resnet_model/batch_normalization_34/gammaB/resnet_model/batch_normalization_34/moving_meanB3resnet_model/batch_normalization_34/moving_varianceB(resnet_model/batch_normalization_35/betaB)resnet_model/batch_normalization_35/gammaB/resnet_model/batch_normalization_35/moving_meanB3resnet_model/batch_normalization_35/moving_varianceB(resnet_model/batch_normalization_36/betaB)resnet_model/batch_normalization_36/gammaB/resnet_model/batch_normalization_36/moving_meanB3resnet_model/batch_normalization_36/moving_varianceB(resnet_model/batch_normalization_37/betaB)resnet_model/batch_normalization_37/gammaB/resnet_model/batch_normalization_37/moving_meanB3resnet_model/batch_normalization_37/moving_varianceB(resnet_model/batch_normalization_38/betaB)resnet_model/batch_normalization_38/gammaB/resnet_model/batch_normalization_38/moving_meanB3resnet_model/batch_normalization_38/moving_varianceB(resnet_model/batch_normalization_39/betaB)resnet_model/batch_normalization_39/gammaB/resnet_model/batch_normalization_39/moving_meanB3resnet_model/batch_normalization_39/moving_varianceB'resnet_model/batch_normalization_4/betaB(resnet_model/batch_normalization_4/gammaB.resnet_model/batch_normalization_4/moving_meanB2resnet_model/batch_normalization_4/moving_varianceB(resnet_model/batch_normalization_40/betaB)resnet_model/batch_normalization_40/gammaB/resnet_model/batch_normalization_40/moving_meanB3resnet_model/batch_normalization_40/moving_varianceB(resnet_model/batch_normalization_41/betaB)resnet_model/batch_normalization_41/gammaB/resnet_model/batch_normalization_41/moving_meanB3resnet_model/batch_normalization_41/moving_varianceB(resnet_model/batch_normalization_42/betaB)resnet_model/batch_normalization_42/gammaB/resnet_model/batch_normalization_42/moving_meanB3resnet_model/batch_normalization_42/moving_varianceB(resnet_model/batch_normalization_43/betaB)resnet_model/batch_normalization_43/gammaB/resnet_model/batch_normalization_43/moving_meanB3resnet_model/batch_normalization_43/moving_varianceB(resnet_model/batch_normalization_44/betaB)resnet_model/batch_normalization_44/gammaB/resnet_model/batch_normalization_44/moving_meanB3resnet_model/batch_normalization_44/moving_varianceB(resnet_model/batch_normalization_45/betaB)resnet_model/batch_normalization_45/gammaB/resnet_model/batch_normalization_45/moving_meanB3resnet_model/batch_normalization_45/moving_varianceB(resnet_model/batch_normalization_46/betaB)resnet_model/batch_normalization_46/gammaB/resnet_model/batch_normalization_46/moving_meanB3resnet_model/batch_normalization_46/moving_varianceB(resnet_model/batch_normalization_47/betaB)resnet_model/batch_normalization_47/gammaB/resnet_model/batch_normalization_47/moving_meanB3resnet_model/batch_normalization_47/moving_varianceB(resnet_model/batch_normalization_48/betaB)resnet_model/batch_normalization_48/gammaB/resnet_model/batch_normalization_48/moving_meanB3resnet_model/batch_normalization_48/moving_varianceB'resnet_model/batch_normalization_5/betaB(resnet_model/batch_normalization_5/gammaB.resnet_model/batch_normalization_5/moving_meanB2resnet_model/batch_normalization_5/moving_varianceB'resnet_model/batch_normalization_6/betaB(resnet_model/batch_normalization_6/gammaB.resnet_model/batch_normalization_6/moving_meanB2resnet_model/batch_normalization_6/moving_varianceB'resnet_model/batch_normalization_7/betaB(resnet_model/batch_normalization_7/gammaB.resnet_model/batch_normalization_7/moving_meanB2resnet_model/batch_normalization_7/moving_varianceB'resnet_model/batch_normalization_8/betaB(resnet_model/batch_normalization_8/gammaB.resnet_model/batch_normalization_8/moving_meanB2resnet_model/batch_normalization_8/moving_varianceB'resnet_model/batch_normalization_9/betaB(resnet_model/batch_normalization_9/gammaB.resnet_model/batch_normalization_9/moving_meanB2resnet_model/batch_normalization_9/moving_varianceBresnet_model/conv2d/kernelBresnet_model/conv2d_1/kernelBresnet_model/conv2d_10/kernelBresnet_model/conv2d_11/kernelBresnet_model/conv2d_12/kernelBresnet_model/conv2d_13/kernelBresnet_model/conv2d_14/kernelBresnet_model/conv2d_15/kernelBresnet_model/conv2d_16/kernelBresnet_model/conv2d_17/kernelBresnet_model/conv2d_18/kernelBresnet_model/conv2d_19/kernelBresnet_model/conv2d_2/kernelBresnet_model/conv2d_20/kernelBresnet_model/conv2d_21/kernelBresnet_model/conv2d_22/kernelBresnet_model/conv2d_23/kernelBresnet_model/conv2d_24/kernelBresnet_model/conv2d_25/kernelBresnet_model/conv2d_26/kernelBresnet_model/conv2d_27/kernelBresnet_model/conv2d_28/kernelBresnet_model/conv2d_29/kernelBresnet_model/conv2d_3/kernelBresnet_model/conv2d_30/kernelBresnet_model/conv2d_31/kernelBresnet_model/conv2d_32/kernelBresnet_model/conv2d_33/kernelBresnet_model/conv2d_34/kernelBresnet_model/conv2d_35/kernelBresnet_model/conv2d_36/kernelBresnet_model/conv2d_37/kernelBresnet_model/conv2d_38/kernelBresnet_model/conv2d_39/kernelBresnet_model/conv2d_4/kernelBresnet_model/conv2d_40/kernelBresnet_model/conv2d_41/kernelBresnet_model/conv2d_42/kernelBresnet_model/conv2d_43/kernelBresnet_model/conv2d_44/kernelBresnet_model/conv2d_45/kernelBresnet_model/conv2d_46/kernelBresnet_model/conv2d_47/kernelBresnet_model/conv2d_48/kernelBresnet_model/conv2d_49/kernelBresnet_model/conv2d_5/kernelBresnet_model/conv2d_50/kernelBresnet_model/conv2d_51/kernelBresnet_model/conv2d_52/kernelBresnet_model/conv2d_6/kernelBresnet_model/conv2d_7/kernelBresnet_model/conv2d_8/kernelBresnet_model/conv2d_9/kernelBresnet_model/dense/biasBresnet_model/dense/kernel*
dtype0*
_output_shapes	
:û
ñ
 save_1/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:û*
valueBÿûB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
¿W
save_1/SaveV2_1SaveV2save_1/ShardedFilename_1save_1/SaveV2_1/tensor_names save_1/SaveV2_1/shape_and_slices%resnet_model/batch_normalization/beta&resnet_model/batch_normalization/gamma,resnet_model/batch_normalization/moving_mean0resnet_model/batch_normalization/moving_variance'resnet_model/batch_normalization_1/beta(resnet_model/batch_normalization_1/gamma.resnet_model/batch_normalization_1/moving_mean2resnet_model/batch_normalization_1/moving_variance(resnet_model/batch_normalization_10/beta)resnet_model/batch_normalization_10/gamma/resnet_model/batch_normalization_10/moving_mean3resnet_model/batch_normalization_10/moving_variance(resnet_model/batch_normalization_11/beta)resnet_model/batch_normalization_11/gamma/resnet_model/batch_normalization_11/moving_mean3resnet_model/batch_normalization_11/moving_variance(resnet_model/batch_normalization_12/beta)resnet_model/batch_normalization_12/gamma/resnet_model/batch_normalization_12/moving_mean3resnet_model/batch_normalization_12/moving_variance(resnet_model/batch_normalization_13/beta)resnet_model/batch_normalization_13/gamma/resnet_model/batch_normalization_13/moving_mean3resnet_model/batch_normalization_13/moving_variance(resnet_model/batch_normalization_14/beta)resnet_model/batch_normalization_14/gamma/resnet_model/batch_normalization_14/moving_mean3resnet_model/batch_normalization_14/moving_variance(resnet_model/batch_normalization_15/beta)resnet_model/batch_normalization_15/gamma/resnet_model/batch_normalization_15/moving_mean3resnet_model/batch_normalization_15/moving_variance(resnet_model/batch_normalization_16/beta)resnet_model/batch_normalization_16/gamma/resnet_model/batch_normalization_16/moving_mean3resnet_model/batch_normalization_16/moving_variance(resnet_model/batch_normalization_17/beta)resnet_model/batch_normalization_17/gamma/resnet_model/batch_normalization_17/moving_mean3resnet_model/batch_normalization_17/moving_variance(resnet_model/batch_normalization_18/beta)resnet_model/batch_normalization_18/gamma/resnet_model/batch_normalization_18/moving_mean3resnet_model/batch_normalization_18/moving_variance(resnet_model/batch_normalization_19/beta)resnet_model/batch_normalization_19/gamma/resnet_model/batch_normalization_19/moving_mean3resnet_model/batch_normalization_19/moving_variance'resnet_model/batch_normalization_2/beta(resnet_model/batch_normalization_2/gamma.resnet_model/batch_normalization_2/moving_mean2resnet_model/batch_normalization_2/moving_variance(resnet_model/batch_normalization_20/beta)resnet_model/batch_normalization_20/gamma/resnet_model/batch_normalization_20/moving_mean3resnet_model/batch_normalization_20/moving_variance(resnet_model/batch_normalization_21/beta)resnet_model/batch_normalization_21/gamma/resnet_model/batch_normalization_21/moving_mean3resnet_model/batch_normalization_21/moving_variance(resnet_model/batch_normalization_22/beta)resnet_model/batch_normalization_22/gamma/resnet_model/batch_normalization_22/moving_mean3resnet_model/batch_normalization_22/moving_variance(resnet_model/batch_normalization_23/beta)resnet_model/batch_normalization_23/gamma/resnet_model/batch_normalization_23/moving_mean3resnet_model/batch_normalization_23/moving_variance(resnet_model/batch_normalization_24/beta)resnet_model/batch_normalization_24/gamma/resnet_model/batch_normalization_24/moving_mean3resnet_model/batch_normalization_24/moving_variance(resnet_model/batch_normalization_25/beta)resnet_model/batch_normalization_25/gamma/resnet_model/batch_normalization_25/moving_mean3resnet_model/batch_normalization_25/moving_variance(resnet_model/batch_normalization_26/beta)resnet_model/batch_normalization_26/gamma/resnet_model/batch_normalization_26/moving_mean3resnet_model/batch_normalization_26/moving_variance(resnet_model/batch_normalization_27/beta)resnet_model/batch_normalization_27/gamma/resnet_model/batch_normalization_27/moving_mean3resnet_model/batch_normalization_27/moving_variance(resnet_model/batch_normalization_28/beta)resnet_model/batch_normalization_28/gamma/resnet_model/batch_normalization_28/moving_mean3resnet_model/batch_normalization_28/moving_variance(resnet_model/batch_normalization_29/beta)resnet_model/batch_normalization_29/gamma/resnet_model/batch_normalization_29/moving_mean3resnet_model/batch_normalization_29/moving_variance'resnet_model/batch_normalization_3/beta(resnet_model/batch_normalization_3/gamma.resnet_model/batch_normalization_3/moving_mean2resnet_model/batch_normalization_3/moving_variance(resnet_model/batch_normalization_30/beta)resnet_model/batch_normalization_30/gamma/resnet_model/batch_normalization_30/moving_mean3resnet_model/batch_normalization_30/moving_variance(resnet_model/batch_normalization_31/beta)resnet_model/batch_normalization_31/gamma/resnet_model/batch_normalization_31/moving_mean3resnet_model/batch_normalization_31/moving_variance(resnet_model/batch_normalization_32/beta)resnet_model/batch_normalization_32/gamma/resnet_model/batch_normalization_32/moving_mean3resnet_model/batch_normalization_32/moving_variance(resnet_model/batch_normalization_33/beta)resnet_model/batch_normalization_33/gamma/resnet_model/batch_normalization_33/moving_mean3resnet_model/batch_normalization_33/moving_variance(resnet_model/batch_normalization_34/beta)resnet_model/batch_normalization_34/gamma/resnet_model/batch_normalization_34/moving_mean3resnet_model/batch_normalization_34/moving_variance(resnet_model/batch_normalization_35/beta)resnet_model/batch_normalization_35/gamma/resnet_model/batch_normalization_35/moving_mean3resnet_model/batch_normalization_35/moving_variance(resnet_model/batch_normalization_36/beta)resnet_model/batch_normalization_36/gamma/resnet_model/batch_normalization_36/moving_mean3resnet_model/batch_normalization_36/moving_variance(resnet_model/batch_normalization_37/beta)resnet_model/batch_normalization_37/gamma/resnet_model/batch_normalization_37/moving_mean3resnet_model/batch_normalization_37/moving_variance(resnet_model/batch_normalization_38/beta)resnet_model/batch_normalization_38/gamma/resnet_model/batch_normalization_38/moving_mean3resnet_model/batch_normalization_38/moving_variance(resnet_model/batch_normalization_39/beta)resnet_model/batch_normalization_39/gamma/resnet_model/batch_normalization_39/moving_mean3resnet_model/batch_normalization_39/moving_variance'resnet_model/batch_normalization_4/beta(resnet_model/batch_normalization_4/gamma.resnet_model/batch_normalization_4/moving_mean2resnet_model/batch_normalization_4/moving_variance(resnet_model/batch_normalization_40/beta)resnet_model/batch_normalization_40/gamma/resnet_model/batch_normalization_40/moving_mean3resnet_model/batch_normalization_40/moving_variance(resnet_model/batch_normalization_41/beta)resnet_model/batch_normalization_41/gamma/resnet_model/batch_normalization_41/moving_mean3resnet_model/batch_normalization_41/moving_variance(resnet_model/batch_normalization_42/beta)resnet_model/batch_normalization_42/gamma/resnet_model/batch_normalization_42/moving_mean3resnet_model/batch_normalization_42/moving_variance(resnet_model/batch_normalization_43/beta)resnet_model/batch_normalization_43/gamma/resnet_model/batch_normalization_43/moving_mean3resnet_model/batch_normalization_43/moving_variance(resnet_model/batch_normalization_44/beta)resnet_model/batch_normalization_44/gamma/resnet_model/batch_normalization_44/moving_mean3resnet_model/batch_normalization_44/moving_variance(resnet_model/batch_normalization_45/beta)resnet_model/batch_normalization_45/gamma/resnet_model/batch_normalization_45/moving_mean3resnet_model/batch_normalization_45/moving_variance(resnet_model/batch_normalization_46/beta)resnet_model/batch_normalization_46/gamma/resnet_model/batch_normalization_46/moving_mean3resnet_model/batch_normalization_46/moving_variance(resnet_model/batch_normalization_47/beta)resnet_model/batch_normalization_47/gamma/resnet_model/batch_normalization_47/moving_mean3resnet_model/batch_normalization_47/moving_variance(resnet_model/batch_normalization_48/beta)resnet_model/batch_normalization_48/gamma/resnet_model/batch_normalization_48/moving_mean3resnet_model/batch_normalization_48/moving_variance'resnet_model/batch_normalization_5/beta(resnet_model/batch_normalization_5/gamma.resnet_model/batch_normalization_5/moving_mean2resnet_model/batch_normalization_5/moving_variance'resnet_model/batch_normalization_6/beta(resnet_model/batch_normalization_6/gamma.resnet_model/batch_normalization_6/moving_mean2resnet_model/batch_normalization_6/moving_variance'resnet_model/batch_normalization_7/beta(resnet_model/batch_normalization_7/gamma.resnet_model/batch_normalization_7/moving_mean2resnet_model/batch_normalization_7/moving_variance'resnet_model/batch_normalization_8/beta(resnet_model/batch_normalization_8/gamma.resnet_model/batch_normalization_8/moving_mean2resnet_model/batch_normalization_8/moving_variance'resnet_model/batch_normalization_9/beta(resnet_model/batch_normalization_9/gamma.resnet_model/batch_normalization_9/moving_mean2resnet_model/batch_normalization_9/moving_varianceresnet_model/conv2d/kernelresnet_model/conv2d_1/kernelresnet_model/conv2d_10/kernelresnet_model/conv2d_11/kernelresnet_model/conv2d_12/kernelresnet_model/conv2d_13/kernelresnet_model/conv2d_14/kernelresnet_model/conv2d_15/kernelresnet_model/conv2d_16/kernelresnet_model/conv2d_17/kernelresnet_model/conv2d_18/kernelresnet_model/conv2d_19/kernelresnet_model/conv2d_2/kernelresnet_model/conv2d_20/kernelresnet_model/conv2d_21/kernelresnet_model/conv2d_22/kernelresnet_model/conv2d_23/kernelresnet_model/conv2d_24/kernelresnet_model/conv2d_25/kernelresnet_model/conv2d_26/kernelresnet_model/conv2d_27/kernelresnet_model/conv2d_28/kernelresnet_model/conv2d_29/kernelresnet_model/conv2d_3/kernelresnet_model/conv2d_30/kernelresnet_model/conv2d_31/kernelresnet_model/conv2d_32/kernelresnet_model/conv2d_33/kernelresnet_model/conv2d_34/kernelresnet_model/conv2d_35/kernelresnet_model/conv2d_36/kernelresnet_model/conv2d_37/kernelresnet_model/conv2d_38/kernelresnet_model/conv2d_39/kernelresnet_model/conv2d_4/kernelresnet_model/conv2d_40/kernelresnet_model/conv2d_41/kernelresnet_model/conv2d_42/kernelresnet_model/conv2d_43/kernelresnet_model/conv2d_44/kernelresnet_model/conv2d_45/kernelresnet_model/conv2d_46/kernelresnet_model/conv2d_47/kernelresnet_model/conv2d_48/kernelresnet_model/conv2d_49/kernelresnet_model/conv2d_5/kernelresnet_model/conv2d_50/kernelresnet_model/conv2d_51/kernelresnet_model/conv2d_52/kernelresnet_model/conv2d_6/kernelresnet_model/conv2d_7/kernelresnet_model/conv2d_8/kernelresnet_model/conv2d_9/kernelresnet_model/dense/biasresnet_model/dense/kernel"/device:CPU:0*
dtypes
þ2û
°
save_1/control_dependency_1Identitysave_1/ShardedFilename_1^save_1/SaveV2_1"/device:CPU:0*
_output_shapes
: *
T0*+
_class!
loc:@save_1/ShardedFilename_1
ê
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilenamesave_1/ShardedFilename_1^save_1/control_dependency^save_1/control_dependency_1"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
¯
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency^save_1/control_dependency_1"/device:CPU:0*
_output_shapes
: *
T0

save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:* 
valueBBglobal_step
y
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
§
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2	
 
save_1/AssignAssignglobal_stepsave_1/RestoreV2*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
,
save_1/restore_shardNoOp^save_1/Assign
¨U
save_1/RestoreV2_1/tensor_namesConst"/device:CPU:0*ÄT
valueºTB·TûB%resnet_model/batch_normalization/betaB&resnet_model/batch_normalization/gammaB,resnet_model/batch_normalization/moving_meanB0resnet_model/batch_normalization/moving_varianceB'resnet_model/batch_normalization_1/betaB(resnet_model/batch_normalization_1/gammaB.resnet_model/batch_normalization_1/moving_meanB2resnet_model/batch_normalization_1/moving_varianceB(resnet_model/batch_normalization_10/betaB)resnet_model/batch_normalization_10/gammaB/resnet_model/batch_normalization_10/moving_meanB3resnet_model/batch_normalization_10/moving_varianceB(resnet_model/batch_normalization_11/betaB)resnet_model/batch_normalization_11/gammaB/resnet_model/batch_normalization_11/moving_meanB3resnet_model/batch_normalization_11/moving_varianceB(resnet_model/batch_normalization_12/betaB)resnet_model/batch_normalization_12/gammaB/resnet_model/batch_normalization_12/moving_meanB3resnet_model/batch_normalization_12/moving_varianceB(resnet_model/batch_normalization_13/betaB)resnet_model/batch_normalization_13/gammaB/resnet_model/batch_normalization_13/moving_meanB3resnet_model/batch_normalization_13/moving_varianceB(resnet_model/batch_normalization_14/betaB)resnet_model/batch_normalization_14/gammaB/resnet_model/batch_normalization_14/moving_meanB3resnet_model/batch_normalization_14/moving_varianceB(resnet_model/batch_normalization_15/betaB)resnet_model/batch_normalization_15/gammaB/resnet_model/batch_normalization_15/moving_meanB3resnet_model/batch_normalization_15/moving_varianceB(resnet_model/batch_normalization_16/betaB)resnet_model/batch_normalization_16/gammaB/resnet_model/batch_normalization_16/moving_meanB3resnet_model/batch_normalization_16/moving_varianceB(resnet_model/batch_normalization_17/betaB)resnet_model/batch_normalization_17/gammaB/resnet_model/batch_normalization_17/moving_meanB3resnet_model/batch_normalization_17/moving_varianceB(resnet_model/batch_normalization_18/betaB)resnet_model/batch_normalization_18/gammaB/resnet_model/batch_normalization_18/moving_meanB3resnet_model/batch_normalization_18/moving_varianceB(resnet_model/batch_normalization_19/betaB)resnet_model/batch_normalization_19/gammaB/resnet_model/batch_normalization_19/moving_meanB3resnet_model/batch_normalization_19/moving_varianceB'resnet_model/batch_normalization_2/betaB(resnet_model/batch_normalization_2/gammaB.resnet_model/batch_normalization_2/moving_meanB2resnet_model/batch_normalization_2/moving_varianceB(resnet_model/batch_normalization_20/betaB)resnet_model/batch_normalization_20/gammaB/resnet_model/batch_normalization_20/moving_meanB3resnet_model/batch_normalization_20/moving_varianceB(resnet_model/batch_normalization_21/betaB)resnet_model/batch_normalization_21/gammaB/resnet_model/batch_normalization_21/moving_meanB3resnet_model/batch_normalization_21/moving_varianceB(resnet_model/batch_normalization_22/betaB)resnet_model/batch_normalization_22/gammaB/resnet_model/batch_normalization_22/moving_meanB3resnet_model/batch_normalization_22/moving_varianceB(resnet_model/batch_normalization_23/betaB)resnet_model/batch_normalization_23/gammaB/resnet_model/batch_normalization_23/moving_meanB3resnet_model/batch_normalization_23/moving_varianceB(resnet_model/batch_normalization_24/betaB)resnet_model/batch_normalization_24/gammaB/resnet_model/batch_normalization_24/moving_meanB3resnet_model/batch_normalization_24/moving_varianceB(resnet_model/batch_normalization_25/betaB)resnet_model/batch_normalization_25/gammaB/resnet_model/batch_normalization_25/moving_meanB3resnet_model/batch_normalization_25/moving_varianceB(resnet_model/batch_normalization_26/betaB)resnet_model/batch_normalization_26/gammaB/resnet_model/batch_normalization_26/moving_meanB3resnet_model/batch_normalization_26/moving_varianceB(resnet_model/batch_normalization_27/betaB)resnet_model/batch_normalization_27/gammaB/resnet_model/batch_normalization_27/moving_meanB3resnet_model/batch_normalization_27/moving_varianceB(resnet_model/batch_normalization_28/betaB)resnet_model/batch_normalization_28/gammaB/resnet_model/batch_normalization_28/moving_meanB3resnet_model/batch_normalization_28/moving_varianceB(resnet_model/batch_normalization_29/betaB)resnet_model/batch_normalization_29/gammaB/resnet_model/batch_normalization_29/moving_meanB3resnet_model/batch_normalization_29/moving_varianceB'resnet_model/batch_normalization_3/betaB(resnet_model/batch_normalization_3/gammaB.resnet_model/batch_normalization_3/moving_meanB2resnet_model/batch_normalization_3/moving_varianceB(resnet_model/batch_normalization_30/betaB)resnet_model/batch_normalization_30/gammaB/resnet_model/batch_normalization_30/moving_meanB3resnet_model/batch_normalization_30/moving_varianceB(resnet_model/batch_normalization_31/betaB)resnet_model/batch_normalization_31/gammaB/resnet_model/batch_normalization_31/moving_meanB3resnet_model/batch_normalization_31/moving_varianceB(resnet_model/batch_normalization_32/betaB)resnet_model/batch_normalization_32/gammaB/resnet_model/batch_normalization_32/moving_meanB3resnet_model/batch_normalization_32/moving_varianceB(resnet_model/batch_normalization_33/betaB)resnet_model/batch_normalization_33/gammaB/resnet_model/batch_normalization_33/moving_meanB3resnet_model/batch_normalization_33/moving_varianceB(resnet_model/batch_normalization_34/betaB)resnet_model/batch_normalization_34/gammaB/resnet_model/batch_normalization_34/moving_meanB3resnet_model/batch_normalization_34/moving_varianceB(resnet_model/batch_normalization_35/betaB)resnet_model/batch_normalization_35/gammaB/resnet_model/batch_normalization_35/moving_meanB3resnet_model/batch_normalization_35/moving_varianceB(resnet_model/batch_normalization_36/betaB)resnet_model/batch_normalization_36/gammaB/resnet_model/batch_normalization_36/moving_meanB3resnet_model/batch_normalization_36/moving_varianceB(resnet_model/batch_normalization_37/betaB)resnet_model/batch_normalization_37/gammaB/resnet_model/batch_normalization_37/moving_meanB3resnet_model/batch_normalization_37/moving_varianceB(resnet_model/batch_normalization_38/betaB)resnet_model/batch_normalization_38/gammaB/resnet_model/batch_normalization_38/moving_meanB3resnet_model/batch_normalization_38/moving_varianceB(resnet_model/batch_normalization_39/betaB)resnet_model/batch_normalization_39/gammaB/resnet_model/batch_normalization_39/moving_meanB3resnet_model/batch_normalization_39/moving_varianceB'resnet_model/batch_normalization_4/betaB(resnet_model/batch_normalization_4/gammaB.resnet_model/batch_normalization_4/moving_meanB2resnet_model/batch_normalization_4/moving_varianceB(resnet_model/batch_normalization_40/betaB)resnet_model/batch_normalization_40/gammaB/resnet_model/batch_normalization_40/moving_meanB3resnet_model/batch_normalization_40/moving_varianceB(resnet_model/batch_normalization_41/betaB)resnet_model/batch_normalization_41/gammaB/resnet_model/batch_normalization_41/moving_meanB3resnet_model/batch_normalization_41/moving_varianceB(resnet_model/batch_normalization_42/betaB)resnet_model/batch_normalization_42/gammaB/resnet_model/batch_normalization_42/moving_meanB3resnet_model/batch_normalization_42/moving_varianceB(resnet_model/batch_normalization_43/betaB)resnet_model/batch_normalization_43/gammaB/resnet_model/batch_normalization_43/moving_meanB3resnet_model/batch_normalization_43/moving_varianceB(resnet_model/batch_normalization_44/betaB)resnet_model/batch_normalization_44/gammaB/resnet_model/batch_normalization_44/moving_meanB3resnet_model/batch_normalization_44/moving_varianceB(resnet_model/batch_normalization_45/betaB)resnet_model/batch_normalization_45/gammaB/resnet_model/batch_normalization_45/moving_meanB3resnet_model/batch_normalization_45/moving_varianceB(resnet_model/batch_normalization_46/betaB)resnet_model/batch_normalization_46/gammaB/resnet_model/batch_normalization_46/moving_meanB3resnet_model/batch_normalization_46/moving_varianceB(resnet_model/batch_normalization_47/betaB)resnet_model/batch_normalization_47/gammaB/resnet_model/batch_normalization_47/moving_meanB3resnet_model/batch_normalization_47/moving_varianceB(resnet_model/batch_normalization_48/betaB)resnet_model/batch_normalization_48/gammaB/resnet_model/batch_normalization_48/moving_meanB3resnet_model/batch_normalization_48/moving_varianceB'resnet_model/batch_normalization_5/betaB(resnet_model/batch_normalization_5/gammaB.resnet_model/batch_normalization_5/moving_meanB2resnet_model/batch_normalization_5/moving_varianceB'resnet_model/batch_normalization_6/betaB(resnet_model/batch_normalization_6/gammaB.resnet_model/batch_normalization_6/moving_meanB2resnet_model/batch_normalization_6/moving_varianceB'resnet_model/batch_normalization_7/betaB(resnet_model/batch_normalization_7/gammaB.resnet_model/batch_normalization_7/moving_meanB2resnet_model/batch_normalization_7/moving_varianceB'resnet_model/batch_normalization_8/betaB(resnet_model/batch_normalization_8/gammaB.resnet_model/batch_normalization_8/moving_meanB2resnet_model/batch_normalization_8/moving_varianceB'resnet_model/batch_normalization_9/betaB(resnet_model/batch_normalization_9/gammaB.resnet_model/batch_normalization_9/moving_meanB2resnet_model/batch_normalization_9/moving_varianceBresnet_model/conv2d/kernelBresnet_model/conv2d_1/kernelBresnet_model/conv2d_10/kernelBresnet_model/conv2d_11/kernelBresnet_model/conv2d_12/kernelBresnet_model/conv2d_13/kernelBresnet_model/conv2d_14/kernelBresnet_model/conv2d_15/kernelBresnet_model/conv2d_16/kernelBresnet_model/conv2d_17/kernelBresnet_model/conv2d_18/kernelBresnet_model/conv2d_19/kernelBresnet_model/conv2d_2/kernelBresnet_model/conv2d_20/kernelBresnet_model/conv2d_21/kernelBresnet_model/conv2d_22/kernelBresnet_model/conv2d_23/kernelBresnet_model/conv2d_24/kernelBresnet_model/conv2d_25/kernelBresnet_model/conv2d_26/kernelBresnet_model/conv2d_27/kernelBresnet_model/conv2d_28/kernelBresnet_model/conv2d_29/kernelBresnet_model/conv2d_3/kernelBresnet_model/conv2d_30/kernelBresnet_model/conv2d_31/kernelBresnet_model/conv2d_32/kernelBresnet_model/conv2d_33/kernelBresnet_model/conv2d_34/kernelBresnet_model/conv2d_35/kernelBresnet_model/conv2d_36/kernelBresnet_model/conv2d_37/kernelBresnet_model/conv2d_38/kernelBresnet_model/conv2d_39/kernelBresnet_model/conv2d_4/kernelBresnet_model/conv2d_40/kernelBresnet_model/conv2d_41/kernelBresnet_model/conv2d_42/kernelBresnet_model/conv2d_43/kernelBresnet_model/conv2d_44/kernelBresnet_model/conv2d_45/kernelBresnet_model/conv2d_46/kernelBresnet_model/conv2d_47/kernelBresnet_model/conv2d_48/kernelBresnet_model/conv2d_49/kernelBresnet_model/conv2d_5/kernelBresnet_model/conv2d_50/kernelBresnet_model/conv2d_51/kernelBresnet_model/conv2d_52/kernelBresnet_model/conv2d_6/kernelBresnet_model/conv2d_7/kernelBresnet_model/conv2d_8/kernelBresnet_model/conv2d_9/kernelBresnet_model/dense/biasBresnet_model/dense/kernel*
dtype0*
_output_shapes	
:û
ô
#save_1/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueBÿûB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:û

save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices"/device:CPU:0*
_output_shapesï
ì:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
þ2û
ë
save_1/Assign_1Assign%resnet_model/batch_normalization/betasave_1/RestoreV2_1"/device:CPU:0*8
_class.
,*loc:@resnet_model/batch_normalization/beta*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
ï
save_1/Assign_2Assign&resnet_model/batch_normalization/gammasave_1/RestoreV2_1:1"/device:CPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*9
_class/
-+loc:@resnet_model/batch_normalization/gamma
û
save_1/Assign_3Assign,resnet_model/batch_normalization/moving_meansave_1/RestoreV2_1:2"/device:CPU:0*
use_locking(*
T0*?
_class5
31loc:@resnet_model/batch_normalization/moving_mean*
validate_shape(*
_output_shapes
:@

save_1/Assign_4Assign0resnet_model/batch_normalization/moving_variancesave_1/RestoreV2_1:3"/device:CPU:0*
use_locking(*
T0*C
_class9
75loc:@resnet_model/batch_normalization/moving_variance*
validate_shape(*
_output_shapes
:@
ñ
save_1/Assign_5Assign'resnet_model/batch_normalization_1/betasave_1/RestoreV2_1:4"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_1/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
ó
save_1/Assign_6Assign(resnet_model/batch_normalization_1/gammasave_1/RestoreV2_1:5"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_1/gamma*
validate_shape(*
_output_shapes
:@
ÿ
save_1/Assign_7Assign.resnet_model/batch_normalization_1/moving_meansave_1/RestoreV2_1:6"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_1/moving_mean*
validate_shape(*
_output_shapes
:@

save_1/Assign_8Assign2resnet_model/batch_normalization_1/moving_variancesave_1/RestoreV2_1:7"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_1/moving_variance*
validate_shape(*
_output_shapes
:@
ô
save_1/Assign_9Assign(resnet_model/batch_normalization_10/betasave_1/RestoreV2_1:8"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_10/beta*
validate_shape(*
_output_shapes	
:
÷
save_1/Assign_10Assign)resnet_model/batch_normalization_10/gammasave_1/RestoreV2_1:9"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_10/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_11Assign/resnet_model/batch_normalization_10/moving_meansave_1/RestoreV2_1:10"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_10/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_12Assign3resnet_model/batch_normalization_10/moving_variancesave_1/RestoreV2_1:11"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_10/moving_variance*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_13Assign(resnet_model/batch_normalization_11/betasave_1/RestoreV2_1:12"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_11/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_14Assign)resnet_model/batch_normalization_11/gammasave_1/RestoreV2_1:13"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_11/gamma*
validate_shape(

save_1/Assign_15Assign/resnet_model/batch_normalization_11/moving_meansave_1/RestoreV2_1:14"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_11/moving_mean*
validate_shape(

save_1/Assign_16Assign3resnet_model/batch_normalization_11/moving_variancesave_1/RestoreV2_1:15"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_11/moving_variance*
validate_shape(
ö
save_1/Assign_17Assign(resnet_model/batch_normalization_12/betasave_1/RestoreV2_1:16"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_12/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_18Assign)resnet_model/batch_normalization_12/gammasave_1/RestoreV2_1:17"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_12/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_19Assign/resnet_model/batch_normalization_12/moving_meansave_1/RestoreV2_1:18"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_12/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_20Assign3resnet_model/batch_normalization_12/moving_variancesave_1/RestoreV2_1:19"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_12/moving_variance*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_21Assign(resnet_model/batch_normalization_13/betasave_1/RestoreV2_1:20"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_13/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_22Assign)resnet_model/batch_normalization_13/gammasave_1/RestoreV2_1:21"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_13/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_23Assign/resnet_model/batch_normalization_13/moving_meansave_1/RestoreV2_1:22"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_13/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_24Assign3resnet_model/batch_normalization_13/moving_variancesave_1/RestoreV2_1:23"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_13/moving_variance
ö
save_1/Assign_25Assign(resnet_model/batch_normalization_14/betasave_1/RestoreV2_1:24"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_14/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_26Assign)resnet_model/batch_normalization_14/gammasave_1/RestoreV2_1:25"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_14/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_27Assign/resnet_model/batch_normalization_14/moving_meansave_1/RestoreV2_1:26"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_14/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_28Assign3resnet_model/batch_normalization_14/moving_variancesave_1/RestoreV2_1:27"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_14/moving_variance
ö
save_1/Assign_29Assign(resnet_model/batch_normalization_15/betasave_1/RestoreV2_1:28"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_15/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_30Assign)resnet_model/batch_normalization_15/gammasave_1/RestoreV2_1:29"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_15/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_31Assign/resnet_model/batch_normalization_15/moving_meansave_1/RestoreV2_1:30"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_15/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_32Assign3resnet_model/batch_normalization_15/moving_variancesave_1/RestoreV2_1:31"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_15/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
save_1/Assign_33Assign(resnet_model/batch_normalization_16/betasave_1/RestoreV2_1:32"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_16/beta
ø
save_1/Assign_34Assign)resnet_model/batch_normalization_16/gammasave_1/RestoreV2_1:33"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_16/gamma*
validate_shape(

save_1/Assign_35Assign/resnet_model/batch_normalization_16/moving_meansave_1/RestoreV2_1:34"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_16/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_36Assign3resnet_model/batch_normalization_16/moving_variancesave_1/RestoreV2_1:35"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_16/moving_variance*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_37Assign(resnet_model/batch_normalization_17/betasave_1/RestoreV2_1:36"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_17/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_38Assign)resnet_model/batch_normalization_17/gammasave_1/RestoreV2_1:37"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_17/gamma*
validate_shape(

save_1/Assign_39Assign/resnet_model/batch_normalization_17/moving_meansave_1/RestoreV2_1:38"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_17/moving_mean

save_1/Assign_40Assign3resnet_model/batch_normalization_17/moving_variancesave_1/RestoreV2_1:39"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_17/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
save_1/Assign_41Assign(resnet_model/batch_normalization_18/betasave_1/RestoreV2_1:40"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_18/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save_1/Assign_42Assign)resnet_model/batch_normalization_18/gammasave_1/RestoreV2_1:41"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_18/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_43Assign/resnet_model/batch_normalization_18/moving_meansave_1/RestoreV2_1:42"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_18/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_44Assign3resnet_model/batch_normalization_18/moving_variancesave_1/RestoreV2_1:43"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_18/moving_variance*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_45Assign(resnet_model/batch_normalization_19/betasave_1/RestoreV2_1:44"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_19/beta*
validate_shape(
ø
save_1/Assign_46Assign)resnet_model/batch_normalization_19/gammasave_1/RestoreV2_1:45"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_19/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_47Assign/resnet_model/batch_normalization_19/moving_meansave_1/RestoreV2_1:46"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_19/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_48Assign3resnet_model/batch_normalization_19/moving_variancesave_1/RestoreV2_1:47"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_19/moving_variance*
validate_shape(*
_output_shapes	
:
ó
save_1/Assign_49Assign'resnet_model/batch_normalization_2/betasave_1/RestoreV2_1:48"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_2/beta*
validate_shape(*
_output_shapes
:@
õ
save_1/Assign_50Assign(resnet_model/batch_normalization_2/gammasave_1/RestoreV2_1:49"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_2/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

save_1/Assign_51Assign.resnet_model/batch_normalization_2/moving_meansave_1/RestoreV2_1:50"/device:CPU:0*A
_class7
53loc:@resnet_model/batch_normalization_2/moving_mean*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

save_1/Assign_52Assign2resnet_model/batch_normalization_2/moving_variancesave_1/RestoreV2_1:51"/device:CPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_2/moving_variance
ö
save_1/Assign_53Assign(resnet_model/batch_normalization_20/betasave_1/RestoreV2_1:52"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_20/beta*
validate_shape(
ø
save_1/Assign_54Assign)resnet_model/batch_normalization_20/gammasave_1/RestoreV2_1:53"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_20/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_55Assign/resnet_model/batch_normalization_20/moving_meansave_1/RestoreV2_1:54"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_20/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_56Assign3resnet_model/batch_normalization_20/moving_variancesave_1/RestoreV2_1:55"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_20/moving_variance*
validate_shape(
ö
save_1/Assign_57Assign(resnet_model/batch_normalization_21/betasave_1/RestoreV2_1:56"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_21/beta*
validate_shape(
ø
save_1/Assign_58Assign)resnet_model/batch_normalization_21/gammasave_1/RestoreV2_1:57"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_21/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_59Assign/resnet_model/batch_normalization_21/moving_meansave_1/RestoreV2_1:58"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_21/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_60Assign3resnet_model/batch_normalization_21/moving_variancesave_1/RestoreV2_1:59"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_21/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
save_1/Assign_61Assign(resnet_model/batch_normalization_22/betasave_1/RestoreV2_1:60"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_22/beta
ø
save_1/Assign_62Assign)resnet_model/batch_normalization_22/gammasave_1/RestoreV2_1:61"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_22/gamma*
validate_shape(

save_1/Assign_63Assign/resnet_model/batch_normalization_22/moving_meansave_1/RestoreV2_1:62"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_22/moving_mean

save_1/Assign_64Assign3resnet_model/batch_normalization_22/moving_variancesave_1/RestoreV2_1:63"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_22/moving_variance*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_65Assign(resnet_model/batch_normalization_23/betasave_1/RestoreV2_1:64"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_23/beta*
validate_shape(
ø
save_1/Assign_66Assign)resnet_model/batch_normalization_23/gammasave_1/RestoreV2_1:65"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_23/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_67Assign/resnet_model/batch_normalization_23/moving_meansave_1/RestoreV2_1:66"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_23/moving_mean*
validate_shape(

save_1/Assign_68Assign3resnet_model/batch_normalization_23/moving_variancesave_1/RestoreV2_1:67"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_23/moving_variance*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_69Assign(resnet_model/batch_normalization_24/betasave_1/RestoreV2_1:68"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_24/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_70Assign)resnet_model/batch_normalization_24/gammasave_1/RestoreV2_1:69"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_24/gamma

save_1/Assign_71Assign/resnet_model/batch_normalization_24/moving_meansave_1/RestoreV2_1:70"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_24/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_72Assign3resnet_model/batch_normalization_24/moving_variancesave_1/RestoreV2_1:71"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_24/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
save_1/Assign_73Assign(resnet_model/batch_normalization_25/betasave_1/RestoreV2_1:72"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_25/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_74Assign)resnet_model/batch_normalization_25/gammasave_1/RestoreV2_1:73"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_25/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_75Assign/resnet_model/batch_normalization_25/moving_meansave_1/RestoreV2_1:74"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_25/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_76Assign3resnet_model/batch_normalization_25/moving_variancesave_1/RestoreV2_1:75"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_25/moving_variance*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_77Assign(resnet_model/batch_normalization_26/betasave_1/RestoreV2_1:76"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_26/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_78Assign)resnet_model/batch_normalization_26/gammasave_1/RestoreV2_1:77"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_26/gamma

save_1/Assign_79Assign/resnet_model/batch_normalization_26/moving_meansave_1/RestoreV2_1:78"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_26/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_80Assign3resnet_model/batch_normalization_26/moving_variancesave_1/RestoreV2_1:79"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_26/moving_variance*
validate_shape(
ö
save_1/Assign_81Assign(resnet_model/batch_normalization_27/betasave_1/RestoreV2_1:80"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_27/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save_1/Assign_82Assign)resnet_model/batch_normalization_27/gammasave_1/RestoreV2_1:81"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_27/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_83Assign/resnet_model/batch_normalization_27/moving_meansave_1/RestoreV2_1:82"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_27/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_84Assign3resnet_model/batch_normalization_27/moving_variancesave_1/RestoreV2_1:83"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_27/moving_variance
ö
save_1/Assign_85Assign(resnet_model/batch_normalization_28/betasave_1/RestoreV2_1:84"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_28/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save_1/Assign_86Assign)resnet_model/batch_normalization_28/gammasave_1/RestoreV2_1:85"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_28/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_87Assign/resnet_model/batch_normalization_28/moving_meansave_1/RestoreV2_1:86"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_28/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_88Assign3resnet_model/batch_normalization_28/moving_variancesave_1/RestoreV2_1:87"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_28/moving_variance*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_89Assign(resnet_model/batch_normalization_29/betasave_1/RestoreV2_1:88"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_29/beta
ø
save_1/Assign_90Assign)resnet_model/batch_normalization_29/gammasave_1/RestoreV2_1:89"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_29/gamma

save_1/Assign_91Assign/resnet_model/batch_normalization_29/moving_meansave_1/RestoreV2_1:90"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_29/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_92Assign3resnet_model/batch_normalization_29/moving_variancesave_1/RestoreV2_1:91"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_29/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ô
save_1/Assign_93Assign'resnet_model/batch_normalization_3/betasave_1/RestoreV2_1:92"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_3/beta*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_94Assign(resnet_model/batch_normalization_3/gammasave_1/RestoreV2_1:93"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_3/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_95Assign.resnet_model/batch_normalization_3/moving_meansave_1/RestoreV2_1:94"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_3/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_96Assign2resnet_model/batch_normalization_3/moving_variancesave_1/RestoreV2_1:95"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_3/moving_variance*
validate_shape(
ö
save_1/Assign_97Assign(resnet_model/batch_normalization_30/betasave_1/RestoreV2_1:96"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_30/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_98Assign)resnet_model/batch_normalization_30/gammasave_1/RestoreV2_1:97"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_30/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_99Assign/resnet_model/batch_normalization_30/moving_meansave_1/RestoreV2_1:98"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_30/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_100Assign3resnet_model/batch_normalization_30/moving_variancesave_1/RestoreV2_1:99"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_30/moving_variance*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_101Assign(resnet_model/batch_normalization_31/betasave_1/RestoreV2_1:100"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_31/beta*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_102Assign)resnet_model/batch_normalization_31/gammasave_1/RestoreV2_1:101"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_31/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_103Assign/resnet_model/batch_normalization_31/moving_meansave_1/RestoreV2_1:102"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_31/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_104Assign3resnet_model/batch_normalization_31/moving_variancesave_1/RestoreV2_1:103"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_31/moving_variance
ø
save_1/Assign_105Assign(resnet_model/batch_normalization_32/betasave_1/RestoreV2_1:104"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_32/beta*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_106Assign)resnet_model/batch_normalization_32/gammasave_1/RestoreV2_1:105"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_32/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_107Assign/resnet_model/batch_normalization_32/moving_meansave_1/RestoreV2_1:106"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_32/moving_mean*
validate_shape(

save_1/Assign_108Assign3resnet_model/batch_normalization_32/moving_variancesave_1/RestoreV2_1:107"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_32/moving_variance
ø
save_1/Assign_109Assign(resnet_model/batch_normalization_33/betasave_1/RestoreV2_1:108"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_33/beta*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_110Assign)resnet_model/batch_normalization_33/gammasave_1/RestoreV2_1:109"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_33/gamma

save_1/Assign_111Assign/resnet_model/batch_normalization_33/moving_meansave_1/RestoreV2_1:110"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_33/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_112Assign3resnet_model/batch_normalization_33/moving_variancesave_1/RestoreV2_1:111"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_33/moving_variance
ø
save_1/Assign_113Assign(resnet_model/batch_normalization_34/betasave_1/RestoreV2_1:112"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_34/beta*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_114Assign)resnet_model/batch_normalization_34/gammasave_1/RestoreV2_1:113"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_34/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_115Assign/resnet_model/batch_normalization_34/moving_meansave_1/RestoreV2_1:114"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_34/moving_mean*
validate_shape(

save_1/Assign_116Assign3resnet_model/batch_normalization_34/moving_variancesave_1/RestoreV2_1:115"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_34/moving_variance*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_117Assign(resnet_model/batch_normalization_35/betasave_1/RestoreV2_1:116"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_35/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ú
save_1/Assign_118Assign)resnet_model/batch_normalization_35/gammasave_1/RestoreV2_1:117"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_35/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_119Assign/resnet_model/batch_normalization_35/moving_meansave_1/RestoreV2_1:118"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_35/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_120Assign3resnet_model/batch_normalization_35/moving_variancesave_1/RestoreV2_1:119"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_35/moving_variance*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_121Assign(resnet_model/batch_normalization_36/betasave_1/RestoreV2_1:120"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_36/beta*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_122Assign)resnet_model/batch_normalization_36/gammasave_1/RestoreV2_1:121"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_36/gamma*
validate_shape(

save_1/Assign_123Assign/resnet_model/batch_normalization_36/moving_meansave_1/RestoreV2_1:122"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_36/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_124Assign3resnet_model/batch_normalization_36/moving_variancesave_1/RestoreV2_1:123"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_36/moving_variance*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_125Assign(resnet_model/batch_normalization_37/betasave_1/RestoreV2_1:124"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_37/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ú
save_1/Assign_126Assign)resnet_model/batch_normalization_37/gammasave_1/RestoreV2_1:125"/device:CPU:0*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_37/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_127Assign/resnet_model/batch_normalization_37/moving_meansave_1/RestoreV2_1:126"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_37/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_128Assign3resnet_model/batch_normalization_37/moving_variancesave_1/RestoreV2_1:127"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_37/moving_variance*
validate_shape(
ø
save_1/Assign_129Assign(resnet_model/batch_normalization_38/betasave_1/RestoreV2_1:128"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_38/beta*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_130Assign)resnet_model/batch_normalization_38/gammasave_1/RestoreV2_1:129"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_38/gamma

save_1/Assign_131Assign/resnet_model/batch_normalization_38/moving_meansave_1/RestoreV2_1:130"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_38/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_132Assign3resnet_model/batch_normalization_38/moving_variancesave_1/RestoreV2_1:131"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_38/moving_variance*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_133Assign(resnet_model/batch_normalization_39/betasave_1/RestoreV2_1:132"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_39/beta
ú
save_1/Assign_134Assign)resnet_model/batch_normalization_39/gammasave_1/RestoreV2_1:133"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_39/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_135Assign/resnet_model/batch_normalization_39/moving_meansave_1/RestoreV2_1:134"/device:CPU:0*
T0*B
_class8
64loc:@resnet_model/batch_normalization_39/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_136Assign3resnet_model/batch_normalization_39/moving_variancesave_1/RestoreV2_1:135"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_39/moving_variance
õ
save_1/Assign_137Assign'resnet_model/batch_normalization_4/betasave_1/RestoreV2_1:136"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_4/beta*
validate_shape(*
_output_shapes
:@
÷
save_1/Assign_138Assign(resnet_model/batch_normalization_4/gammasave_1/RestoreV2_1:137"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_4/gamma*
validate_shape(*
_output_shapes
:@

save_1/Assign_139Assign.resnet_model/batch_normalization_4/moving_meansave_1/RestoreV2_1:138"/device:CPU:0*
_output_shapes
:@*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_4/moving_mean*
validate_shape(

save_1/Assign_140Assign2resnet_model/batch_normalization_4/moving_variancesave_1/RestoreV2_1:139"/device:CPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_4/moving_variance
ø
save_1/Assign_141Assign(resnet_model/batch_normalization_40/betasave_1/RestoreV2_1:140"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_40/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ú
save_1/Assign_142Assign)resnet_model/batch_normalization_40/gammasave_1/RestoreV2_1:141"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_40/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_143Assign/resnet_model/batch_normalization_40/moving_meansave_1/RestoreV2_1:142"/device:CPU:0*B
_class8
64loc:@resnet_model/batch_normalization_40/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_144Assign3resnet_model/batch_normalization_40/moving_variancesave_1/RestoreV2_1:143"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_40/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save_1/Assign_145Assign(resnet_model/batch_normalization_41/betasave_1/RestoreV2_1:144"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_41/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ú
save_1/Assign_146Assign)resnet_model/batch_normalization_41/gammasave_1/RestoreV2_1:145"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_41/gamma

save_1/Assign_147Assign/resnet_model/batch_normalization_41/moving_meansave_1/RestoreV2_1:146"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_41/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_148Assign3resnet_model/batch_normalization_41/moving_variancesave_1/RestoreV2_1:147"/device:CPU:0*F
_class<
:8loc:@resnet_model/batch_normalization_41/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ø
save_1/Assign_149Assign(resnet_model/batch_normalization_42/betasave_1/RestoreV2_1:148"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_42/beta
ú
save_1/Assign_150Assign)resnet_model/batch_normalization_42/gammasave_1/RestoreV2_1:149"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_42/gamma*
validate_shape(

save_1/Assign_151Assign/resnet_model/batch_normalization_42/moving_meansave_1/RestoreV2_1:150"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_42/moving_mean*
validate_shape(

save_1/Assign_152Assign3resnet_model/batch_normalization_42/moving_variancesave_1/RestoreV2_1:151"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_42/moving_variance*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_153Assign(resnet_model/batch_normalization_43/betasave_1/RestoreV2_1:152"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_43/beta*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_154Assign)resnet_model/batch_normalization_43/gammasave_1/RestoreV2_1:153"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_43/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_155Assign/resnet_model/batch_normalization_43/moving_meansave_1/RestoreV2_1:154"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_43/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_156Assign3resnet_model/batch_normalization_43/moving_variancesave_1/RestoreV2_1:155"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_43/moving_variance*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_157Assign(resnet_model/batch_normalization_44/betasave_1/RestoreV2_1:156"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_44/beta*
validate_shape(
ú
save_1/Assign_158Assign)resnet_model/batch_normalization_44/gammasave_1/RestoreV2_1:157"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_44/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_159Assign/resnet_model/batch_normalization_44/moving_meansave_1/RestoreV2_1:158"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_44/moving_mean

save_1/Assign_160Assign3resnet_model/batch_normalization_44/moving_variancesave_1/RestoreV2_1:159"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_44/moving_variance*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_161Assign(resnet_model/batch_normalization_45/betasave_1/RestoreV2_1:160"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_45/beta*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_162Assign)resnet_model/batch_normalization_45/gammasave_1/RestoreV2_1:161"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_45/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_163Assign/resnet_model/batch_normalization_45/moving_meansave_1/RestoreV2_1:162"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_45/moving_mean*
validate_shape(

save_1/Assign_164Assign3resnet_model/batch_normalization_45/moving_variancesave_1/RestoreV2_1:163"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_45/moving_variance*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_165Assign(resnet_model/batch_normalization_46/betasave_1/RestoreV2_1:164"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_46/beta*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_166Assign)resnet_model/batch_normalization_46/gammasave_1/RestoreV2_1:165"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_46/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_167Assign/resnet_model/batch_normalization_46/moving_meansave_1/RestoreV2_1:166"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_46/moving_mean*
validate_shape(

save_1/Assign_168Assign3resnet_model/batch_normalization_46/moving_variancesave_1/RestoreV2_1:167"/device:CPU:0*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_46/moving_variance*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_169Assign(resnet_model/batch_normalization_47/betasave_1/RestoreV2_1:168"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_47/beta*
validate_shape(
ú
save_1/Assign_170Assign)resnet_model/batch_normalization_47/gammasave_1/RestoreV2_1:169"/device:CPU:0*
use_locking(*
T0*<
_class2
0.loc:@resnet_model/batch_normalization_47/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_171Assign/resnet_model/batch_normalization_47/moving_meansave_1/RestoreV2_1:170"/device:CPU:0*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_47/moving_mean*
validate_shape(*
_output_shapes	
:

save_1/Assign_172Assign3resnet_model/batch_normalization_47/moving_variancesave_1/RestoreV2_1:171"/device:CPU:0*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_47/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save_1/Assign_173Assign(resnet_model/batch_normalization_48/betasave_1/RestoreV2_1:172"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_48/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ú
save_1/Assign_174Assign)resnet_model/batch_normalization_48/gammasave_1/RestoreV2_1:173"/device:CPU:0*<
_class2
0.loc:@resnet_model/batch_normalization_48/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_175Assign/resnet_model/batch_normalization_48/moving_meansave_1/RestoreV2_1:174"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@resnet_model/batch_normalization_48/moving_mean*
validate_shape(

save_1/Assign_176Assign3resnet_model/batch_normalization_48/moving_variancesave_1/RestoreV2_1:175"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@resnet_model/batch_normalization_48/moving_variance
õ
save_1/Assign_177Assign'resnet_model/batch_normalization_5/betasave_1/RestoreV2_1:176"/device:CPU:0*:
_class0
.,loc:@resnet_model/batch_normalization_5/beta*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
÷
save_1/Assign_178Assign(resnet_model/batch_normalization_5/gammasave_1/RestoreV2_1:177"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_5/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

save_1/Assign_179Assign.resnet_model/batch_normalization_5/moving_meansave_1/RestoreV2_1:178"/device:CPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_5/moving_mean

save_1/Assign_180Assign2resnet_model/batch_normalization_5/moving_variancesave_1/RestoreV2_1:179"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_5/moving_variance*
validate_shape(*
_output_shapes
:@
ö
save_1/Assign_181Assign'resnet_model/batch_normalization_6/betasave_1/RestoreV2_1:180"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_6/beta*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_182Assign(resnet_model/batch_normalization_6/gammasave_1/RestoreV2_1:181"/device:CPU:0*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_6/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_183Assign.resnet_model/batch_normalization_6/moving_meansave_1/RestoreV2_1:182"/device:CPU:0*A
_class7
53loc:@resnet_model/batch_normalization_6/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_184Assign2resnet_model/batch_normalization_6/moving_variancesave_1/RestoreV2_1:183"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_6/moving_variance*
validate_shape(*
_output_shapes	
:
õ
save_1/Assign_185Assign'resnet_model/batch_normalization_7/betasave_1/RestoreV2_1:184"/device:CPU:0*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_7/beta*
validate_shape(*
_output_shapes
:@
÷
save_1/Assign_186Assign(resnet_model/batch_normalization_7/gammasave_1/RestoreV2_1:185"/device:CPU:0*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_7/gamma

save_1/Assign_187Assign.resnet_model/batch_normalization_7/moving_meansave_1/RestoreV2_1:186"/device:CPU:0*A
_class7
53loc:@resnet_model/batch_normalization_7/moving_mean*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

save_1/Assign_188Assign2resnet_model/batch_normalization_7/moving_variancesave_1/RestoreV2_1:187"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_7/moving_variance*
validate_shape(*
_output_shapes
:@
õ
save_1/Assign_189Assign'resnet_model/batch_normalization_8/betasave_1/RestoreV2_1:188"/device:CPU:0*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_8/beta*
validate_shape(*
_output_shapes
:@*
use_locking(
÷
save_1/Assign_190Assign(resnet_model/batch_normalization_8/gammasave_1/RestoreV2_1:189"/device:CPU:0*;
_class1
/-loc:@resnet_model/batch_normalization_8/gamma*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

save_1/Assign_191Assign.resnet_model/batch_normalization_8/moving_meansave_1/RestoreV2_1:190"/device:CPU:0*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_8/moving_mean*
validate_shape(*
_output_shapes
:@

save_1/Assign_192Assign2resnet_model/batch_normalization_8/moving_variancesave_1/RestoreV2_1:191"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_8/moving_variance*
validate_shape(*
_output_shapes
:@
ö
save_1/Assign_193Assign'resnet_model/batch_normalization_9/betasave_1/RestoreV2_1:192"/device:CPU:0*
_output_shapes	
:*
use_locking(*
T0*:
_class0
.,loc:@resnet_model/batch_normalization_9/beta*
validate_shape(
ø
save_1/Assign_194Assign(resnet_model/batch_normalization_9/gammasave_1/RestoreV2_1:193"/device:CPU:0*
use_locking(*
T0*;
_class1
/-loc:@resnet_model/batch_normalization_9/gamma*
validate_shape(*
_output_shapes	
:

save_1/Assign_195Assign.resnet_model/batch_normalization_9/moving_meansave_1/RestoreV2_1:194"/device:CPU:0*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@resnet_model/batch_normalization_9/moving_mean

save_1/Assign_196Assign2resnet_model/batch_normalization_9/moving_variancesave_1/RestoreV2_1:195"/device:CPU:0*
use_locking(*
T0*E
_class;
97loc:@resnet_model/batch_normalization_9/moving_variance*
validate_shape(*
_output_shapes	
:
ç
save_1/Assign_197Assignresnet_model/conv2d/kernelsave_1/RestoreV2_1:196"/device:CPU:0*-
_class#
!loc:@resnet_model/conv2d/kernel*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0
ì
save_1/Assign_198Assignresnet_model/conv2d_1/kernelsave_1/RestoreV2_1:197"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_1/kernel*
validate_shape(*'
_output_shapes
:@
î
save_1/Assign_199Assignresnet_model/conv2d_10/kernelsave_1/RestoreV2_1:198"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_10/kernel*
validate_shape(*'
_output_shapes
:@
ï
save_1/Assign_200Assignresnet_model/conv2d_11/kernelsave_1/RestoreV2_1:199"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_11/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_201Assignresnet_model/conv2d_12/kernelsave_1/RestoreV2_1:200"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_12/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_202Assignresnet_model/conv2d_13/kernelsave_1/RestoreV2_1:201"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_13/kernel
ï
save_1/Assign_203Assignresnet_model/conv2d_14/kernelsave_1/RestoreV2_1:202"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_14/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_204Assignresnet_model/conv2d_15/kernelsave_1/RestoreV2_1:203"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_15/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_205Assignresnet_model/conv2d_16/kernelsave_1/RestoreV2_1:204"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_16/kernel*
validate_shape(
ï
save_1/Assign_206Assignresnet_model/conv2d_17/kernelsave_1/RestoreV2_1:205"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_17/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_207Assignresnet_model/conv2d_18/kernelsave_1/RestoreV2_1:206"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_18/kernel
ï
save_1/Assign_208Assignresnet_model/conv2d_19/kernelsave_1/RestoreV2_1:207"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_19/kernel
ë
save_1/Assign_209Assignresnet_model/conv2d_2/kernelsave_1/RestoreV2_1:208"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_2/kernel*
validate_shape(*&
_output_shapes
:@@
ï
save_1/Assign_210Assignresnet_model/conv2d_20/kernelsave_1/RestoreV2_1:209"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_20/kernel*
validate_shape(
ï
save_1/Assign_211Assignresnet_model/conv2d_21/kernelsave_1/RestoreV2_1:210"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_21/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ï
save_1/Assign_212Assignresnet_model/conv2d_22/kernelsave_1/RestoreV2_1:211"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_22/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_213Assignresnet_model/conv2d_23/kernelsave_1/RestoreV2_1:212"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_23/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_214Assignresnet_model/conv2d_24/kernelsave_1/RestoreV2_1:213"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_24/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_215Assignresnet_model/conv2d_25/kernelsave_1/RestoreV2_1:214"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_25/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_216Assignresnet_model/conv2d_26/kernelsave_1/RestoreV2_1:215"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_26/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_217Assignresnet_model/conv2d_27/kernelsave_1/RestoreV2_1:216"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_27/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_218Assignresnet_model/conv2d_28/kernelsave_1/RestoreV2_1:217"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_28/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ï
save_1/Assign_219Assignresnet_model/conv2d_29/kernelsave_1/RestoreV2_1:218"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_29/kernel*
validate_shape(*(
_output_shapes
:
ë
save_1/Assign_220Assignresnet_model/conv2d_3/kernelsave_1/RestoreV2_1:219"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_3/kernel*
validate_shape(*&
_output_shapes
:@@*
use_locking(
ï
save_1/Assign_221Assignresnet_model/conv2d_30/kernelsave_1/RestoreV2_1:220"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_30/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_222Assignresnet_model/conv2d_31/kernelsave_1/RestoreV2_1:221"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_31/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ï
save_1/Assign_223Assignresnet_model/conv2d_32/kernelsave_1/RestoreV2_1:222"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_32/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ï
save_1/Assign_224Assignresnet_model/conv2d_33/kernelsave_1/RestoreV2_1:223"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_33/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_225Assignresnet_model/conv2d_34/kernelsave_1/RestoreV2_1:224"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_34/kernel*
validate_shape(
ï
save_1/Assign_226Assignresnet_model/conv2d_35/kernelsave_1/RestoreV2_1:225"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_35/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ï
save_1/Assign_227Assignresnet_model/conv2d_36/kernelsave_1/RestoreV2_1:226"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_36/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_228Assignresnet_model/conv2d_37/kernelsave_1/RestoreV2_1:227"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_37/kernel
ï
save_1/Assign_229Assignresnet_model/conv2d_38/kernelsave_1/RestoreV2_1:228"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_38/kernel
ï
save_1/Assign_230Assignresnet_model/conv2d_39/kernelsave_1/RestoreV2_1:229"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_39/kernel*
validate_shape(*(
_output_shapes
:
ì
save_1/Assign_231Assignresnet_model/conv2d_4/kernelsave_1/RestoreV2_1:230"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_4/kernel*
validate_shape(*'
_output_shapes
:@
ï
save_1/Assign_232Assignresnet_model/conv2d_40/kernelsave_1/RestoreV2_1:231"/device:CPU:0*
T0*0
_class&
$"loc:@resnet_model/conv2d_40/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(
ï
save_1/Assign_233Assignresnet_model/conv2d_41/kernelsave_1/RestoreV2_1:232"/device:CPU:0*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_41/kernel
ï
save_1/Assign_234Assignresnet_model/conv2d_42/kernelsave_1/RestoreV2_1:233"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_42/kernel*
validate_shape(
ï
save_1/Assign_235Assignresnet_model/conv2d_43/kernelsave_1/RestoreV2_1:234"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_43/kernel*
validate_shape(
ï
save_1/Assign_236Assignresnet_model/conv2d_44/kernelsave_1/RestoreV2_1:235"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_44/kernel*
validate_shape(
ï
save_1/Assign_237Assignresnet_model/conv2d_45/kernelsave_1/RestoreV2_1:236"/device:CPU:0*(
_output_shapes
:*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_45/kernel*
validate_shape(
ï
save_1/Assign_238Assignresnet_model/conv2d_46/kernelsave_1/RestoreV2_1:237"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_46/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ï
save_1/Assign_239Assignresnet_model/conv2d_47/kernelsave_1/RestoreV2_1:238"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_47/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_240Assignresnet_model/conv2d_48/kernelsave_1/RestoreV2_1:239"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_48/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_241Assignresnet_model/conv2d_49/kernelsave_1/RestoreV2_1:240"/device:CPU:0*0
_class&
$"loc:@resnet_model/conv2d_49/kernel*
validate_shape(*(
_output_shapes
:*
use_locking(*
T0
ì
save_1/Assign_242Assignresnet_model/conv2d_5/kernelsave_1/RestoreV2_1:241"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_5/kernel*
validate_shape(*'
_output_shapes
:@
ï
save_1/Assign_243Assignresnet_model/conv2d_50/kernelsave_1/RestoreV2_1:242"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_50/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_244Assignresnet_model/conv2d_51/kernelsave_1/RestoreV2_1:243"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_51/kernel*
validate_shape(*(
_output_shapes
:
ï
save_1/Assign_245Assignresnet_model/conv2d_52/kernelsave_1/RestoreV2_1:244"/device:CPU:0*
use_locking(*
T0*0
_class&
$"loc:@resnet_model/conv2d_52/kernel*
validate_shape(*(
_output_shapes
:
ë
save_1/Assign_246Assignresnet_model/conv2d_6/kernelsave_1/RestoreV2_1:245"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_6/kernel*
validate_shape(*&
_output_shapes
:@@
ì
save_1/Assign_247Assignresnet_model/conv2d_7/kernelsave_1/RestoreV2_1:246"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_7/kernel*
validate_shape(*'
_output_shapes
:@
ì
save_1/Assign_248Assignresnet_model/conv2d_8/kernelsave_1/RestoreV2_1:247"/device:CPU:0*
T0*/
_class%
#!loc:@resnet_model/conv2d_8/kernel*
validate_shape(*'
_output_shapes
:@*
use_locking(
ë
save_1/Assign_249Assignresnet_model/conv2d_9/kernelsave_1/RestoreV2_1:248"/device:CPU:0*
use_locking(*
T0*/
_class%
#!loc:@resnet_model/conv2d_9/kernel*
validate_shape(*&
_output_shapes
:@@
Ö
save_1/Assign_250Assignresnet_model/dense/biassave_1/RestoreV2_1:249"/device:CPU:0*
T0**
_class 
loc:@resnet_model/dense/bias*
validate_shape(*
_output_shapes	
:é*
use_locking(
ß
save_1/Assign_251Assignresnet_model/dense/kernelsave_1/RestoreV2_1:250"/device:CPU:0*
use_locking(*
T0*,
_class"
 loc:@resnet_model/dense/kernel*
validate_shape(* 
_output_shapes
:
é
Ý&
save_1/restore_shard_1NoOp^save_1/Assign_1^save_1/Assign_10^save_1/Assign_100^save_1/Assign_101^save_1/Assign_102^save_1/Assign_103^save_1/Assign_104^save_1/Assign_105^save_1/Assign_106^save_1/Assign_107^save_1/Assign_108^save_1/Assign_109^save_1/Assign_11^save_1/Assign_110^save_1/Assign_111^save_1/Assign_112^save_1/Assign_113^save_1/Assign_114^save_1/Assign_115^save_1/Assign_116^save_1/Assign_117^save_1/Assign_118^save_1/Assign_119^save_1/Assign_12^save_1/Assign_120^save_1/Assign_121^save_1/Assign_122^save_1/Assign_123^save_1/Assign_124^save_1/Assign_125^save_1/Assign_126^save_1/Assign_127^save_1/Assign_128^save_1/Assign_129^save_1/Assign_13^save_1/Assign_130^save_1/Assign_131^save_1/Assign_132^save_1/Assign_133^save_1/Assign_134^save_1/Assign_135^save_1/Assign_136^save_1/Assign_137^save_1/Assign_138^save_1/Assign_139^save_1/Assign_14^save_1/Assign_140^save_1/Assign_141^save_1/Assign_142^save_1/Assign_143^save_1/Assign_144^save_1/Assign_145^save_1/Assign_146^save_1/Assign_147^save_1/Assign_148^save_1/Assign_149^save_1/Assign_15^save_1/Assign_150^save_1/Assign_151^save_1/Assign_152^save_1/Assign_153^save_1/Assign_154^save_1/Assign_155^save_1/Assign_156^save_1/Assign_157^save_1/Assign_158^save_1/Assign_159^save_1/Assign_16^save_1/Assign_160^save_1/Assign_161^save_1/Assign_162^save_1/Assign_163^save_1/Assign_164^save_1/Assign_165^save_1/Assign_166^save_1/Assign_167^save_1/Assign_168^save_1/Assign_169^save_1/Assign_17^save_1/Assign_170^save_1/Assign_171^save_1/Assign_172^save_1/Assign_173^save_1/Assign_174^save_1/Assign_175^save_1/Assign_176^save_1/Assign_177^save_1/Assign_178^save_1/Assign_179^save_1/Assign_18^save_1/Assign_180^save_1/Assign_181^save_1/Assign_182^save_1/Assign_183^save_1/Assign_184^save_1/Assign_185^save_1/Assign_186^save_1/Assign_187^save_1/Assign_188^save_1/Assign_189^save_1/Assign_19^save_1/Assign_190^save_1/Assign_191^save_1/Assign_192^save_1/Assign_193^save_1/Assign_194^save_1/Assign_195^save_1/Assign_196^save_1/Assign_197^save_1/Assign_198^save_1/Assign_199^save_1/Assign_2^save_1/Assign_20^save_1/Assign_200^save_1/Assign_201^save_1/Assign_202^save_1/Assign_203^save_1/Assign_204^save_1/Assign_205^save_1/Assign_206^save_1/Assign_207^save_1/Assign_208^save_1/Assign_209^save_1/Assign_21^save_1/Assign_210^save_1/Assign_211^save_1/Assign_212^save_1/Assign_213^save_1/Assign_214^save_1/Assign_215^save_1/Assign_216^save_1/Assign_217^save_1/Assign_218^save_1/Assign_219^save_1/Assign_22^save_1/Assign_220^save_1/Assign_221^save_1/Assign_222^save_1/Assign_223^save_1/Assign_224^save_1/Assign_225^save_1/Assign_226^save_1/Assign_227^save_1/Assign_228^save_1/Assign_229^save_1/Assign_23^save_1/Assign_230^save_1/Assign_231^save_1/Assign_232^save_1/Assign_233^save_1/Assign_234^save_1/Assign_235^save_1/Assign_236^save_1/Assign_237^save_1/Assign_238^save_1/Assign_239^save_1/Assign_24^save_1/Assign_240^save_1/Assign_241^save_1/Assign_242^save_1/Assign_243^save_1/Assign_244^save_1/Assign_245^save_1/Assign_246^save_1/Assign_247^save_1/Assign_248^save_1/Assign_249^save_1/Assign_25^save_1/Assign_250^save_1/Assign_251^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79^save_1/Assign_8^save_1/Assign_80^save_1/Assign_81^save_1/Assign_82^save_1/Assign_83^save_1/Assign_84^save_1/Assign_85^save_1/Assign_86^save_1/Assign_87^save_1/Assign_88^save_1/Assign_89^save_1/Assign_9^save_1/Assign_90^save_1/Assign_91^save_1/Assign_92^save_1/Assign_93^save_1/Assign_94^save_1/Assign_95^save_1/Assign_96^save_1/Assign_97^save_1/Assign_98^save_1/Assign_99"/device:CPU:0
6
save_1/restore_all/NoOpNoOp^save_1/restore_shard
I
save_1/restore_all/NoOp_1NoOp^save_1/restore_shard_1"/device:CPU:0
P
save_1/restore_allNoOp^save_1/restore_all/NoOp^save_1/restore_all/NoOp_1"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"+
	summaries

images:0
tower_1/images:0"é
trainable_variablesòèîè

resnet_model/conv2d/kernel:0!resnet_model/conv2d/kernel/Assign!resnet_model/conv2d/kernel/read:029resnet_model/conv2d/kernel/Initializer/truncated_normal:0
Ã
(resnet_model/batch_normalization/gamma:0-resnet_model/batch_normalization/gamma/Assign-resnet_model/batch_normalization/gamma/read:029resnet_model/batch_normalization/gamma/Initializer/ones:0
À
'resnet_model/batch_normalization/beta:0,resnet_model/batch_normalization/beta/Assign,resnet_model/batch_normalization/beta/read:029resnet_model/batch_normalization/beta/Initializer/zeros:0
§
resnet_model/conv2d_1/kernel:0#resnet_model/conv2d_1/kernel/Assign#resnet_model/conv2d_1/kernel/read:02;resnet_model/conv2d_1/kernel/Initializer/truncated_normal:0
§
resnet_model/conv2d_2/kernel:0#resnet_model/conv2d_2/kernel/Assign#resnet_model/conv2d_2/kernel/read:02;resnet_model/conv2d_2/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_1/gamma:0/resnet_model/batch_normalization_1/gamma/Assign/resnet_model/batch_normalization_1/gamma/read:02;resnet_model/batch_normalization_1/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_1/beta:0.resnet_model/batch_normalization_1/beta/Assign.resnet_model/batch_normalization_1/beta/read:02;resnet_model/batch_normalization_1/beta/Initializer/zeros:0
§
resnet_model/conv2d_3/kernel:0#resnet_model/conv2d_3/kernel/Assign#resnet_model/conv2d_3/kernel/read:02;resnet_model/conv2d_3/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_2/gamma:0/resnet_model/batch_normalization_2/gamma/Assign/resnet_model/batch_normalization_2/gamma/read:02;resnet_model/batch_normalization_2/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_2/beta:0.resnet_model/batch_normalization_2/beta/Assign.resnet_model/batch_normalization_2/beta/read:02;resnet_model/batch_normalization_2/beta/Initializer/zeros:0
§
resnet_model/conv2d_4/kernel:0#resnet_model/conv2d_4/kernel/Assign#resnet_model/conv2d_4/kernel/read:02;resnet_model/conv2d_4/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_3/gamma:0/resnet_model/batch_normalization_3/gamma/Assign/resnet_model/batch_normalization_3/gamma/read:02;resnet_model/batch_normalization_3/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_3/beta:0.resnet_model/batch_normalization_3/beta/Assign.resnet_model/batch_normalization_3/beta/read:02;resnet_model/batch_normalization_3/beta/Initializer/zeros:0
§
resnet_model/conv2d_5/kernel:0#resnet_model/conv2d_5/kernel/Assign#resnet_model/conv2d_5/kernel/read:02;resnet_model/conv2d_5/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_4/gamma:0/resnet_model/batch_normalization_4/gamma/Assign/resnet_model/batch_normalization_4/gamma/read:02;resnet_model/batch_normalization_4/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_4/beta:0.resnet_model/batch_normalization_4/beta/Assign.resnet_model/batch_normalization_4/beta/read:02;resnet_model/batch_normalization_4/beta/Initializer/zeros:0
§
resnet_model/conv2d_6/kernel:0#resnet_model/conv2d_6/kernel/Assign#resnet_model/conv2d_6/kernel/read:02;resnet_model/conv2d_6/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_5/gamma:0/resnet_model/batch_normalization_5/gamma/Assign/resnet_model/batch_normalization_5/gamma/read:02;resnet_model/batch_normalization_5/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_5/beta:0.resnet_model/batch_normalization_5/beta/Assign.resnet_model/batch_normalization_5/beta/read:02;resnet_model/batch_normalization_5/beta/Initializer/zeros:0
§
resnet_model/conv2d_7/kernel:0#resnet_model/conv2d_7/kernel/Assign#resnet_model/conv2d_7/kernel/read:02;resnet_model/conv2d_7/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_6/gamma:0/resnet_model/batch_normalization_6/gamma/Assign/resnet_model/batch_normalization_6/gamma/read:02;resnet_model/batch_normalization_6/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_6/beta:0.resnet_model/batch_normalization_6/beta/Assign.resnet_model/batch_normalization_6/beta/read:02;resnet_model/batch_normalization_6/beta/Initializer/zeros:0
§
resnet_model/conv2d_8/kernel:0#resnet_model/conv2d_8/kernel/Assign#resnet_model/conv2d_8/kernel/read:02;resnet_model/conv2d_8/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_7/gamma:0/resnet_model/batch_normalization_7/gamma/Assign/resnet_model/batch_normalization_7/gamma/read:02;resnet_model/batch_normalization_7/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_7/beta:0.resnet_model/batch_normalization_7/beta/Assign.resnet_model/batch_normalization_7/beta/read:02;resnet_model/batch_normalization_7/beta/Initializer/zeros:0
§
resnet_model/conv2d_9/kernel:0#resnet_model/conv2d_9/kernel/Assign#resnet_model/conv2d_9/kernel/read:02;resnet_model/conv2d_9/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_8/gamma:0/resnet_model/batch_normalization_8/gamma/Assign/resnet_model/batch_normalization_8/gamma/read:02;resnet_model/batch_normalization_8/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_8/beta:0.resnet_model/batch_normalization_8/beta/Assign.resnet_model/batch_normalization_8/beta/read:02;resnet_model/batch_normalization_8/beta/Initializer/zeros:0
«
resnet_model/conv2d_10/kernel:0$resnet_model/conv2d_10/kernel/Assign$resnet_model/conv2d_10/kernel/read:02<resnet_model/conv2d_10/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_9/gamma:0/resnet_model/batch_normalization_9/gamma/Assign/resnet_model/batch_normalization_9/gamma/read:02;resnet_model/batch_normalization_9/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_9/beta:0.resnet_model/batch_normalization_9/beta/Assign.resnet_model/batch_normalization_9/beta/read:02;resnet_model/batch_normalization_9/beta/Initializer/zeros:0
«
resnet_model/conv2d_11/kernel:0$resnet_model/conv2d_11/kernel/Assign$resnet_model/conv2d_11/kernel/read:02<resnet_model/conv2d_11/kernel/Initializer/truncated_normal:0
«
resnet_model/conv2d_12/kernel:0$resnet_model/conv2d_12/kernel/Assign$resnet_model/conv2d_12/kernel/read:02<resnet_model/conv2d_12/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_10/gamma:00resnet_model/batch_normalization_10/gamma/Assign0resnet_model/batch_normalization_10/gamma/read:02<resnet_model/batch_normalization_10/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_10/beta:0/resnet_model/batch_normalization_10/beta/Assign/resnet_model/batch_normalization_10/beta/read:02<resnet_model/batch_normalization_10/beta/Initializer/zeros:0
«
resnet_model/conv2d_13/kernel:0$resnet_model/conv2d_13/kernel/Assign$resnet_model/conv2d_13/kernel/read:02<resnet_model/conv2d_13/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_11/gamma:00resnet_model/batch_normalization_11/gamma/Assign0resnet_model/batch_normalization_11/gamma/read:02<resnet_model/batch_normalization_11/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_11/beta:0/resnet_model/batch_normalization_11/beta/Assign/resnet_model/batch_normalization_11/beta/read:02<resnet_model/batch_normalization_11/beta/Initializer/zeros:0
«
resnet_model/conv2d_14/kernel:0$resnet_model/conv2d_14/kernel/Assign$resnet_model/conv2d_14/kernel/read:02<resnet_model/conv2d_14/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_12/gamma:00resnet_model/batch_normalization_12/gamma/Assign0resnet_model/batch_normalization_12/gamma/read:02<resnet_model/batch_normalization_12/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_12/beta:0/resnet_model/batch_normalization_12/beta/Assign/resnet_model/batch_normalization_12/beta/read:02<resnet_model/batch_normalization_12/beta/Initializer/zeros:0
«
resnet_model/conv2d_15/kernel:0$resnet_model/conv2d_15/kernel/Assign$resnet_model/conv2d_15/kernel/read:02<resnet_model/conv2d_15/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_13/gamma:00resnet_model/batch_normalization_13/gamma/Assign0resnet_model/batch_normalization_13/gamma/read:02<resnet_model/batch_normalization_13/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_13/beta:0/resnet_model/batch_normalization_13/beta/Assign/resnet_model/batch_normalization_13/beta/read:02<resnet_model/batch_normalization_13/beta/Initializer/zeros:0
«
resnet_model/conv2d_16/kernel:0$resnet_model/conv2d_16/kernel/Assign$resnet_model/conv2d_16/kernel/read:02<resnet_model/conv2d_16/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_14/gamma:00resnet_model/batch_normalization_14/gamma/Assign0resnet_model/batch_normalization_14/gamma/read:02<resnet_model/batch_normalization_14/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_14/beta:0/resnet_model/batch_normalization_14/beta/Assign/resnet_model/batch_normalization_14/beta/read:02<resnet_model/batch_normalization_14/beta/Initializer/zeros:0
«
resnet_model/conv2d_17/kernel:0$resnet_model/conv2d_17/kernel/Assign$resnet_model/conv2d_17/kernel/read:02<resnet_model/conv2d_17/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_15/gamma:00resnet_model/batch_normalization_15/gamma/Assign0resnet_model/batch_normalization_15/gamma/read:02<resnet_model/batch_normalization_15/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_15/beta:0/resnet_model/batch_normalization_15/beta/Assign/resnet_model/batch_normalization_15/beta/read:02<resnet_model/batch_normalization_15/beta/Initializer/zeros:0
«
resnet_model/conv2d_18/kernel:0$resnet_model/conv2d_18/kernel/Assign$resnet_model/conv2d_18/kernel/read:02<resnet_model/conv2d_18/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_16/gamma:00resnet_model/batch_normalization_16/gamma/Assign0resnet_model/batch_normalization_16/gamma/read:02<resnet_model/batch_normalization_16/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_16/beta:0/resnet_model/batch_normalization_16/beta/Assign/resnet_model/batch_normalization_16/beta/read:02<resnet_model/batch_normalization_16/beta/Initializer/zeros:0
«
resnet_model/conv2d_19/kernel:0$resnet_model/conv2d_19/kernel/Assign$resnet_model/conv2d_19/kernel/read:02<resnet_model/conv2d_19/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_17/gamma:00resnet_model/batch_normalization_17/gamma/Assign0resnet_model/batch_normalization_17/gamma/read:02<resnet_model/batch_normalization_17/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_17/beta:0/resnet_model/batch_normalization_17/beta/Assign/resnet_model/batch_normalization_17/beta/read:02<resnet_model/batch_normalization_17/beta/Initializer/zeros:0
«
resnet_model/conv2d_20/kernel:0$resnet_model/conv2d_20/kernel/Assign$resnet_model/conv2d_20/kernel/read:02<resnet_model/conv2d_20/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_18/gamma:00resnet_model/batch_normalization_18/gamma/Assign0resnet_model/batch_normalization_18/gamma/read:02<resnet_model/batch_normalization_18/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_18/beta:0/resnet_model/batch_normalization_18/beta/Assign/resnet_model/batch_normalization_18/beta/read:02<resnet_model/batch_normalization_18/beta/Initializer/zeros:0
«
resnet_model/conv2d_21/kernel:0$resnet_model/conv2d_21/kernel/Assign$resnet_model/conv2d_21/kernel/read:02<resnet_model/conv2d_21/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_19/gamma:00resnet_model/batch_normalization_19/gamma/Assign0resnet_model/batch_normalization_19/gamma/read:02<resnet_model/batch_normalization_19/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_19/beta:0/resnet_model/batch_normalization_19/beta/Assign/resnet_model/batch_normalization_19/beta/read:02<resnet_model/batch_normalization_19/beta/Initializer/zeros:0
«
resnet_model/conv2d_22/kernel:0$resnet_model/conv2d_22/kernel/Assign$resnet_model/conv2d_22/kernel/read:02<resnet_model/conv2d_22/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_20/gamma:00resnet_model/batch_normalization_20/gamma/Assign0resnet_model/batch_normalization_20/gamma/read:02<resnet_model/batch_normalization_20/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_20/beta:0/resnet_model/batch_normalization_20/beta/Assign/resnet_model/batch_normalization_20/beta/read:02<resnet_model/batch_normalization_20/beta/Initializer/zeros:0
«
resnet_model/conv2d_23/kernel:0$resnet_model/conv2d_23/kernel/Assign$resnet_model/conv2d_23/kernel/read:02<resnet_model/conv2d_23/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_21/gamma:00resnet_model/batch_normalization_21/gamma/Assign0resnet_model/batch_normalization_21/gamma/read:02<resnet_model/batch_normalization_21/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_21/beta:0/resnet_model/batch_normalization_21/beta/Assign/resnet_model/batch_normalization_21/beta/read:02<resnet_model/batch_normalization_21/beta/Initializer/zeros:0
«
resnet_model/conv2d_24/kernel:0$resnet_model/conv2d_24/kernel/Assign$resnet_model/conv2d_24/kernel/read:02<resnet_model/conv2d_24/kernel/Initializer/truncated_normal:0
«
resnet_model/conv2d_25/kernel:0$resnet_model/conv2d_25/kernel/Assign$resnet_model/conv2d_25/kernel/read:02<resnet_model/conv2d_25/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_22/gamma:00resnet_model/batch_normalization_22/gamma/Assign0resnet_model/batch_normalization_22/gamma/read:02<resnet_model/batch_normalization_22/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_22/beta:0/resnet_model/batch_normalization_22/beta/Assign/resnet_model/batch_normalization_22/beta/read:02<resnet_model/batch_normalization_22/beta/Initializer/zeros:0
«
resnet_model/conv2d_26/kernel:0$resnet_model/conv2d_26/kernel/Assign$resnet_model/conv2d_26/kernel/read:02<resnet_model/conv2d_26/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_23/gamma:00resnet_model/batch_normalization_23/gamma/Assign0resnet_model/batch_normalization_23/gamma/read:02<resnet_model/batch_normalization_23/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_23/beta:0/resnet_model/batch_normalization_23/beta/Assign/resnet_model/batch_normalization_23/beta/read:02<resnet_model/batch_normalization_23/beta/Initializer/zeros:0
«
resnet_model/conv2d_27/kernel:0$resnet_model/conv2d_27/kernel/Assign$resnet_model/conv2d_27/kernel/read:02<resnet_model/conv2d_27/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_24/gamma:00resnet_model/batch_normalization_24/gamma/Assign0resnet_model/batch_normalization_24/gamma/read:02<resnet_model/batch_normalization_24/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_24/beta:0/resnet_model/batch_normalization_24/beta/Assign/resnet_model/batch_normalization_24/beta/read:02<resnet_model/batch_normalization_24/beta/Initializer/zeros:0
«
resnet_model/conv2d_28/kernel:0$resnet_model/conv2d_28/kernel/Assign$resnet_model/conv2d_28/kernel/read:02<resnet_model/conv2d_28/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_25/gamma:00resnet_model/batch_normalization_25/gamma/Assign0resnet_model/batch_normalization_25/gamma/read:02<resnet_model/batch_normalization_25/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_25/beta:0/resnet_model/batch_normalization_25/beta/Assign/resnet_model/batch_normalization_25/beta/read:02<resnet_model/batch_normalization_25/beta/Initializer/zeros:0
«
resnet_model/conv2d_29/kernel:0$resnet_model/conv2d_29/kernel/Assign$resnet_model/conv2d_29/kernel/read:02<resnet_model/conv2d_29/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_26/gamma:00resnet_model/batch_normalization_26/gamma/Assign0resnet_model/batch_normalization_26/gamma/read:02<resnet_model/batch_normalization_26/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_26/beta:0/resnet_model/batch_normalization_26/beta/Assign/resnet_model/batch_normalization_26/beta/read:02<resnet_model/batch_normalization_26/beta/Initializer/zeros:0
«
resnet_model/conv2d_30/kernel:0$resnet_model/conv2d_30/kernel/Assign$resnet_model/conv2d_30/kernel/read:02<resnet_model/conv2d_30/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_27/gamma:00resnet_model/batch_normalization_27/gamma/Assign0resnet_model/batch_normalization_27/gamma/read:02<resnet_model/batch_normalization_27/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_27/beta:0/resnet_model/batch_normalization_27/beta/Assign/resnet_model/batch_normalization_27/beta/read:02<resnet_model/batch_normalization_27/beta/Initializer/zeros:0
«
resnet_model/conv2d_31/kernel:0$resnet_model/conv2d_31/kernel/Assign$resnet_model/conv2d_31/kernel/read:02<resnet_model/conv2d_31/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_28/gamma:00resnet_model/batch_normalization_28/gamma/Assign0resnet_model/batch_normalization_28/gamma/read:02<resnet_model/batch_normalization_28/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_28/beta:0/resnet_model/batch_normalization_28/beta/Assign/resnet_model/batch_normalization_28/beta/read:02<resnet_model/batch_normalization_28/beta/Initializer/zeros:0
«
resnet_model/conv2d_32/kernel:0$resnet_model/conv2d_32/kernel/Assign$resnet_model/conv2d_32/kernel/read:02<resnet_model/conv2d_32/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_29/gamma:00resnet_model/batch_normalization_29/gamma/Assign0resnet_model/batch_normalization_29/gamma/read:02<resnet_model/batch_normalization_29/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_29/beta:0/resnet_model/batch_normalization_29/beta/Assign/resnet_model/batch_normalization_29/beta/read:02<resnet_model/batch_normalization_29/beta/Initializer/zeros:0
«
resnet_model/conv2d_33/kernel:0$resnet_model/conv2d_33/kernel/Assign$resnet_model/conv2d_33/kernel/read:02<resnet_model/conv2d_33/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_30/gamma:00resnet_model/batch_normalization_30/gamma/Assign0resnet_model/batch_normalization_30/gamma/read:02<resnet_model/batch_normalization_30/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_30/beta:0/resnet_model/batch_normalization_30/beta/Assign/resnet_model/batch_normalization_30/beta/read:02<resnet_model/batch_normalization_30/beta/Initializer/zeros:0
«
resnet_model/conv2d_34/kernel:0$resnet_model/conv2d_34/kernel/Assign$resnet_model/conv2d_34/kernel/read:02<resnet_model/conv2d_34/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_31/gamma:00resnet_model/batch_normalization_31/gamma/Assign0resnet_model/batch_normalization_31/gamma/read:02<resnet_model/batch_normalization_31/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_31/beta:0/resnet_model/batch_normalization_31/beta/Assign/resnet_model/batch_normalization_31/beta/read:02<resnet_model/batch_normalization_31/beta/Initializer/zeros:0
«
resnet_model/conv2d_35/kernel:0$resnet_model/conv2d_35/kernel/Assign$resnet_model/conv2d_35/kernel/read:02<resnet_model/conv2d_35/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_32/gamma:00resnet_model/batch_normalization_32/gamma/Assign0resnet_model/batch_normalization_32/gamma/read:02<resnet_model/batch_normalization_32/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_32/beta:0/resnet_model/batch_normalization_32/beta/Assign/resnet_model/batch_normalization_32/beta/read:02<resnet_model/batch_normalization_32/beta/Initializer/zeros:0
«
resnet_model/conv2d_36/kernel:0$resnet_model/conv2d_36/kernel/Assign$resnet_model/conv2d_36/kernel/read:02<resnet_model/conv2d_36/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_33/gamma:00resnet_model/batch_normalization_33/gamma/Assign0resnet_model/batch_normalization_33/gamma/read:02<resnet_model/batch_normalization_33/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_33/beta:0/resnet_model/batch_normalization_33/beta/Assign/resnet_model/batch_normalization_33/beta/read:02<resnet_model/batch_normalization_33/beta/Initializer/zeros:0
«
resnet_model/conv2d_37/kernel:0$resnet_model/conv2d_37/kernel/Assign$resnet_model/conv2d_37/kernel/read:02<resnet_model/conv2d_37/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_34/gamma:00resnet_model/batch_normalization_34/gamma/Assign0resnet_model/batch_normalization_34/gamma/read:02<resnet_model/batch_normalization_34/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_34/beta:0/resnet_model/batch_normalization_34/beta/Assign/resnet_model/batch_normalization_34/beta/read:02<resnet_model/batch_normalization_34/beta/Initializer/zeros:0
«
resnet_model/conv2d_38/kernel:0$resnet_model/conv2d_38/kernel/Assign$resnet_model/conv2d_38/kernel/read:02<resnet_model/conv2d_38/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_35/gamma:00resnet_model/batch_normalization_35/gamma/Assign0resnet_model/batch_normalization_35/gamma/read:02<resnet_model/batch_normalization_35/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_35/beta:0/resnet_model/batch_normalization_35/beta/Assign/resnet_model/batch_normalization_35/beta/read:02<resnet_model/batch_normalization_35/beta/Initializer/zeros:0
«
resnet_model/conv2d_39/kernel:0$resnet_model/conv2d_39/kernel/Assign$resnet_model/conv2d_39/kernel/read:02<resnet_model/conv2d_39/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_36/gamma:00resnet_model/batch_normalization_36/gamma/Assign0resnet_model/batch_normalization_36/gamma/read:02<resnet_model/batch_normalization_36/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_36/beta:0/resnet_model/batch_normalization_36/beta/Assign/resnet_model/batch_normalization_36/beta/read:02<resnet_model/batch_normalization_36/beta/Initializer/zeros:0
«
resnet_model/conv2d_40/kernel:0$resnet_model/conv2d_40/kernel/Assign$resnet_model/conv2d_40/kernel/read:02<resnet_model/conv2d_40/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_37/gamma:00resnet_model/batch_normalization_37/gamma/Assign0resnet_model/batch_normalization_37/gamma/read:02<resnet_model/batch_normalization_37/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_37/beta:0/resnet_model/batch_normalization_37/beta/Assign/resnet_model/batch_normalization_37/beta/read:02<resnet_model/batch_normalization_37/beta/Initializer/zeros:0
«
resnet_model/conv2d_41/kernel:0$resnet_model/conv2d_41/kernel/Assign$resnet_model/conv2d_41/kernel/read:02<resnet_model/conv2d_41/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_38/gamma:00resnet_model/batch_normalization_38/gamma/Assign0resnet_model/batch_normalization_38/gamma/read:02<resnet_model/batch_normalization_38/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_38/beta:0/resnet_model/batch_normalization_38/beta/Assign/resnet_model/batch_normalization_38/beta/read:02<resnet_model/batch_normalization_38/beta/Initializer/zeros:0
«
resnet_model/conv2d_42/kernel:0$resnet_model/conv2d_42/kernel/Assign$resnet_model/conv2d_42/kernel/read:02<resnet_model/conv2d_42/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_39/gamma:00resnet_model/batch_normalization_39/gamma/Assign0resnet_model/batch_normalization_39/gamma/read:02<resnet_model/batch_normalization_39/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_39/beta:0/resnet_model/batch_normalization_39/beta/Assign/resnet_model/batch_normalization_39/beta/read:02<resnet_model/batch_normalization_39/beta/Initializer/zeros:0
«
resnet_model/conv2d_43/kernel:0$resnet_model/conv2d_43/kernel/Assign$resnet_model/conv2d_43/kernel/read:02<resnet_model/conv2d_43/kernel/Initializer/truncated_normal:0
«
resnet_model/conv2d_44/kernel:0$resnet_model/conv2d_44/kernel/Assign$resnet_model/conv2d_44/kernel/read:02<resnet_model/conv2d_44/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_40/gamma:00resnet_model/batch_normalization_40/gamma/Assign0resnet_model/batch_normalization_40/gamma/read:02<resnet_model/batch_normalization_40/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_40/beta:0/resnet_model/batch_normalization_40/beta/Assign/resnet_model/batch_normalization_40/beta/read:02<resnet_model/batch_normalization_40/beta/Initializer/zeros:0
«
resnet_model/conv2d_45/kernel:0$resnet_model/conv2d_45/kernel/Assign$resnet_model/conv2d_45/kernel/read:02<resnet_model/conv2d_45/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_41/gamma:00resnet_model/batch_normalization_41/gamma/Assign0resnet_model/batch_normalization_41/gamma/read:02<resnet_model/batch_normalization_41/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_41/beta:0/resnet_model/batch_normalization_41/beta/Assign/resnet_model/batch_normalization_41/beta/read:02<resnet_model/batch_normalization_41/beta/Initializer/zeros:0
«
resnet_model/conv2d_46/kernel:0$resnet_model/conv2d_46/kernel/Assign$resnet_model/conv2d_46/kernel/read:02<resnet_model/conv2d_46/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_42/gamma:00resnet_model/batch_normalization_42/gamma/Assign0resnet_model/batch_normalization_42/gamma/read:02<resnet_model/batch_normalization_42/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_42/beta:0/resnet_model/batch_normalization_42/beta/Assign/resnet_model/batch_normalization_42/beta/read:02<resnet_model/batch_normalization_42/beta/Initializer/zeros:0
«
resnet_model/conv2d_47/kernel:0$resnet_model/conv2d_47/kernel/Assign$resnet_model/conv2d_47/kernel/read:02<resnet_model/conv2d_47/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_43/gamma:00resnet_model/batch_normalization_43/gamma/Assign0resnet_model/batch_normalization_43/gamma/read:02<resnet_model/batch_normalization_43/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_43/beta:0/resnet_model/batch_normalization_43/beta/Assign/resnet_model/batch_normalization_43/beta/read:02<resnet_model/batch_normalization_43/beta/Initializer/zeros:0
«
resnet_model/conv2d_48/kernel:0$resnet_model/conv2d_48/kernel/Assign$resnet_model/conv2d_48/kernel/read:02<resnet_model/conv2d_48/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_44/gamma:00resnet_model/batch_normalization_44/gamma/Assign0resnet_model/batch_normalization_44/gamma/read:02<resnet_model/batch_normalization_44/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_44/beta:0/resnet_model/batch_normalization_44/beta/Assign/resnet_model/batch_normalization_44/beta/read:02<resnet_model/batch_normalization_44/beta/Initializer/zeros:0
«
resnet_model/conv2d_49/kernel:0$resnet_model/conv2d_49/kernel/Assign$resnet_model/conv2d_49/kernel/read:02<resnet_model/conv2d_49/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_45/gamma:00resnet_model/batch_normalization_45/gamma/Assign0resnet_model/batch_normalization_45/gamma/read:02<resnet_model/batch_normalization_45/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_45/beta:0/resnet_model/batch_normalization_45/beta/Assign/resnet_model/batch_normalization_45/beta/read:02<resnet_model/batch_normalization_45/beta/Initializer/zeros:0
«
resnet_model/conv2d_50/kernel:0$resnet_model/conv2d_50/kernel/Assign$resnet_model/conv2d_50/kernel/read:02<resnet_model/conv2d_50/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_46/gamma:00resnet_model/batch_normalization_46/gamma/Assign0resnet_model/batch_normalization_46/gamma/read:02<resnet_model/batch_normalization_46/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_46/beta:0/resnet_model/batch_normalization_46/beta/Assign/resnet_model/batch_normalization_46/beta/read:02<resnet_model/batch_normalization_46/beta/Initializer/zeros:0
«
resnet_model/conv2d_51/kernel:0$resnet_model/conv2d_51/kernel/Assign$resnet_model/conv2d_51/kernel/read:02<resnet_model/conv2d_51/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_47/gamma:00resnet_model/batch_normalization_47/gamma/Assign0resnet_model/batch_normalization_47/gamma/read:02<resnet_model/batch_normalization_47/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_47/beta:0/resnet_model/batch_normalization_47/beta/Assign/resnet_model/batch_normalization_47/beta/read:02<resnet_model/batch_normalization_47/beta/Initializer/zeros:0
«
resnet_model/conv2d_52/kernel:0$resnet_model/conv2d_52/kernel/Assign$resnet_model/conv2d_52/kernel/read:02<resnet_model/conv2d_52/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_48/gamma:00resnet_model/batch_normalization_48/gamma/Assign0resnet_model/batch_normalization_48/gamma/read:02<resnet_model/batch_normalization_48/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_48/beta:0/resnet_model/batch_normalization_48/beta/Assign/resnet_model/batch_normalization_48/beta/read:02<resnet_model/batch_normalization_48/beta/Initializer/zeros:0

resnet_model/dense/kernel:0 resnet_model/dense/kernel/Assign resnet_model/dense/kernel/read:026resnet_model/dense/kernel/Initializer/random_uniform:0

resnet_model/dense/bias:0resnet_model/dense/bias/Assignresnet_model/dense/bias/read:02+resnet_model/dense/bias/Initializer/zeros:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"Ð¢
	variablesÁ¢½¢
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0

resnet_model/conv2d/kernel:0!resnet_model/conv2d/kernel/Assign!resnet_model/conv2d/kernel/read:029resnet_model/conv2d/kernel/Initializer/truncated_normal:0
Ã
(resnet_model/batch_normalization/gamma:0-resnet_model/batch_normalization/gamma/Assign-resnet_model/batch_normalization/gamma/read:029resnet_model/batch_normalization/gamma/Initializer/ones:0
À
'resnet_model/batch_normalization/beta:0,resnet_model/batch_normalization/beta/Assign,resnet_model/batch_normalization/beta/read:029resnet_model/batch_normalization/beta/Initializer/zeros:0
Ü
.resnet_model/batch_normalization/moving_mean:03resnet_model/batch_normalization/moving_mean/Assign3resnet_model/batch_normalization/moving_mean/read:02@resnet_model/batch_normalization/moving_mean/Initializer/zeros:0
ë
2resnet_model/batch_normalization/moving_variance:07resnet_model/batch_normalization/moving_variance/Assign7resnet_model/batch_normalization/moving_variance/read:02Cresnet_model/batch_normalization/moving_variance/Initializer/ones:0
§
resnet_model/conv2d_1/kernel:0#resnet_model/conv2d_1/kernel/Assign#resnet_model/conv2d_1/kernel/read:02;resnet_model/conv2d_1/kernel/Initializer/truncated_normal:0
§
resnet_model/conv2d_2/kernel:0#resnet_model/conv2d_2/kernel/Assign#resnet_model/conv2d_2/kernel/read:02;resnet_model/conv2d_2/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_1/gamma:0/resnet_model/batch_normalization_1/gamma/Assign/resnet_model/batch_normalization_1/gamma/read:02;resnet_model/batch_normalization_1/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_1/beta:0.resnet_model/batch_normalization_1/beta/Assign.resnet_model/batch_normalization_1/beta/read:02;resnet_model/batch_normalization_1/beta/Initializer/zeros:0
ä
0resnet_model/batch_normalization_1/moving_mean:05resnet_model/batch_normalization_1/moving_mean/Assign5resnet_model/batch_normalization_1/moving_mean/read:02Bresnet_model/batch_normalization_1/moving_mean/Initializer/zeros:0
ó
4resnet_model/batch_normalization_1/moving_variance:09resnet_model/batch_normalization_1/moving_variance/Assign9resnet_model/batch_normalization_1/moving_variance/read:02Eresnet_model/batch_normalization_1/moving_variance/Initializer/ones:0
§
resnet_model/conv2d_3/kernel:0#resnet_model/conv2d_3/kernel/Assign#resnet_model/conv2d_3/kernel/read:02;resnet_model/conv2d_3/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_2/gamma:0/resnet_model/batch_normalization_2/gamma/Assign/resnet_model/batch_normalization_2/gamma/read:02;resnet_model/batch_normalization_2/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_2/beta:0.resnet_model/batch_normalization_2/beta/Assign.resnet_model/batch_normalization_2/beta/read:02;resnet_model/batch_normalization_2/beta/Initializer/zeros:0
ä
0resnet_model/batch_normalization_2/moving_mean:05resnet_model/batch_normalization_2/moving_mean/Assign5resnet_model/batch_normalization_2/moving_mean/read:02Bresnet_model/batch_normalization_2/moving_mean/Initializer/zeros:0
ó
4resnet_model/batch_normalization_2/moving_variance:09resnet_model/batch_normalization_2/moving_variance/Assign9resnet_model/batch_normalization_2/moving_variance/read:02Eresnet_model/batch_normalization_2/moving_variance/Initializer/ones:0
§
resnet_model/conv2d_4/kernel:0#resnet_model/conv2d_4/kernel/Assign#resnet_model/conv2d_4/kernel/read:02;resnet_model/conv2d_4/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_3/gamma:0/resnet_model/batch_normalization_3/gamma/Assign/resnet_model/batch_normalization_3/gamma/read:02;resnet_model/batch_normalization_3/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_3/beta:0.resnet_model/batch_normalization_3/beta/Assign.resnet_model/batch_normalization_3/beta/read:02;resnet_model/batch_normalization_3/beta/Initializer/zeros:0
ä
0resnet_model/batch_normalization_3/moving_mean:05resnet_model/batch_normalization_3/moving_mean/Assign5resnet_model/batch_normalization_3/moving_mean/read:02Bresnet_model/batch_normalization_3/moving_mean/Initializer/zeros:0
ó
4resnet_model/batch_normalization_3/moving_variance:09resnet_model/batch_normalization_3/moving_variance/Assign9resnet_model/batch_normalization_3/moving_variance/read:02Eresnet_model/batch_normalization_3/moving_variance/Initializer/ones:0
§
resnet_model/conv2d_5/kernel:0#resnet_model/conv2d_5/kernel/Assign#resnet_model/conv2d_5/kernel/read:02;resnet_model/conv2d_5/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_4/gamma:0/resnet_model/batch_normalization_4/gamma/Assign/resnet_model/batch_normalization_4/gamma/read:02;resnet_model/batch_normalization_4/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_4/beta:0.resnet_model/batch_normalization_4/beta/Assign.resnet_model/batch_normalization_4/beta/read:02;resnet_model/batch_normalization_4/beta/Initializer/zeros:0
ä
0resnet_model/batch_normalization_4/moving_mean:05resnet_model/batch_normalization_4/moving_mean/Assign5resnet_model/batch_normalization_4/moving_mean/read:02Bresnet_model/batch_normalization_4/moving_mean/Initializer/zeros:0
ó
4resnet_model/batch_normalization_4/moving_variance:09resnet_model/batch_normalization_4/moving_variance/Assign9resnet_model/batch_normalization_4/moving_variance/read:02Eresnet_model/batch_normalization_4/moving_variance/Initializer/ones:0
§
resnet_model/conv2d_6/kernel:0#resnet_model/conv2d_6/kernel/Assign#resnet_model/conv2d_6/kernel/read:02;resnet_model/conv2d_6/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_5/gamma:0/resnet_model/batch_normalization_5/gamma/Assign/resnet_model/batch_normalization_5/gamma/read:02;resnet_model/batch_normalization_5/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_5/beta:0.resnet_model/batch_normalization_5/beta/Assign.resnet_model/batch_normalization_5/beta/read:02;resnet_model/batch_normalization_5/beta/Initializer/zeros:0
ä
0resnet_model/batch_normalization_5/moving_mean:05resnet_model/batch_normalization_5/moving_mean/Assign5resnet_model/batch_normalization_5/moving_mean/read:02Bresnet_model/batch_normalization_5/moving_mean/Initializer/zeros:0
ó
4resnet_model/batch_normalization_5/moving_variance:09resnet_model/batch_normalization_5/moving_variance/Assign9resnet_model/batch_normalization_5/moving_variance/read:02Eresnet_model/batch_normalization_5/moving_variance/Initializer/ones:0
§
resnet_model/conv2d_7/kernel:0#resnet_model/conv2d_7/kernel/Assign#resnet_model/conv2d_7/kernel/read:02;resnet_model/conv2d_7/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_6/gamma:0/resnet_model/batch_normalization_6/gamma/Assign/resnet_model/batch_normalization_6/gamma/read:02;resnet_model/batch_normalization_6/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_6/beta:0.resnet_model/batch_normalization_6/beta/Assign.resnet_model/batch_normalization_6/beta/read:02;resnet_model/batch_normalization_6/beta/Initializer/zeros:0
ä
0resnet_model/batch_normalization_6/moving_mean:05resnet_model/batch_normalization_6/moving_mean/Assign5resnet_model/batch_normalization_6/moving_mean/read:02Bresnet_model/batch_normalization_6/moving_mean/Initializer/zeros:0
ó
4resnet_model/batch_normalization_6/moving_variance:09resnet_model/batch_normalization_6/moving_variance/Assign9resnet_model/batch_normalization_6/moving_variance/read:02Eresnet_model/batch_normalization_6/moving_variance/Initializer/ones:0
§
resnet_model/conv2d_8/kernel:0#resnet_model/conv2d_8/kernel/Assign#resnet_model/conv2d_8/kernel/read:02;resnet_model/conv2d_8/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_7/gamma:0/resnet_model/batch_normalization_7/gamma/Assign/resnet_model/batch_normalization_7/gamma/read:02;resnet_model/batch_normalization_7/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_7/beta:0.resnet_model/batch_normalization_7/beta/Assign.resnet_model/batch_normalization_7/beta/read:02;resnet_model/batch_normalization_7/beta/Initializer/zeros:0
ä
0resnet_model/batch_normalization_7/moving_mean:05resnet_model/batch_normalization_7/moving_mean/Assign5resnet_model/batch_normalization_7/moving_mean/read:02Bresnet_model/batch_normalization_7/moving_mean/Initializer/zeros:0
ó
4resnet_model/batch_normalization_7/moving_variance:09resnet_model/batch_normalization_7/moving_variance/Assign9resnet_model/batch_normalization_7/moving_variance/read:02Eresnet_model/batch_normalization_7/moving_variance/Initializer/ones:0
§
resnet_model/conv2d_9/kernel:0#resnet_model/conv2d_9/kernel/Assign#resnet_model/conv2d_9/kernel/read:02;resnet_model/conv2d_9/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_8/gamma:0/resnet_model/batch_normalization_8/gamma/Assign/resnet_model/batch_normalization_8/gamma/read:02;resnet_model/batch_normalization_8/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_8/beta:0.resnet_model/batch_normalization_8/beta/Assign.resnet_model/batch_normalization_8/beta/read:02;resnet_model/batch_normalization_8/beta/Initializer/zeros:0
ä
0resnet_model/batch_normalization_8/moving_mean:05resnet_model/batch_normalization_8/moving_mean/Assign5resnet_model/batch_normalization_8/moving_mean/read:02Bresnet_model/batch_normalization_8/moving_mean/Initializer/zeros:0
ó
4resnet_model/batch_normalization_8/moving_variance:09resnet_model/batch_normalization_8/moving_variance/Assign9resnet_model/batch_normalization_8/moving_variance/read:02Eresnet_model/batch_normalization_8/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_10/kernel:0$resnet_model/conv2d_10/kernel/Assign$resnet_model/conv2d_10/kernel/read:02<resnet_model/conv2d_10/kernel/Initializer/truncated_normal:0
Ë
*resnet_model/batch_normalization_9/gamma:0/resnet_model/batch_normalization_9/gamma/Assign/resnet_model/batch_normalization_9/gamma/read:02;resnet_model/batch_normalization_9/gamma/Initializer/ones:0
È
)resnet_model/batch_normalization_9/beta:0.resnet_model/batch_normalization_9/beta/Assign.resnet_model/batch_normalization_9/beta/read:02;resnet_model/batch_normalization_9/beta/Initializer/zeros:0
ä
0resnet_model/batch_normalization_9/moving_mean:05resnet_model/batch_normalization_9/moving_mean/Assign5resnet_model/batch_normalization_9/moving_mean/read:02Bresnet_model/batch_normalization_9/moving_mean/Initializer/zeros:0
ó
4resnet_model/batch_normalization_9/moving_variance:09resnet_model/batch_normalization_9/moving_variance/Assign9resnet_model/batch_normalization_9/moving_variance/read:02Eresnet_model/batch_normalization_9/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_11/kernel:0$resnet_model/conv2d_11/kernel/Assign$resnet_model/conv2d_11/kernel/read:02<resnet_model/conv2d_11/kernel/Initializer/truncated_normal:0
«
resnet_model/conv2d_12/kernel:0$resnet_model/conv2d_12/kernel/Assign$resnet_model/conv2d_12/kernel/read:02<resnet_model/conv2d_12/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_10/gamma:00resnet_model/batch_normalization_10/gamma/Assign0resnet_model/batch_normalization_10/gamma/read:02<resnet_model/batch_normalization_10/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_10/beta:0/resnet_model/batch_normalization_10/beta/Assign/resnet_model/batch_normalization_10/beta/read:02<resnet_model/batch_normalization_10/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_10/moving_mean:06resnet_model/batch_normalization_10/moving_mean/Assign6resnet_model/batch_normalization_10/moving_mean/read:02Cresnet_model/batch_normalization_10/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_10/moving_variance:0:resnet_model/batch_normalization_10/moving_variance/Assign:resnet_model/batch_normalization_10/moving_variance/read:02Fresnet_model/batch_normalization_10/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_13/kernel:0$resnet_model/conv2d_13/kernel/Assign$resnet_model/conv2d_13/kernel/read:02<resnet_model/conv2d_13/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_11/gamma:00resnet_model/batch_normalization_11/gamma/Assign0resnet_model/batch_normalization_11/gamma/read:02<resnet_model/batch_normalization_11/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_11/beta:0/resnet_model/batch_normalization_11/beta/Assign/resnet_model/batch_normalization_11/beta/read:02<resnet_model/batch_normalization_11/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_11/moving_mean:06resnet_model/batch_normalization_11/moving_mean/Assign6resnet_model/batch_normalization_11/moving_mean/read:02Cresnet_model/batch_normalization_11/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_11/moving_variance:0:resnet_model/batch_normalization_11/moving_variance/Assign:resnet_model/batch_normalization_11/moving_variance/read:02Fresnet_model/batch_normalization_11/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_14/kernel:0$resnet_model/conv2d_14/kernel/Assign$resnet_model/conv2d_14/kernel/read:02<resnet_model/conv2d_14/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_12/gamma:00resnet_model/batch_normalization_12/gamma/Assign0resnet_model/batch_normalization_12/gamma/read:02<resnet_model/batch_normalization_12/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_12/beta:0/resnet_model/batch_normalization_12/beta/Assign/resnet_model/batch_normalization_12/beta/read:02<resnet_model/batch_normalization_12/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_12/moving_mean:06resnet_model/batch_normalization_12/moving_mean/Assign6resnet_model/batch_normalization_12/moving_mean/read:02Cresnet_model/batch_normalization_12/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_12/moving_variance:0:resnet_model/batch_normalization_12/moving_variance/Assign:resnet_model/batch_normalization_12/moving_variance/read:02Fresnet_model/batch_normalization_12/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_15/kernel:0$resnet_model/conv2d_15/kernel/Assign$resnet_model/conv2d_15/kernel/read:02<resnet_model/conv2d_15/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_13/gamma:00resnet_model/batch_normalization_13/gamma/Assign0resnet_model/batch_normalization_13/gamma/read:02<resnet_model/batch_normalization_13/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_13/beta:0/resnet_model/batch_normalization_13/beta/Assign/resnet_model/batch_normalization_13/beta/read:02<resnet_model/batch_normalization_13/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_13/moving_mean:06resnet_model/batch_normalization_13/moving_mean/Assign6resnet_model/batch_normalization_13/moving_mean/read:02Cresnet_model/batch_normalization_13/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_13/moving_variance:0:resnet_model/batch_normalization_13/moving_variance/Assign:resnet_model/batch_normalization_13/moving_variance/read:02Fresnet_model/batch_normalization_13/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_16/kernel:0$resnet_model/conv2d_16/kernel/Assign$resnet_model/conv2d_16/kernel/read:02<resnet_model/conv2d_16/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_14/gamma:00resnet_model/batch_normalization_14/gamma/Assign0resnet_model/batch_normalization_14/gamma/read:02<resnet_model/batch_normalization_14/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_14/beta:0/resnet_model/batch_normalization_14/beta/Assign/resnet_model/batch_normalization_14/beta/read:02<resnet_model/batch_normalization_14/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_14/moving_mean:06resnet_model/batch_normalization_14/moving_mean/Assign6resnet_model/batch_normalization_14/moving_mean/read:02Cresnet_model/batch_normalization_14/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_14/moving_variance:0:resnet_model/batch_normalization_14/moving_variance/Assign:resnet_model/batch_normalization_14/moving_variance/read:02Fresnet_model/batch_normalization_14/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_17/kernel:0$resnet_model/conv2d_17/kernel/Assign$resnet_model/conv2d_17/kernel/read:02<resnet_model/conv2d_17/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_15/gamma:00resnet_model/batch_normalization_15/gamma/Assign0resnet_model/batch_normalization_15/gamma/read:02<resnet_model/batch_normalization_15/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_15/beta:0/resnet_model/batch_normalization_15/beta/Assign/resnet_model/batch_normalization_15/beta/read:02<resnet_model/batch_normalization_15/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_15/moving_mean:06resnet_model/batch_normalization_15/moving_mean/Assign6resnet_model/batch_normalization_15/moving_mean/read:02Cresnet_model/batch_normalization_15/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_15/moving_variance:0:resnet_model/batch_normalization_15/moving_variance/Assign:resnet_model/batch_normalization_15/moving_variance/read:02Fresnet_model/batch_normalization_15/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_18/kernel:0$resnet_model/conv2d_18/kernel/Assign$resnet_model/conv2d_18/kernel/read:02<resnet_model/conv2d_18/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_16/gamma:00resnet_model/batch_normalization_16/gamma/Assign0resnet_model/batch_normalization_16/gamma/read:02<resnet_model/batch_normalization_16/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_16/beta:0/resnet_model/batch_normalization_16/beta/Assign/resnet_model/batch_normalization_16/beta/read:02<resnet_model/batch_normalization_16/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_16/moving_mean:06resnet_model/batch_normalization_16/moving_mean/Assign6resnet_model/batch_normalization_16/moving_mean/read:02Cresnet_model/batch_normalization_16/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_16/moving_variance:0:resnet_model/batch_normalization_16/moving_variance/Assign:resnet_model/batch_normalization_16/moving_variance/read:02Fresnet_model/batch_normalization_16/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_19/kernel:0$resnet_model/conv2d_19/kernel/Assign$resnet_model/conv2d_19/kernel/read:02<resnet_model/conv2d_19/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_17/gamma:00resnet_model/batch_normalization_17/gamma/Assign0resnet_model/batch_normalization_17/gamma/read:02<resnet_model/batch_normalization_17/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_17/beta:0/resnet_model/batch_normalization_17/beta/Assign/resnet_model/batch_normalization_17/beta/read:02<resnet_model/batch_normalization_17/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_17/moving_mean:06resnet_model/batch_normalization_17/moving_mean/Assign6resnet_model/batch_normalization_17/moving_mean/read:02Cresnet_model/batch_normalization_17/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_17/moving_variance:0:resnet_model/batch_normalization_17/moving_variance/Assign:resnet_model/batch_normalization_17/moving_variance/read:02Fresnet_model/batch_normalization_17/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_20/kernel:0$resnet_model/conv2d_20/kernel/Assign$resnet_model/conv2d_20/kernel/read:02<resnet_model/conv2d_20/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_18/gamma:00resnet_model/batch_normalization_18/gamma/Assign0resnet_model/batch_normalization_18/gamma/read:02<resnet_model/batch_normalization_18/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_18/beta:0/resnet_model/batch_normalization_18/beta/Assign/resnet_model/batch_normalization_18/beta/read:02<resnet_model/batch_normalization_18/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_18/moving_mean:06resnet_model/batch_normalization_18/moving_mean/Assign6resnet_model/batch_normalization_18/moving_mean/read:02Cresnet_model/batch_normalization_18/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_18/moving_variance:0:resnet_model/batch_normalization_18/moving_variance/Assign:resnet_model/batch_normalization_18/moving_variance/read:02Fresnet_model/batch_normalization_18/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_21/kernel:0$resnet_model/conv2d_21/kernel/Assign$resnet_model/conv2d_21/kernel/read:02<resnet_model/conv2d_21/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_19/gamma:00resnet_model/batch_normalization_19/gamma/Assign0resnet_model/batch_normalization_19/gamma/read:02<resnet_model/batch_normalization_19/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_19/beta:0/resnet_model/batch_normalization_19/beta/Assign/resnet_model/batch_normalization_19/beta/read:02<resnet_model/batch_normalization_19/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_19/moving_mean:06resnet_model/batch_normalization_19/moving_mean/Assign6resnet_model/batch_normalization_19/moving_mean/read:02Cresnet_model/batch_normalization_19/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_19/moving_variance:0:resnet_model/batch_normalization_19/moving_variance/Assign:resnet_model/batch_normalization_19/moving_variance/read:02Fresnet_model/batch_normalization_19/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_22/kernel:0$resnet_model/conv2d_22/kernel/Assign$resnet_model/conv2d_22/kernel/read:02<resnet_model/conv2d_22/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_20/gamma:00resnet_model/batch_normalization_20/gamma/Assign0resnet_model/batch_normalization_20/gamma/read:02<resnet_model/batch_normalization_20/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_20/beta:0/resnet_model/batch_normalization_20/beta/Assign/resnet_model/batch_normalization_20/beta/read:02<resnet_model/batch_normalization_20/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_20/moving_mean:06resnet_model/batch_normalization_20/moving_mean/Assign6resnet_model/batch_normalization_20/moving_mean/read:02Cresnet_model/batch_normalization_20/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_20/moving_variance:0:resnet_model/batch_normalization_20/moving_variance/Assign:resnet_model/batch_normalization_20/moving_variance/read:02Fresnet_model/batch_normalization_20/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_23/kernel:0$resnet_model/conv2d_23/kernel/Assign$resnet_model/conv2d_23/kernel/read:02<resnet_model/conv2d_23/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_21/gamma:00resnet_model/batch_normalization_21/gamma/Assign0resnet_model/batch_normalization_21/gamma/read:02<resnet_model/batch_normalization_21/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_21/beta:0/resnet_model/batch_normalization_21/beta/Assign/resnet_model/batch_normalization_21/beta/read:02<resnet_model/batch_normalization_21/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_21/moving_mean:06resnet_model/batch_normalization_21/moving_mean/Assign6resnet_model/batch_normalization_21/moving_mean/read:02Cresnet_model/batch_normalization_21/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_21/moving_variance:0:resnet_model/batch_normalization_21/moving_variance/Assign:resnet_model/batch_normalization_21/moving_variance/read:02Fresnet_model/batch_normalization_21/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_24/kernel:0$resnet_model/conv2d_24/kernel/Assign$resnet_model/conv2d_24/kernel/read:02<resnet_model/conv2d_24/kernel/Initializer/truncated_normal:0
«
resnet_model/conv2d_25/kernel:0$resnet_model/conv2d_25/kernel/Assign$resnet_model/conv2d_25/kernel/read:02<resnet_model/conv2d_25/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_22/gamma:00resnet_model/batch_normalization_22/gamma/Assign0resnet_model/batch_normalization_22/gamma/read:02<resnet_model/batch_normalization_22/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_22/beta:0/resnet_model/batch_normalization_22/beta/Assign/resnet_model/batch_normalization_22/beta/read:02<resnet_model/batch_normalization_22/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_22/moving_mean:06resnet_model/batch_normalization_22/moving_mean/Assign6resnet_model/batch_normalization_22/moving_mean/read:02Cresnet_model/batch_normalization_22/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_22/moving_variance:0:resnet_model/batch_normalization_22/moving_variance/Assign:resnet_model/batch_normalization_22/moving_variance/read:02Fresnet_model/batch_normalization_22/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_26/kernel:0$resnet_model/conv2d_26/kernel/Assign$resnet_model/conv2d_26/kernel/read:02<resnet_model/conv2d_26/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_23/gamma:00resnet_model/batch_normalization_23/gamma/Assign0resnet_model/batch_normalization_23/gamma/read:02<resnet_model/batch_normalization_23/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_23/beta:0/resnet_model/batch_normalization_23/beta/Assign/resnet_model/batch_normalization_23/beta/read:02<resnet_model/batch_normalization_23/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_23/moving_mean:06resnet_model/batch_normalization_23/moving_mean/Assign6resnet_model/batch_normalization_23/moving_mean/read:02Cresnet_model/batch_normalization_23/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_23/moving_variance:0:resnet_model/batch_normalization_23/moving_variance/Assign:resnet_model/batch_normalization_23/moving_variance/read:02Fresnet_model/batch_normalization_23/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_27/kernel:0$resnet_model/conv2d_27/kernel/Assign$resnet_model/conv2d_27/kernel/read:02<resnet_model/conv2d_27/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_24/gamma:00resnet_model/batch_normalization_24/gamma/Assign0resnet_model/batch_normalization_24/gamma/read:02<resnet_model/batch_normalization_24/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_24/beta:0/resnet_model/batch_normalization_24/beta/Assign/resnet_model/batch_normalization_24/beta/read:02<resnet_model/batch_normalization_24/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_24/moving_mean:06resnet_model/batch_normalization_24/moving_mean/Assign6resnet_model/batch_normalization_24/moving_mean/read:02Cresnet_model/batch_normalization_24/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_24/moving_variance:0:resnet_model/batch_normalization_24/moving_variance/Assign:resnet_model/batch_normalization_24/moving_variance/read:02Fresnet_model/batch_normalization_24/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_28/kernel:0$resnet_model/conv2d_28/kernel/Assign$resnet_model/conv2d_28/kernel/read:02<resnet_model/conv2d_28/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_25/gamma:00resnet_model/batch_normalization_25/gamma/Assign0resnet_model/batch_normalization_25/gamma/read:02<resnet_model/batch_normalization_25/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_25/beta:0/resnet_model/batch_normalization_25/beta/Assign/resnet_model/batch_normalization_25/beta/read:02<resnet_model/batch_normalization_25/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_25/moving_mean:06resnet_model/batch_normalization_25/moving_mean/Assign6resnet_model/batch_normalization_25/moving_mean/read:02Cresnet_model/batch_normalization_25/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_25/moving_variance:0:resnet_model/batch_normalization_25/moving_variance/Assign:resnet_model/batch_normalization_25/moving_variance/read:02Fresnet_model/batch_normalization_25/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_29/kernel:0$resnet_model/conv2d_29/kernel/Assign$resnet_model/conv2d_29/kernel/read:02<resnet_model/conv2d_29/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_26/gamma:00resnet_model/batch_normalization_26/gamma/Assign0resnet_model/batch_normalization_26/gamma/read:02<resnet_model/batch_normalization_26/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_26/beta:0/resnet_model/batch_normalization_26/beta/Assign/resnet_model/batch_normalization_26/beta/read:02<resnet_model/batch_normalization_26/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_26/moving_mean:06resnet_model/batch_normalization_26/moving_mean/Assign6resnet_model/batch_normalization_26/moving_mean/read:02Cresnet_model/batch_normalization_26/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_26/moving_variance:0:resnet_model/batch_normalization_26/moving_variance/Assign:resnet_model/batch_normalization_26/moving_variance/read:02Fresnet_model/batch_normalization_26/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_30/kernel:0$resnet_model/conv2d_30/kernel/Assign$resnet_model/conv2d_30/kernel/read:02<resnet_model/conv2d_30/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_27/gamma:00resnet_model/batch_normalization_27/gamma/Assign0resnet_model/batch_normalization_27/gamma/read:02<resnet_model/batch_normalization_27/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_27/beta:0/resnet_model/batch_normalization_27/beta/Assign/resnet_model/batch_normalization_27/beta/read:02<resnet_model/batch_normalization_27/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_27/moving_mean:06resnet_model/batch_normalization_27/moving_mean/Assign6resnet_model/batch_normalization_27/moving_mean/read:02Cresnet_model/batch_normalization_27/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_27/moving_variance:0:resnet_model/batch_normalization_27/moving_variance/Assign:resnet_model/batch_normalization_27/moving_variance/read:02Fresnet_model/batch_normalization_27/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_31/kernel:0$resnet_model/conv2d_31/kernel/Assign$resnet_model/conv2d_31/kernel/read:02<resnet_model/conv2d_31/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_28/gamma:00resnet_model/batch_normalization_28/gamma/Assign0resnet_model/batch_normalization_28/gamma/read:02<resnet_model/batch_normalization_28/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_28/beta:0/resnet_model/batch_normalization_28/beta/Assign/resnet_model/batch_normalization_28/beta/read:02<resnet_model/batch_normalization_28/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_28/moving_mean:06resnet_model/batch_normalization_28/moving_mean/Assign6resnet_model/batch_normalization_28/moving_mean/read:02Cresnet_model/batch_normalization_28/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_28/moving_variance:0:resnet_model/batch_normalization_28/moving_variance/Assign:resnet_model/batch_normalization_28/moving_variance/read:02Fresnet_model/batch_normalization_28/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_32/kernel:0$resnet_model/conv2d_32/kernel/Assign$resnet_model/conv2d_32/kernel/read:02<resnet_model/conv2d_32/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_29/gamma:00resnet_model/batch_normalization_29/gamma/Assign0resnet_model/batch_normalization_29/gamma/read:02<resnet_model/batch_normalization_29/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_29/beta:0/resnet_model/batch_normalization_29/beta/Assign/resnet_model/batch_normalization_29/beta/read:02<resnet_model/batch_normalization_29/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_29/moving_mean:06resnet_model/batch_normalization_29/moving_mean/Assign6resnet_model/batch_normalization_29/moving_mean/read:02Cresnet_model/batch_normalization_29/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_29/moving_variance:0:resnet_model/batch_normalization_29/moving_variance/Assign:resnet_model/batch_normalization_29/moving_variance/read:02Fresnet_model/batch_normalization_29/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_33/kernel:0$resnet_model/conv2d_33/kernel/Assign$resnet_model/conv2d_33/kernel/read:02<resnet_model/conv2d_33/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_30/gamma:00resnet_model/batch_normalization_30/gamma/Assign0resnet_model/batch_normalization_30/gamma/read:02<resnet_model/batch_normalization_30/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_30/beta:0/resnet_model/batch_normalization_30/beta/Assign/resnet_model/batch_normalization_30/beta/read:02<resnet_model/batch_normalization_30/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_30/moving_mean:06resnet_model/batch_normalization_30/moving_mean/Assign6resnet_model/batch_normalization_30/moving_mean/read:02Cresnet_model/batch_normalization_30/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_30/moving_variance:0:resnet_model/batch_normalization_30/moving_variance/Assign:resnet_model/batch_normalization_30/moving_variance/read:02Fresnet_model/batch_normalization_30/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_34/kernel:0$resnet_model/conv2d_34/kernel/Assign$resnet_model/conv2d_34/kernel/read:02<resnet_model/conv2d_34/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_31/gamma:00resnet_model/batch_normalization_31/gamma/Assign0resnet_model/batch_normalization_31/gamma/read:02<resnet_model/batch_normalization_31/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_31/beta:0/resnet_model/batch_normalization_31/beta/Assign/resnet_model/batch_normalization_31/beta/read:02<resnet_model/batch_normalization_31/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_31/moving_mean:06resnet_model/batch_normalization_31/moving_mean/Assign6resnet_model/batch_normalization_31/moving_mean/read:02Cresnet_model/batch_normalization_31/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_31/moving_variance:0:resnet_model/batch_normalization_31/moving_variance/Assign:resnet_model/batch_normalization_31/moving_variance/read:02Fresnet_model/batch_normalization_31/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_35/kernel:0$resnet_model/conv2d_35/kernel/Assign$resnet_model/conv2d_35/kernel/read:02<resnet_model/conv2d_35/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_32/gamma:00resnet_model/batch_normalization_32/gamma/Assign0resnet_model/batch_normalization_32/gamma/read:02<resnet_model/batch_normalization_32/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_32/beta:0/resnet_model/batch_normalization_32/beta/Assign/resnet_model/batch_normalization_32/beta/read:02<resnet_model/batch_normalization_32/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_32/moving_mean:06resnet_model/batch_normalization_32/moving_mean/Assign6resnet_model/batch_normalization_32/moving_mean/read:02Cresnet_model/batch_normalization_32/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_32/moving_variance:0:resnet_model/batch_normalization_32/moving_variance/Assign:resnet_model/batch_normalization_32/moving_variance/read:02Fresnet_model/batch_normalization_32/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_36/kernel:0$resnet_model/conv2d_36/kernel/Assign$resnet_model/conv2d_36/kernel/read:02<resnet_model/conv2d_36/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_33/gamma:00resnet_model/batch_normalization_33/gamma/Assign0resnet_model/batch_normalization_33/gamma/read:02<resnet_model/batch_normalization_33/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_33/beta:0/resnet_model/batch_normalization_33/beta/Assign/resnet_model/batch_normalization_33/beta/read:02<resnet_model/batch_normalization_33/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_33/moving_mean:06resnet_model/batch_normalization_33/moving_mean/Assign6resnet_model/batch_normalization_33/moving_mean/read:02Cresnet_model/batch_normalization_33/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_33/moving_variance:0:resnet_model/batch_normalization_33/moving_variance/Assign:resnet_model/batch_normalization_33/moving_variance/read:02Fresnet_model/batch_normalization_33/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_37/kernel:0$resnet_model/conv2d_37/kernel/Assign$resnet_model/conv2d_37/kernel/read:02<resnet_model/conv2d_37/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_34/gamma:00resnet_model/batch_normalization_34/gamma/Assign0resnet_model/batch_normalization_34/gamma/read:02<resnet_model/batch_normalization_34/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_34/beta:0/resnet_model/batch_normalization_34/beta/Assign/resnet_model/batch_normalization_34/beta/read:02<resnet_model/batch_normalization_34/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_34/moving_mean:06resnet_model/batch_normalization_34/moving_mean/Assign6resnet_model/batch_normalization_34/moving_mean/read:02Cresnet_model/batch_normalization_34/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_34/moving_variance:0:resnet_model/batch_normalization_34/moving_variance/Assign:resnet_model/batch_normalization_34/moving_variance/read:02Fresnet_model/batch_normalization_34/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_38/kernel:0$resnet_model/conv2d_38/kernel/Assign$resnet_model/conv2d_38/kernel/read:02<resnet_model/conv2d_38/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_35/gamma:00resnet_model/batch_normalization_35/gamma/Assign0resnet_model/batch_normalization_35/gamma/read:02<resnet_model/batch_normalization_35/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_35/beta:0/resnet_model/batch_normalization_35/beta/Assign/resnet_model/batch_normalization_35/beta/read:02<resnet_model/batch_normalization_35/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_35/moving_mean:06resnet_model/batch_normalization_35/moving_mean/Assign6resnet_model/batch_normalization_35/moving_mean/read:02Cresnet_model/batch_normalization_35/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_35/moving_variance:0:resnet_model/batch_normalization_35/moving_variance/Assign:resnet_model/batch_normalization_35/moving_variance/read:02Fresnet_model/batch_normalization_35/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_39/kernel:0$resnet_model/conv2d_39/kernel/Assign$resnet_model/conv2d_39/kernel/read:02<resnet_model/conv2d_39/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_36/gamma:00resnet_model/batch_normalization_36/gamma/Assign0resnet_model/batch_normalization_36/gamma/read:02<resnet_model/batch_normalization_36/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_36/beta:0/resnet_model/batch_normalization_36/beta/Assign/resnet_model/batch_normalization_36/beta/read:02<resnet_model/batch_normalization_36/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_36/moving_mean:06resnet_model/batch_normalization_36/moving_mean/Assign6resnet_model/batch_normalization_36/moving_mean/read:02Cresnet_model/batch_normalization_36/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_36/moving_variance:0:resnet_model/batch_normalization_36/moving_variance/Assign:resnet_model/batch_normalization_36/moving_variance/read:02Fresnet_model/batch_normalization_36/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_40/kernel:0$resnet_model/conv2d_40/kernel/Assign$resnet_model/conv2d_40/kernel/read:02<resnet_model/conv2d_40/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_37/gamma:00resnet_model/batch_normalization_37/gamma/Assign0resnet_model/batch_normalization_37/gamma/read:02<resnet_model/batch_normalization_37/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_37/beta:0/resnet_model/batch_normalization_37/beta/Assign/resnet_model/batch_normalization_37/beta/read:02<resnet_model/batch_normalization_37/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_37/moving_mean:06resnet_model/batch_normalization_37/moving_mean/Assign6resnet_model/batch_normalization_37/moving_mean/read:02Cresnet_model/batch_normalization_37/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_37/moving_variance:0:resnet_model/batch_normalization_37/moving_variance/Assign:resnet_model/batch_normalization_37/moving_variance/read:02Fresnet_model/batch_normalization_37/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_41/kernel:0$resnet_model/conv2d_41/kernel/Assign$resnet_model/conv2d_41/kernel/read:02<resnet_model/conv2d_41/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_38/gamma:00resnet_model/batch_normalization_38/gamma/Assign0resnet_model/batch_normalization_38/gamma/read:02<resnet_model/batch_normalization_38/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_38/beta:0/resnet_model/batch_normalization_38/beta/Assign/resnet_model/batch_normalization_38/beta/read:02<resnet_model/batch_normalization_38/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_38/moving_mean:06resnet_model/batch_normalization_38/moving_mean/Assign6resnet_model/batch_normalization_38/moving_mean/read:02Cresnet_model/batch_normalization_38/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_38/moving_variance:0:resnet_model/batch_normalization_38/moving_variance/Assign:resnet_model/batch_normalization_38/moving_variance/read:02Fresnet_model/batch_normalization_38/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_42/kernel:0$resnet_model/conv2d_42/kernel/Assign$resnet_model/conv2d_42/kernel/read:02<resnet_model/conv2d_42/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_39/gamma:00resnet_model/batch_normalization_39/gamma/Assign0resnet_model/batch_normalization_39/gamma/read:02<resnet_model/batch_normalization_39/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_39/beta:0/resnet_model/batch_normalization_39/beta/Assign/resnet_model/batch_normalization_39/beta/read:02<resnet_model/batch_normalization_39/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_39/moving_mean:06resnet_model/batch_normalization_39/moving_mean/Assign6resnet_model/batch_normalization_39/moving_mean/read:02Cresnet_model/batch_normalization_39/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_39/moving_variance:0:resnet_model/batch_normalization_39/moving_variance/Assign:resnet_model/batch_normalization_39/moving_variance/read:02Fresnet_model/batch_normalization_39/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_43/kernel:0$resnet_model/conv2d_43/kernel/Assign$resnet_model/conv2d_43/kernel/read:02<resnet_model/conv2d_43/kernel/Initializer/truncated_normal:0
«
resnet_model/conv2d_44/kernel:0$resnet_model/conv2d_44/kernel/Assign$resnet_model/conv2d_44/kernel/read:02<resnet_model/conv2d_44/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_40/gamma:00resnet_model/batch_normalization_40/gamma/Assign0resnet_model/batch_normalization_40/gamma/read:02<resnet_model/batch_normalization_40/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_40/beta:0/resnet_model/batch_normalization_40/beta/Assign/resnet_model/batch_normalization_40/beta/read:02<resnet_model/batch_normalization_40/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_40/moving_mean:06resnet_model/batch_normalization_40/moving_mean/Assign6resnet_model/batch_normalization_40/moving_mean/read:02Cresnet_model/batch_normalization_40/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_40/moving_variance:0:resnet_model/batch_normalization_40/moving_variance/Assign:resnet_model/batch_normalization_40/moving_variance/read:02Fresnet_model/batch_normalization_40/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_45/kernel:0$resnet_model/conv2d_45/kernel/Assign$resnet_model/conv2d_45/kernel/read:02<resnet_model/conv2d_45/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_41/gamma:00resnet_model/batch_normalization_41/gamma/Assign0resnet_model/batch_normalization_41/gamma/read:02<resnet_model/batch_normalization_41/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_41/beta:0/resnet_model/batch_normalization_41/beta/Assign/resnet_model/batch_normalization_41/beta/read:02<resnet_model/batch_normalization_41/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_41/moving_mean:06resnet_model/batch_normalization_41/moving_mean/Assign6resnet_model/batch_normalization_41/moving_mean/read:02Cresnet_model/batch_normalization_41/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_41/moving_variance:0:resnet_model/batch_normalization_41/moving_variance/Assign:resnet_model/batch_normalization_41/moving_variance/read:02Fresnet_model/batch_normalization_41/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_46/kernel:0$resnet_model/conv2d_46/kernel/Assign$resnet_model/conv2d_46/kernel/read:02<resnet_model/conv2d_46/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_42/gamma:00resnet_model/batch_normalization_42/gamma/Assign0resnet_model/batch_normalization_42/gamma/read:02<resnet_model/batch_normalization_42/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_42/beta:0/resnet_model/batch_normalization_42/beta/Assign/resnet_model/batch_normalization_42/beta/read:02<resnet_model/batch_normalization_42/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_42/moving_mean:06resnet_model/batch_normalization_42/moving_mean/Assign6resnet_model/batch_normalization_42/moving_mean/read:02Cresnet_model/batch_normalization_42/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_42/moving_variance:0:resnet_model/batch_normalization_42/moving_variance/Assign:resnet_model/batch_normalization_42/moving_variance/read:02Fresnet_model/batch_normalization_42/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_47/kernel:0$resnet_model/conv2d_47/kernel/Assign$resnet_model/conv2d_47/kernel/read:02<resnet_model/conv2d_47/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_43/gamma:00resnet_model/batch_normalization_43/gamma/Assign0resnet_model/batch_normalization_43/gamma/read:02<resnet_model/batch_normalization_43/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_43/beta:0/resnet_model/batch_normalization_43/beta/Assign/resnet_model/batch_normalization_43/beta/read:02<resnet_model/batch_normalization_43/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_43/moving_mean:06resnet_model/batch_normalization_43/moving_mean/Assign6resnet_model/batch_normalization_43/moving_mean/read:02Cresnet_model/batch_normalization_43/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_43/moving_variance:0:resnet_model/batch_normalization_43/moving_variance/Assign:resnet_model/batch_normalization_43/moving_variance/read:02Fresnet_model/batch_normalization_43/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_48/kernel:0$resnet_model/conv2d_48/kernel/Assign$resnet_model/conv2d_48/kernel/read:02<resnet_model/conv2d_48/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_44/gamma:00resnet_model/batch_normalization_44/gamma/Assign0resnet_model/batch_normalization_44/gamma/read:02<resnet_model/batch_normalization_44/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_44/beta:0/resnet_model/batch_normalization_44/beta/Assign/resnet_model/batch_normalization_44/beta/read:02<resnet_model/batch_normalization_44/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_44/moving_mean:06resnet_model/batch_normalization_44/moving_mean/Assign6resnet_model/batch_normalization_44/moving_mean/read:02Cresnet_model/batch_normalization_44/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_44/moving_variance:0:resnet_model/batch_normalization_44/moving_variance/Assign:resnet_model/batch_normalization_44/moving_variance/read:02Fresnet_model/batch_normalization_44/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_49/kernel:0$resnet_model/conv2d_49/kernel/Assign$resnet_model/conv2d_49/kernel/read:02<resnet_model/conv2d_49/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_45/gamma:00resnet_model/batch_normalization_45/gamma/Assign0resnet_model/batch_normalization_45/gamma/read:02<resnet_model/batch_normalization_45/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_45/beta:0/resnet_model/batch_normalization_45/beta/Assign/resnet_model/batch_normalization_45/beta/read:02<resnet_model/batch_normalization_45/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_45/moving_mean:06resnet_model/batch_normalization_45/moving_mean/Assign6resnet_model/batch_normalization_45/moving_mean/read:02Cresnet_model/batch_normalization_45/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_45/moving_variance:0:resnet_model/batch_normalization_45/moving_variance/Assign:resnet_model/batch_normalization_45/moving_variance/read:02Fresnet_model/batch_normalization_45/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_50/kernel:0$resnet_model/conv2d_50/kernel/Assign$resnet_model/conv2d_50/kernel/read:02<resnet_model/conv2d_50/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_46/gamma:00resnet_model/batch_normalization_46/gamma/Assign0resnet_model/batch_normalization_46/gamma/read:02<resnet_model/batch_normalization_46/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_46/beta:0/resnet_model/batch_normalization_46/beta/Assign/resnet_model/batch_normalization_46/beta/read:02<resnet_model/batch_normalization_46/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_46/moving_mean:06resnet_model/batch_normalization_46/moving_mean/Assign6resnet_model/batch_normalization_46/moving_mean/read:02Cresnet_model/batch_normalization_46/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_46/moving_variance:0:resnet_model/batch_normalization_46/moving_variance/Assign:resnet_model/batch_normalization_46/moving_variance/read:02Fresnet_model/batch_normalization_46/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_51/kernel:0$resnet_model/conv2d_51/kernel/Assign$resnet_model/conv2d_51/kernel/read:02<resnet_model/conv2d_51/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_47/gamma:00resnet_model/batch_normalization_47/gamma/Assign0resnet_model/batch_normalization_47/gamma/read:02<resnet_model/batch_normalization_47/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_47/beta:0/resnet_model/batch_normalization_47/beta/Assign/resnet_model/batch_normalization_47/beta/read:02<resnet_model/batch_normalization_47/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_47/moving_mean:06resnet_model/batch_normalization_47/moving_mean/Assign6resnet_model/batch_normalization_47/moving_mean/read:02Cresnet_model/batch_normalization_47/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_47/moving_variance:0:resnet_model/batch_normalization_47/moving_variance/Assign:resnet_model/batch_normalization_47/moving_variance/read:02Fresnet_model/batch_normalization_47/moving_variance/Initializer/ones:0
«
resnet_model/conv2d_52/kernel:0$resnet_model/conv2d_52/kernel/Assign$resnet_model/conv2d_52/kernel/read:02<resnet_model/conv2d_52/kernel/Initializer/truncated_normal:0
Ï
+resnet_model/batch_normalization_48/gamma:00resnet_model/batch_normalization_48/gamma/Assign0resnet_model/batch_normalization_48/gamma/read:02<resnet_model/batch_normalization_48/gamma/Initializer/ones:0
Ì
*resnet_model/batch_normalization_48/beta:0/resnet_model/batch_normalization_48/beta/Assign/resnet_model/batch_normalization_48/beta/read:02<resnet_model/batch_normalization_48/beta/Initializer/zeros:0
è
1resnet_model/batch_normalization_48/moving_mean:06resnet_model/batch_normalization_48/moving_mean/Assign6resnet_model/batch_normalization_48/moving_mean/read:02Cresnet_model/batch_normalization_48/moving_mean/Initializer/zeros:0
÷
5resnet_model/batch_normalization_48/moving_variance:0:resnet_model/batch_normalization_48/moving_variance/Assign:resnet_model/batch_normalization_48/moving_variance/read:02Fresnet_model/batch_normalization_48/moving_variance/Initializer/ones:0

resnet_model/dense/kernel:0 resnet_model/dense/kernel/Assign resnet_model/dense/kernel/read:026resnet_model/dense/kernel/Initializer/random_uniform:0

resnet_model/dense/bias:0resnet_model/dense/bias/Assignresnet_model/dense/bias/read:02+resnet_model/dense/bias/Initializer/zeros:0" 
legacy_init_op


group_deps*±
predict¥
0
input'
input_tensor:0àà2
probabilities!
probabilities_2:0
é!
classes
classes_2:0	tensorflow/serving/predict*¹
serving_default¥
0
input'
input_tensor:0àà2
probabilities!
probabilities_1:0
é!
classes
classes_1:0	tensorflow/serving/predict