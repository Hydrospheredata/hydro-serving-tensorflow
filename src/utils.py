import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework.tensor_util import _AssertCompatible, _GetDenseDimensions, _FlattenToStrings, GetNumpyAppendFn
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import compat
import numpy as np
import hydro_serving_grpc as hs

DTYPE_TO_GETTER = {
    hs.DT_STRING: lambda x: x.string_val,

    hs.DT_BOOL: lambda x: x.bool_val,

    hs.DT_COMPLEX64: lambda x: x.scomplex_val,
    hs.DT_COMPLEX128: lambda x: x.dcomplex_val,

    hs.DT_FLOAT: lambda x: x.float_val,
    hs.DT_DOUBLE: lambda x: x.double_val,

    hs.DT_INT8: lambda x: x.int_val,
    hs.DT_INT16: lambda x: x.int_val,
    hs.DT_INT32: lambda x: x.int_val,
    hs.DT_INT64: lambda x: x.int64_val,

    hs.DT_UINT8: lambda x: x.int_val,
    hs.DT_UINT16: lambda x: x.uint32_val,
    hs.DT_UINT32: lambda x: x.uint32_val,
    hs.DT_UINT64: lambda x: x.uint64_val,

    hs.DT_HALF: lambda x: x.half_val,

    hs.DT_MAP: lambda x: x.map_val
}

def convert_to_python(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.matrix):
        return data.tolist()
    else:
        print("{0} isn't convertible to python".format(type(data)))
        return data


def extract_tensor_data(tensor):
    dtype = tensor.dtype
    getter = DTYPE_TO_GETTER[dtype]
    return getter(tensor)


def create_tensor(data, dtype, shape):
    getter = DTYPE_TO_GETTER[dtype]
    tensor = hs.TensorProto(
        type=dtype,
        tensor_shape=shape.as_proto()
    )
    field_ref = getter(tensor)
    field_ref = data
    return tensor


def convert_data_to_tensor_shape(data, shape: tf.TensorShape):
    if shape.ndims == 2:
        return np.matrix(data)
    elif shape.ndims > 2:
        return np.array(data)
    else:
        return data


def load_and_optimize(model_path):
    with tf.Session() as temp_sess:
        meta_graph = tf.saved_model.loader.load(temp_sess, [tf.saved_model.tag_constants.SERVING], model_path)
        print("Model loaded.")
        signatures = {}
        for name, sig in meta_graph.signature_def.items():
            signatures["inputs"] = {}
            for inp_name, inp in sig.inputs.items():
                signatures["inputs"][inp_name] = temp_sess.graph.get_tensor_by_name(inp_name)

            signatures["outputs"] = {}
            for out_name, out in sig.outputs.items():
                signatures["outputs"][out_name] = temp_sess.graph.get_tensor_by_name(out_name)
        print("Loaded a model with signature: {}".format(signatures))


def fixed_make_tensor_proto(values, dtype=None, shape=None, verify_shape=False):
    """Create a TensorProto.

    Args:
      values:         Values to put in the TensorProto.
      dtype:          Optional tensor_pb2 DataType value.
      shape:          List of integers representing the dimensions of tensor.
      verify_shape:   Boolean that enables verification of a shape of values.

    Returns:
      A `TensorProto`. Depending on the type, it may contain data in the
      "tensor_content" attribute, which is not directly useful to Python programs.
      To access the values you should convert the proto back to a numpy ndarray
      with `tensor_util.MakeNdarray(proto)`.

      If `values` is a `TensorProto`, it is immediately returned; `dtype` and
      `shape` are ignored.

    Raises:
      TypeError:  if unsupported types are provided.
      ValueError: if arguments have inappropriate values or if verify_shape is
       True and shape of values is not equals to a shape from the argument.

    make_tensor_proto accepts "values" of a python scalar, a python list, a
    numpy ndarray, or a numpy scalar.

    If "values" is a python scalar or a python list, make_tensor_proto
    first convert it to numpy ndarray. If dtype is None, the
    conversion tries its best to infer the right numpy data
    type. Otherwise, the resulting numpy array has a compatible data
    type with the given dtype.

    In either case above, the numpy ndarray (either the caller provided
    or the auto converted) must have the compatible type with dtype.

    make_tensor_proto then converts the numpy array to a tensor proto.

    If "shape" is None, the resulting tensor proto represents the numpy
    array precisely.

    Otherwise, "shape" specifies the tensor's shape and the numpy array
    can not have more elements than what "shape" specifies.

    """
    if isinstance(values, tensor_pb2.TensorProto):
        return values

    if dtype:
        dtype = dtypes.as_dtype(dtype)

    is_quantized = (
            dtype in [
        dtypes.qint8, dtypes.quint8, dtypes.qint16, dtypes.quint16,
        dtypes.qint32
    ])

    # We first convert value to a numpy array or scalar.
    if isinstance(values, (np.ndarray, np.generic)):
        if dtype:
            nparray = values.astype(dtype.as_numpy_dtype)
        else:
            nparray = values
    elif callable(getattr(values, "__array__", None)) or isinstance(
            getattr(values, "__array_interface__", None), dict):
        # If a class has the __array__ method, or __array_interface__ dict, then it
        # is possible to convert to numpy array.
        nparray = np.asarray(values, dtype=dtype)

        # This is the preferred way to create an array from the object, so replace
        # the `values` with the array so that _FlattenToStrings is not run.
        values = nparray
    else:
        if values is None:
            raise ValueError("None values not supported.")
        # if dtype is provided, forces numpy array to be the type
        # provided if possible.
        if dtype and dtype.is_numpy_compatible:
            np_dt = dtype.as_numpy_dtype
        else:
            np_dt = None
        # If shape is None, numpy.prod returns None when dtype is not set, but raises
        # exception when dtype is set to np.int64
        if shape is not None and np.prod(shape, dtype=np.int64) == 0:
            nparray = np.empty(shape, dtype=np_dt)
        else:
            _AssertCompatible(values, dtype)
            nparray = np.array(values, dtype=np_dt)
            # check to them.
            # We need to pass in quantized values as tuples, so don't apply the shape
            if (list(nparray.shape) != _GetDenseDimensions(values) and
                    not is_quantized):
                raise ValueError("""Argument must be a dense tensor: %s"""
                                 """ - got shape %s, but wanted %s.""" %
                                 (values, list(nparray.shape),
                                  _GetDenseDimensions(values)))

        # python/numpy default float type is float64. We prefer float32 instead.
        if (nparray.dtype == np.float64) and dtype is None:
            nparray = nparray.astype(np.float32)
        # python/numpy default int type is int64. We prefer int32 instead.
        elif (nparray.dtype == np.int64) and dtype is None:
            downcasted_array = nparray.astype(np.int32)
            # Do not down cast if it leads to precision loss.
            if np.array_equal(downcasted_array, nparray):
                nparray = downcasted_array

    # if dtype is provided, it must be compatible with what numpy
    # conversion says.
    numpy_dtype = dtypes.as_dtype(nparray.dtype)
    if numpy_dtype is None:
        raise TypeError("Unrecognized data type: %s" % nparray.dtype)

    # If dtype was specified and is a quantized type, we convert
    # numpy_dtype back into the quantized version.
    if is_quantized:
        numpy_dtype = dtype

    if dtype is not None and (not hasattr(dtype, "base_dtype") or
                              dtype.base_dtype != numpy_dtype.base_dtype):
        raise TypeError("Incompatible types: %s vs. %s. Value is %s" %
                        (dtype, nparray.dtype, values))

    # If shape is not given, get the shape from the numpy array.
    if shape is None:
        shape = nparray.shape
        is_same_size = True
        shape_size = nparray.size
    # note: removed else branch since we also deal with ? dimensions

    tensor_proto = tensor_pb2.TensorProto(
        dtype=numpy_dtype.as_datatype_enum,
        tensor_shape=tensor_shape.as_shape(shape).as_proto())

    # If we were not given values as a numpy array, compute the proto_values
    # from the given values directly, to avoid numpy trimming nulls from the
    # strings. Since values could be a list of strings, or a multi-dimensional
    # list of lists that might or might not correspond to the given shape,
    # we flatten it conservatively.
    if numpy_dtype == dtypes.string and not isinstance(values, np.ndarray):
        proto_values = _FlattenToStrings(values)

        # At this point, values may be a list of objects that we could not
        # identify a common type for (hence it was inferred as
        # np.object/dtypes.string).  If we are unable to convert it to a
        # string, we raise a more helpful error message.
        #
        # Ideally, we'd be able to convert the elements of the list to a
        # common type, but this type inference requires some thinking and
        # so we defer it for now.
        try:
            str_values = [compat.as_bytes(x) for x in proto_values]
        except TypeError:
            raise TypeError("Failed to convert object of type %s to Tensor. "
                            "Contents: %s. Consider casting elements to a "
                            "supported type." % (type(values), values))
        tensor_proto.string_val.extend(str_values)
        return tensor_proto

    # TensorFlow expects C order (a.k.a., eigen row major).
    proto_values = nparray.ravel()

    append_fn = GetNumpyAppendFn(proto_values.dtype)
    if append_fn is None:
        raise TypeError(
            "Element type not supported in TensorProto: %s" % numpy_dtype.name)
    append_fn(tensor_proto, proto_values)

    return tensor_proto


def fixed_make_ndarray(tensor):
    """Create a numpy ndarray from a tensor.

    Create a numpy ndarray with the same shape and data as the tensor.

    Args:
      tensor: A TensorProto.

    Returns:
      A numpy array with the tensor contents.

    Raises:
      TypeError: if tensor has unsupported type.

    """
    shape = [d.size for d in tensor.tensor_shape.dim]
    num_elements = np.prod(shape, dtype=np.int64)
    tensor_dtype = dtypes.as_dtype(tensor.dtype)
    dtype = tensor_dtype.as_numpy_dtype

    # tensor_content
    if tensor.tensor_content:
        return np.fromstring(tensor.tensor_content, dtype=dtype).reshape(shape)
    # half_val
    elif tensor_dtype == dtypes.float16:
        # the half_val field of the TensorProto stores the binary representation
        # of the fp16: we need to reinterpret this as a proper float16
        if len(tensor.half_val) == 1:
            tmp = np.array(tensor.half_val[0], dtype=np.uint16)
            tmp.dtype = np.float16
            return np.repeat(tmp, num_elements).reshape(shape)
        else:
            tmp = np.fromiter(tensor.half_val, dtype=np.uint16)
            tmp.dtype = np.float16
            return tmp.reshape(shape)
    # float_val
    elif tensor_dtype == dtypes.float32:
        if len(tensor.float_val) == 1:
            return np.repeat(
                np.array(tensor.float_val[0], dtype=dtype),
                num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.float_val, dtype=dtype).reshape(shape)
    # double_val
    elif tensor_dtype == dtypes.float64:
        if len(tensor.double_val) == 1:
            return np.repeat(
                np.array(tensor.double_val[0], dtype=dtype),
                num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.double_val, dtype=dtype).reshape(shape)
    # int_val
    elif tensor_dtype in [
        dtypes.int32, dtypes.uint8, dtypes.uint16, dtypes.int16, dtypes.int8,
        dtypes.qint32, dtypes.quint8, dtypes.qint8, dtypes.qint16, dtypes.quint16,
        dtypes.bfloat16
    ]:
        if len(tensor.int_val) == 1:
            return np.repeat(np.array(tensor.int_val[0], dtype=dtype),
                             num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.int_val, dtype=dtype).reshape(shape)
    # int64_val
    elif tensor_dtype == dtypes.int64:
        if len(tensor.int64_val) == 1:
            return np.repeat(
                np.array(tensor.int64_val[0], dtype=dtype),
                num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.int64_val, dtype=dtype).reshape(shape)
    # uint32_val
    elif tensor_dtype == dtypes.uint32:
        if len(tensor.uint32_val) == 1:
            return np.repeat(
                np.array(tensor.uint32_val[0], dtype=dtype),
                num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.uint32_val, dtype=dtype).reshape(shape)
    # uint64_val
    elif tensor_dtype == dtypes.uint64:
        if len(tensor.uint64_val) == 1:
            return np.repeat(
                np.array(tensor.uint64_val[0], dtype=dtype),
                num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.uint64_val, dtype=dtype).reshape(shape)
    # string_val
    elif tensor_dtype == dtypes.string:
        if len(tensor.string_val) == 1:
            return np.repeat(
                np.array(tensor.string_val[0], dtype=dtype),
                num_elements).reshape(shape)
        else:
            return np.array(
                [x for x in tensor.string_val], dtype=dtype).reshape(shape)
    # scomplex_val
    elif tensor_dtype == dtypes.complex64:
        it = iter(tensor.scomplex_val)
        if len(tensor.scomplex_val) == 2:
            return np.repeat(
                np.array(
                    complex(tensor.scomplex_val[0], tensor.scomplex_val[1]),
                    dtype=dtype), num_elements).reshape(shape)
        else:
            return np.array(
                [complex(x[0], x[1]) for x in zip(it, it)],
                dtype=dtype).reshape(shape)
    # dcomplex_val
    elif tensor_dtype == dtypes.complex128:
        it = iter(tensor.dcomplex_val)
        if len(tensor.dcomplex_val) == 2:
            return np.repeat(
                np.array(
                    complex(tensor.dcomplex_val[0], tensor.dcomplex_val[1]),
                    dtype=dtype), num_elements).reshape(shape)
        else:
            return np.array(
                [complex(x[0], x[1]) for x in zip(it, it)],
                dtype=dtype).reshape(shape)
    # bool_val
    elif tensor_dtype == dtypes.bool:
        if len(tensor.bool_val) == 1:
            return np.repeat(np.array(tensor.bool_val[0], dtype=dtype),
                             num_elements).reshape(shape)
        else:
            return np.fromiter(tensor.bool_val, dtype=dtype).reshape(shape)
    else:
        raise TypeError("Unsupported tensor type: %s" % tensor.dtype)