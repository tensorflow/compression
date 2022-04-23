import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
class Circle(tfp.distributions.Distribution):
    """The "circle": [cos x, sin x] where x is uniform over [0,2pi]"""
    def __init__(self,width=0.,dtype=tf.float32,validate_args=False,allow_nan_stats=True,name='circle'):
        """
        Args: 
        width: Float in [0,1]. Allows for realizations to be uniformly distributed in a band between radius 1-width and 1+width. 
        dtype: Data type of returned realization. Defaults to tf.float32.
        validate_args: required by tensoflow Distribution class but unused.
        allow_nan_stats: required by tensorflow Distribution class but unused.
        name: String. Name of the created object.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            self._width=width
        super().__init__(dtype=dtype,
                reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
                validate_args = validate_args,
                allow_nan_stats = allow_nan_stats,
                parameters = parameters,
                name=name)
    @property
    def width(self):
        return self._width

    def _batch_shape_tensor(self):
        return tf.constant([],dtype=tf.int32)
    def _batch_shape(self):
        return tf.TensorShape([])
    def _event_shape(self):
        smp = tf.linspace(0.,1.,2)
        return smp.shape
    def _event_shape_tensor(self):
        smp = tf.linspace(0.,1.,2)
        return tf.shape(smp)
    def _sample_n(self,n,seed=None):
        angles = tf.random.uniform((1,n),minval = 0, maxval = 2*tf.constant(np.pi),seed=seed,dtype=self.dtype)
        sample = tf.convert_to_tensor([tf.math.cos(angles),tf.math.sin(angles)])
        if self.width>0:
            dev = tf.random.uniform((1,n),minval = -(self.width)/2.0, maxval = (self.width)/2.0, seed=seed, dtype=self.dtype)
            sample = sample + dev*sample
        sample = tf.transpose(sample,[2,1,0])
        sample = tf.reshape(sample,[n,2])
        return sample

    def _prob(self, x):
        new_x  = tf.reshape(x,[-1,tf.shape(x)[-1]])
        flat_norm = tf.norm(new_x,axis=1)
        if self.width>0:
            prob = tf.where((flat_norm >= 1-self.width)&(flat_norm <= 1+self.width),
                    1.0/(2*tf.convert_to_tensor(np.pi)*self.width),
                    0.
                    )
        else:
            prob = tf.where(flat_norm==1,
                1.0/(2*tf.convert_to_tensor(np.pi)),
                0.
                )
        return tf.reshape(prob,tf.shape(x)[:-1])
