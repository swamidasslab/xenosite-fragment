from keras import layers
from keras import Model
from .feature import FragmentVectorize, FragmentVectors, FragmentVectorSizes
import tensorflow as tf

class FragmentNetwork(Model):
    def __init__(self, sizes: FragmentVectorSizes, dropout: float = 0.3):
        super().__init__()

        #self.null_attention = tf.Variable(initial_value=tf.ones((1,)), trainable=False) 
        self.bias = tf.Variable(initial_value=tf.zeros((1,)), trainable=True) 

        self.dropout = layers.Dropout(dropout)

        self.frag_embedding = layers.Embedding(sizes.frag, 1, name="frag_embedding")
        self.site_embedding = layers.Embedding(sizes.frag_site, 1, name="site_embedding")

    def call(self, vectors : tf.RaggedTensor):
              
        frag = vectors[...,1]
        frag = self.frag_embedding(frag)[...,0] #type: ignore

        site = vectors[...,0]  
        site = self.site_embedding(site)[...,0] #type: ignore
        
        attn = tf.exp(frag)
        attn = self.dropout(attn)
        sum_attn = tf.reduce_sum(attn, axis=-1) + 0.001 # + self.null_attention
        sum_logit_attnw = tf.reduce_sum(attn * site, axis=-1)   # self.null_attention 

        return sum_logit_attnw / sum_attn + self.bias

    



        