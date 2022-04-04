from einops import rearrange, reduce

import jax.numpy as jnp
import haiku as hk

class IntEncoder(hk.Module):
    def __init__(self, n_bits, embed_dim, name="int_encoder"):
        super().__init__(name=name)
        self.bits = hk.Embed(vocab_size=n_bits, embed_dim=embed_dim)
        self.n_bits = n_bits
        self.embed_dim = embed_dim
        self.pows_two = jnp.array([1<<x for x in range(n_bits)])
        
    def __call__(self, nums):
        # b: batch size, n: num bits, h: hidden dim
        
        x = (jnp.bitwise_and(
            rearrange(nums, '... b -> ... b () ()'), 
            rearrange(self.pows_two, 'n -> () n ()')) > 0).astype(jnp.float32)
        
        # bitvecs : n h
        bitvecs = rearrange(self.bits(jnp.arange(0, self.n_bits)), 'n h -> () n h')
        
        return reduce(x * bitvecs, '... b n h -> ... b h', 'sum')