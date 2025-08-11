import torch
class KDEEmbedding:
    def __init__(self,  xmin=0.0, xmax=1.0, width=0.01, deg=100, decode_res=512) -> None:
        self.xmin = xmin
        self.xmax = xmax
        self.deg = deg
        self.shifts = torch.arange(xmin, xmax, step=(xmax-xmin)/deg)
        self.embed_fns = [
            lambda x, s=s: self.kernel((x-s) / width) / width for s in self.shifts
        ]
        self.decode_res = decode_res

    def kernel(self, x):
        return (-0.5 * x.pow(2)).exp() / (2.0 * torch.pi) ** 0.5

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


    def get_estimator(self, coefs):
        return lambda x: sum([coefs[i] * fn(x) for i, fn in enumerate(self.embed_fns)])
    
    def decode(self, feats):
        x = torch.linspace(self.xmin, self.xmax, self.decode_res).to("cuda")
        return sum([feats[:, i:i+1] * fn(x)[None, :, None, None] for i, fn in enumerate(self.embed_fns)])