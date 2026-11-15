using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
namespace WordGenerator
{
    public sealed class CausalTransformerBlock : Module<Tensor, Tensor>
    {
        private readonly int d;
        private readonly Linear qLayer, kLayer, vLayer, attentionOutput;
        private readonly Linear ff1, ff2;
        private readonly LayerNorm norm1, norm2;
        public CausalTransformerBlock(int d, int ffSize) : base("CausalTransformerBlock")
        {
            this.d = d;
            qLayer = Linear(d, d);
            kLayer = Linear(d, d);
            vLayer = Linear(d, d);
            attentionOutput = Linear(d, d);
            norm1 = LayerNorm(new long[] { d });
            norm2 = LayerNorm(new long[] { d });
            ff1 = Linear(d, ffSize);
            ff2 = Linear(ffSize, d);
            RegisterComponents();
        }
        public override Tensor forward(Tensor input)
        {
            using var scope = torch.NewDisposeScope();
            long t = input.shape[1];
            var normalized = norm1.forward(input);
            var q = qLayer.forward(normalized);
            var k = kLayer.forward(normalized);
            var v = vLayer.forward(normalized);
            // scores[b,i,j] measures how strongly position i attends to position j.
            var scores = q.matmul(k.transpose(1, 2)) / (float)Math.Sqrt(d);
            // Causal mask: position i can only see positions j <= i.
            var mask = torch.tril(torch.ones(new long[] { t, t },
                dtype: ScalarType.Bool, device: input.device));
            var weights = scores.masked_fill(mask.logical_not(), float.NegativeInfinity)
                                .softmax(-1);
            var attended = weights.matmul(v);
            var x = input + attentionOutput.forward(attended);
            var hidden = ff1.forward(norm2.forward(x)).relu();
            var y = x + ff2.forward(hidden);
            return y.MoveToOuterDisposeScope();
        }
    }
    public sealed class CharacterTransformer : Module<Tensor, Tensor>
    {
        private readonly int maxContext;
        private readonly Embedding tokenEmbedding;
        private readonly Embedding positionEmbedding;
        private readonly CausalTransformerBlock block1, block2;
        private readonly LayerNorm finalNorm;
        private readonly Linear output;
        public CharacterTransformer(int maxContext, int d = 24, int ffSize = 48)
            : base("CharacterTransformer")
        {
            this.maxContext = maxContext;
            tokenEmbedding = Embedding(WordEncoding.VocabularySize, d);
            positionEmbedding = Embedding(maxContext, d);
            block1 = new CausalTransformerBlock(d, ffSize);
            block2 = new CausalTransformerBlock(d, ffSize);
            finalNorm = LayerNorm(new long[] { d });
            output = Linear(d, WordEncoding.VocabularySize);
            RegisterComponents();
        }
        // Input: [batch, time] Int64 token IDs.
        // Output: [batch, time, vocabulary] raw, unnormalized logits.
        public override Tensor forward(Tensor tokens)
        {
            if (tokens.shape.Length != 2 || tokens.shape[1] > maxContext)
                throw new ArgumentException("Expected [batch,time] with time <= maxContext.");
            using var scope = torch.NewDisposeScope();
            long t = tokens.shape[1];
            var positions = torch.arange(0, t, dtype: ScalarType.Int64,
                device: tokens.device);
            var x = tokenEmbedding.forward(tokens)
                  + positionEmbedding.forward(positions).unsqueeze(0);
            x = block1.forward(x);
            x = block2.forward(x);
            x = finalNorm.forward(x);
            return output.forward(x).MoveToOuterDisposeScope();
        }
    }
}
