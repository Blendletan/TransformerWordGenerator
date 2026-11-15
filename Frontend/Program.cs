using System.Diagnostics;
using TorchSharp;
using WordGenerator;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim.lr_scheduler.impl.CyclicLR;
namespace Frontend
{
    internal class Program
    {
        static void Shuffle<T>(T[] a, Random rng)
        {
            for (int i = a.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (a[i], a[j]) = (a[j], a[i]);
            }
        }
        static (Tensor X, Tensor Y) MakeBatch(WordExample[] examples, int context, Device device)
        {
            var x = new long[examples.Length, context];
            var y = new long[examples.Length, context];
            for (int i = 0; i < examples.Length; i++)
            {
                for (int t = 0; t < context; t++)
                {
                    x[i, t] = examples[i].Input[t];
                    y[i, t] = examples[i].Target[t];
                }
            }
            return (torch.tensor(x).to(device), torch.tensor(y).to(device));
        }
        static double Evaluate(CharacterTransformer model, Loss<Tensor, Tensor, Tensor> lossFunction,
            WordExample[] examples, int context, int batchSize, Device device)
        {
            model.eval();
            using var noGrad = torch.no_grad();
            double total = 0;
            long tokenCount = 0;
            for (int start = 0; start < examples.Length; start += batchSize)
            {
                using var scope = torch.NewDisposeScope();
                var batch = examples.Skip(start).Take(batchSize).ToArray();
                var (x, y) = MakeBatch(batch, context, device);
                var logits = model.forward(x);
                var loss = lossFunction.call(
                    logits.reshape(-1, WordEncoding.VocabularySize),
                    y.reshape(-1));
                long tokens = batch.Sum(b => b.NonPaddingTargets);
                total += loss.ToSingle() * tokens;
                tokenCount += tokens;
            }
            return total / tokenCount;
        }
        static void PrintGenerated(CharacterTransformer model, Device device, int maxLetters, Random rng, int n)
        {
            model.eval();
            using var noGrad = torch.no_grad();
            Console.WriteLine("Generated: " + string.Join(", ",
                Enumerable.Range(0, n).Select(_ => SampleWord(model, device, maxLetters, rng, temperature: 0.9))));
        }
        static string SampleWord(CharacterTransformer model, Device device,int maxLetters, Random rng, double temperature)
        {
            var ids = new List<long> { WordEncoding.Start };
            for (int i = 0; i < maxLetters; i++)
            {
                using var scope = torch.NewDisposeScope();
                var x = new long[1, ids.Count];
                for (int j = 0; j < ids.Count; j++) x[0, j] = ids[j];
                var tokenTensor = torch.tensor(x).to(device);
                var logits = model.forward(tokenTensor);
                var last = logits.select(1, ids.Count - 1).squeeze(0);
                var scores = (last / temperature).softmax(-1).to(torch.CPU).data<float>().ToArray();
                long id = SampleAllowed(scores, rng);
                if (id == WordEncoding.End) break;
                ids.Add(id);
            }
            return new string(ids.Skip(1).Select(WordEncoding.Decode).ToArray());
        }
        static long SampleAllowed(float[] probabilities, Random rng)
        {
            // Never generate PAD or START. Only END and A..Z are allowed.
            double total = 0;
            for (int i = 2; i < probabilities.Length; ++i) total += probabilities[i];
            double draw = rng.NextDouble() * total;
            for (int i = 2; i < probabilities.Length; ++i)
            {
                draw -= probabilities[i];
                if (draw <= 0) return i;
            }
            return probabilities.Length - 1;
        }
        static void PrintNextLetterProbabilities(CharacterTransformer model,
            Device device, string prefix)
        {
            using var noGrad = torch.no_grad();
            using var scope = torch.NewDisposeScope();
            var ids = new long[1, prefix.Length + 1];
            ids[0, 0] = WordEncoding.Start;
            for (int i = 0; i < prefix.Length; i++)
                ids[0, i + 1] = WordEncoding.Letter(prefix[i]);
            var x = torch.tensor(ids).to(device);
            var logits = model.forward(x).select(1, prefix.Length).squeeze(0);
            var probabilities = logits.softmax(-1).to(torch.CPU).data<float>().ToArray();
            var highest = Enumerable.Range(2, 27)
                .OrderByDescending(i => probabilities[i]).Take(8);
            Console.WriteLine($"Next character after '{prefix}': " + string.Join(", ",
                highest.Select(id => $"{(id == 2 ? "<END>" : WordEncoding.Decode(id).ToString())}: {probabilities[id]:P1}")));
        }
        static void Train()
        {
            const string filePath = "words.txt";
            int maxLetters = 22;
            const int epochs = 10;
            var words = WordData.ReadWords(filePath, maxLetters);
            maxLetters = words.OrderByDescending(w => w.Length).First().Length;
            var (train, val) = WordData.Split(words, maxLetters, 4);
            const int batchSize = 64;
            const int d = 24;
            const int ffSize = 48;
            const int seed = 12345;
            const double learningRate = 0.0005;
            string checkpointPath = $"best-word-model-{maxLetters}.dat";
            Console.WriteLine($"Words: {words.Length} | Training: {train.Length} | Validation: {val.Length}");
            Console.WriteLine($"Model: D={d}, one attention head, 2 blocks, FF={ffSize}");
            Console.WriteLine($"Vocabulary: {WordEncoding.VocabularySize} | Context: {maxLetters + 1}");
            torch.manual_seed(seed);
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine($"Device: {device}");
            using var model = new CharacterTransformer(maxLetters + 1, d, ffSize);
            model.to(device);
            using var lossFunction = CrossEntropyLoss(ignore_index: -100);
            using var optimizer = torch.optim.Adam(model.parameters(), lr: learningRate);
            var rng = new Random(seed);
            double bestVal = double.PositiveInfinity;
            int bestEpoch = -1;
            var sw = new Stopwatch();
            sw.Start();
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                model.train();
                // Shuffle batches but preserve the whole-word split.
                var order = Enumerable.Range(0, train.Length).ToArray();
                Shuffle(order, rng);
                double weightedTrainLoss = 0;
                long trainingTokens = 0;
                for (int offset = 0; offset < order.Length; offset += batchSize)
                {
                    using var scope = torch.NewDisposeScope();
                    int size = Math.Min(batchSize, order.Length - offset);
                    var batch = new WordExample[size];
                    for (int i = 0; i < size; i++) batch[i] = train[order[offset + i]];
                    var (x, y) = MakeBatch(batch, maxLetters + 1, device);
                    optimizer.zero_grad();
                    var logits = model.forward(x);
                    var loss = lossFunction.call(
                        logits.reshape(-1, WordEncoding.VocabularySize),
                        y.reshape(-1));
                    loss.backward();
                    optimizer.step();
                    long tokens = batch.Sum(b => b.NonPaddingTargets);
                    weightedTrainLoss += loss.ToSingle() * tokens;
                    trainingTokens += tokens;
                }
                double trainLoss = weightedTrainLoss / trainingTokens;
                Console.WriteLine($"Epoch {epoch} training loss was {trainLoss}");
                double valLoss = Evaluate(model, lossFunction, val, maxLetters + 1, batchSize, device);
                if (valLoss < bestVal)
                {
                    bestVal = valLoss;
                    bestEpoch = epoch;
                    model.save(checkpointPath);
                }
                Console.WriteLine($"Epoch {epoch,3}: train CE={trainLoss:F4}," +
                    $"validation CE={valLoss:F4}, val perplexity={Math.Exp(valLoss):F2}, " +
                    $"best={bestVal:F4} ({bestEpoch})");
                if (epoch == 1 || epoch % 5 == 0)
                {
                    PrintGenerated(model, device, maxLetters, rng, 8);
                }
            }
            sw.Stop();
            model.load(checkpointPath);
            model.eval();
            Console.WriteLine($"\nRestored epoch {bestEpoch}, validation CE {bestVal:F4}");
            PrintGenerated(model, device, maxLetters, rng, 30);
            PrintNextLetterProbabilities(model, device, "trans");
            string formattedTimeElapsed = sw.Elapsed.ToString(@"dd\.hh\:mm\:ss");
            Console.WriteLine($"Training took {formattedTimeElapsed} (dd.hh:mm:ss)");
        }
        static void Main(string[] args)
        {
            Train();
        }
    }
}
