using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
namespace WordGenerator
{
    public static class WordData
    {
        // One lowercase ASCII word per line, no frequency weighting or special obscure-word filtering.
        public static string[] ReadWords(string path, int maxLetters)
        {
            if (File.Exists(path)==false)
            {
                throw new FileNotFoundException("Supply a word list: one word per line.", path);
            }
            var accepted = new HashSet<string>(StringComparer.Ordinal);
            foreach (string raw in File.ReadLines(path))
            {
                string word = raw.Trim().ToLowerInvariant();
                if (word.Length < 2 || word.Length > maxLetters)
                {
                    Console.WriteLine($"Rejected word {word} since it had {word.Length} letters");
                    continue;
                }
                if (Regex.IsMatch(word, "^[a-z]+$")==false)
                {
                    continue;
                }
                accepted.Add(word);
            }
            return accepted.OrderBy(w => w, StringComparer.Ordinal).ToArray();
        }
        public static WordExample EncodeWord(string word, int maxLetters)
        {
            int context = maxLetters + 1;
            var input = new long[context];
            var target = Enumerable.Repeat(WordEncoding.IgnoreTarget, context).ToArray();
            input[0] = WordEncoding.Start;
            for (int i = 0; i < word.Length; ++i)
            {
                input[i + 1] = WordEncoding.Letter(word[i]);
                target[i] = WordEncoding.Letter(word[i]);
            }
            target[word.Length] = WordEncoding.End;
            return new WordExample(input, target, word.Length + 1);
        }
        // Partition whole words, not individual next-character examples.
        public static (WordExample[] Train, WordExample[] Val) Split(string[] words, int maxLetters, int seed)
        {
            var shuffled = words.ToArray();
            var rng = new Random(seed);
            for (int i = shuffled.Length - 1; i > 0; --i)
            {
                int j = rng.Next(i + 1);
                (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
            }
            int cut = (int)(0.9 * shuffled.Length);
            return (
                shuffled.Take(cut).Select(w => EncodeWord(w, maxLetters)).ToArray(),
                shuffled.Skip(cut).Select(w => EncodeWord(w, maxLetters)).ToArray()
            );
        }
    }
}
