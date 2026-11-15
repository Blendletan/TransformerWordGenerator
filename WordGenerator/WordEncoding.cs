using System;
using System.Collections.Generic;
using System.Text;
namespace WordGenerator
{
    public static class WordEncoding
    {
        public const long Pad = 0;
        public const long Start = 1;
        public const long End = 2;
        public const int VocabularySize = 29; // PAD, START, END, A..Z
        public const long IgnoreTarget = -100;
        public static long Letter(char c) => 3 + (c - 'a');
        public static char Decode(long id) => (char)('a' + id - 3);
    }
    public sealed record WordExample(long[] Input, long[] Target, int NonPaddingTargets);
}
