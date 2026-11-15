using System;
using System.Collections.Generic;
using System.Text;
namespace Frontend
{
    public static class TwoNN
    {
        public static double Estimate(IReadOnlyList<float[]> points)
        {
            int n = points.Count;
            if (n < 3)
                throw new ArgumentException("Need at least 3 points.");
            int dim = points[0].Length;
            double sumLogRatios = 0;
            int valid = 0;
            int duplicates = 0;
            for (int i = 0; i < n; i++)
            {
                double first = double.PositiveInfinity;
                double second = double.PositiveInfinity;
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    double distanceSquared = 0;
                    for (int k = 0; k < dim; k++)
                    {
                        double delta = points[i][k] - points[j][k];
                        distanceSquared += delta * delta;
                    }
                    if (distanceSquared < first)
                    {
                        second = first;
                        first = distanceSquared;
                    }
                    else if (distanceSquared < second)
                    {
                        second = distanceSquared;
                    }
                }
                // Exact duplicate representations are not valid
                // for the standard TwoNN calculation.
                if (first == 0)
                {
                    duplicates++;
                    continue;
                }
                if (!double.IsFinite(second))
                    continue;
                // Distances were squared, hence the factor 1/2.
                double logRatio = 0.5 * Math.Log(second / first);
                sumLogRatios += logRatio;
                valid++;
            }
            Console.WriteLine($"TwoNN: {valid} usable, {duplicates} duplicates");
            if (valid < 3 || sumLogRatios <= 0)
                throw new InvalidOperationException("Cannot estimate dimension.");
            return valid / sumLogRatios;
        }
    }
}
