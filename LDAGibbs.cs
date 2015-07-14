using CPAIS.Updater;
using System;

namespace CPAIS.Alglorithm
{
    internal class LDAGibbs
    {
        public class Params
        {
            public int[][] docs = null;
            public int K = 0;
            public int V = 0;
            public double alpha = 0;
            public double beta = 0;
            public int iterations = 0;
        }

        private readonly int K, M, V;
        private readonly double alpha, beta;
        private readonly int iterations;
        private readonly int[][] docs;
        private double[,] theta;
        private double[,] phi;
        private Random rand;

        public LDAGibbs(Params param)
        {
            this.K = param.K;
            this.M = param.docs.Length;
            this.V = param.V;
            this.alpha = param.alpha;
            this.beta = param.beta;
            this.iterations = param.iterations;
            this.docs = param.docs;
            this.theta = new double[this.M, this.K];
            this.phi = new double[this.K, this.V];

            this.rand = new Random();
        }

        /// <summary>
        /// Gets Document-Topic association.
        /// </summary>
        public double[,] Theta
        {
            get { return theta; }
        }

        /// <summary>
        /// Gets Topic-Word association.
        /// </summary>
        public double[,] Phi
        {
            get { return phi; }
        }

        /// <summary>
        /// Gibbs sampling iterations.
        /// </summary>
        public void GibbsSampling(ProgressUpdater updater)
        {
            #region Local variables

            // Count statistics and their sums
            int[,] nmk = new int[M, K];
            int[] nm = new int[M];
            int[,] nkv = new int[K, V];
            int[] nk = new int[K];

            // Memory for full conditional array
            int[][] zassign = new int[M][];

            #endregion

            #region Initialization

            if (updater != null)
                updater.UpdateMessage("Initializing...");
            
            for (int m = 0; m < M; m++)
            {
                int N = docs[m].Length;
                zassign[m] = new int[N];
                for (int n = 0; n < N; n++)
                {
                    int z = rand.Next(0, K);
                    nmk[m, z]++;
                    int v = docs[m][n];
                    nkv[z, v]++;
                    nk[z]++;
                    zassign[m][n] = z;
                }
                nm[m] = N;
            }

            #endregion

            #region Gibbs sampling

            if (updater != null)
                updater.UpdateMessage("Gibbs sampling...");

            for (int itr = 0; itr < iterations; itr++)
            {
                for (int m = 0; m < M; m++)
                {
                    int N = docs[m].Length;
                    for (int n = 0; n < N; n++)
                    {
                        // For the current topic assignment z to word token docs[m][n]
                        int z = zassign[m][n];
                        int v = docs[m][n];
                        nmk[m, z]--;
                        nm[m]--;
                        nkv[z, v]--;
                        nk[z]--;

                        // For the new topic assignment z to the word token docs[m][n]
                        z = SimpleZ(nkv, nk, nmk, nm, m, n);
                        nm[m]++;
                        nmk[m, z]++;
                        nk[z]++;
                        nkv[z, v]++;

                        zassign[m][n] = z;
                    }
                }
                if (updater != null)
                    updater.UpdateProgress((double)(itr + 1) / iterations);                
            }

            #endregion

            #region Calculate result, theta and phi

            if (updater != null)
                updater.UpdateMessage("Calcaulating result...");

            for (int k = 0; k < K; k++)
            {
                for (int v = 0; v < V; v++)
                {
                    phi[k, v] = (nkv[k, v] + beta) / (nk[k] + V * beta);
                }
                for (int m = 0; m < M; m++)
                {
                    theta[m, k] = (nmk[m, k] + alpha) / (nm[m] + K * alpha);
                }
            }

            #endregion
        }

        public double CalcPerplexity()
        {
            double dP = 0.0;
            int num = 0;
            for (int m = 0; m < M; m++)
            {
                double dTrace = 1.0;
                int N = docs[m].Length;
                for (int n = 0; n < N; n++)
                {
                    double dCur = 0.0;
                    for (int k = 0; k < K; k++)
                        dCur += phi[k, docs[m][n]] * theta[m, k];
                    dTrace *= dCur;
                }
                if (dTrace != 0.0)
                {
                    dP += Math.Log(dTrace);
                    num += docs[m].Length;
                }
            }

            return Math.Exp(-(dP / num));
        }
        public double CalcKLDivergence()
        {
            int nCount = 0;
            double dRet = 0.0;
            for (int m = 0; m < M; m++)
            {
                for (int q = m + 1; q < M; q++)
                {
                    nCount++;
                    for (int k = 0; k < K; k++)
                    {
                        double theta1 = theta[m, k];
                        double theta2 = theta[q, k];
                        double dk = theta1 * Math.Log(theta1 / theta2) +
                            theta2 * Math.Log(theta2 / theta1);
                        dRet += dk;
                    }
                }
            }
            return dRet / nCount;
        }
        public double CalcEntropy()
        {
            double dRet = 0.0;
            for (int k = 0; k < K; k++)
            {
                for (int v = 0; v < V; v++)
                {
                    dRet += phi[k, v] * Math.Log(phi[k, v]);
                }
            }
            return -dRet;
        }

        private int SimpleZ(int[,] nkv, int[] nk, int[,] nmk, int[] nm, int m, int n)
        {
            double[] p = new double[K];
            for (int k = 0; k < K; k++)
            {
                int v = docs[m][n];
                p[k] = ((nkv[k, v] + beta) / (nk[k] + V * beta)) *
                    ((nmk[m, k] + alpha) / (nm[m] + K * alpha));
            }

            // Calculate multinomial parameters
            for (int k = 1; k < K; k++)
                p[k] += p[k - 1];

            // Scaled sample because of unnormalized p[]
            double u = rand.NextDouble() * p[K - 1];
            int z = 0;
            for (; z < K; z++)
            {
                if (p[z] > u)
                    break;
            }
            return z;
        }
    }
}
