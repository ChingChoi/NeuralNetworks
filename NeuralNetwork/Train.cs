using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Train<T>
    {
        public struct DataStruct
        {
            public double[][] data;
            public T[] labels;

            public DataStruct(double[][] data, T[] labels)
            {
                this.data = data;
                this.labels = labels;
            }
        }

        /// <summary>
        /// Train a network using given data, label, and batchSize
        /// </summary>
        /// <param name="nn">neural network</param>
        /// <param name="data">training data</param>
        /// <param name="validset">validset data</param>
        /// <param name="labels">data's label</param>
        /// <param name="validsetLabels">validset labels</param>
        /// <param name="epochs">number of epochs</param>
        /// <param name="batchSize">Batch size</param>
        public static void trainNetwork(NeuralNetwork<T> nn, 
            double[][] data, double[][] validset, T[] labels, T[] validsetLabels, int epochs, int batchSize, NeuralNetwork.CustomTextBox aiTextBox)
        {
            for (int e = 0; e < epochs; e++)
            {
                DataStruct dataS = ArrayTransform.Shuffle<T>(data, labels);
                int iteration = data.Length / batchSize;
                for (int i = 0; i < 100; i++)
                {
                    nn.BackPropInit(batchSize);
                    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
                    {
                        nn.InitdCda();
                        nn.getInputs(dataS.data[batchIndex + i * batchSize], dataS.labels[batchIndex + i * batchSize]);
                        nn.calcOutput();
                        nn.calculateCost(dataS.labels[batchIndex + i * batchSize]);
                        nn.calcGradientVector();
                    }
                    nn.BackPropApplication();
                }
                trainResult(nn, validset, validsetLabels, e, validset.Length, aiTextBox);
            }
        }

        /// <summary>
        /// Print out epoch sessions
        /// </summary>
        /// <param name="nn">neural network</param>
        /// <param name="data">whole input data</param>
        /// <param name="labels">whole labels</param>
        /// <param name="epochs">number of epochs</param>
        /// <param name="curEpoch">current epoch</param>
        /// <param name="index">current data index</param>
        public static void trainResult(NeuralNetwork<T> nn,
            double[][] data, T[] labels, int curEpoch, int batchSize, NeuralNetwork.CustomTextBox aiTextBox)
        {
            int success = 0;
            for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
            {
                if (labels[batchIndex].Equals(nn.determine(data[batchIndex])))
                {
                    success++;
                }
            }
            // edge case to be implemented
            Console.WriteLine("Epoch " + curEpoch + ": " + success + " / " + batchSize);
            if (aiTextBox.Lines.Length == 6)
            {
                NeuralNetwork.CustomTextBox temp = new NeuralNetwork.CustomTextBox();
                for (int i = 1; i < aiTextBox.Lines.Length; i++)
                {
                    temp.Text += aiTextBox.Lines[i] + "\n";
                }
                temp.Text += "Epoch " + curEpoch + ": " + success + " / " + batchSize;
                aiTextBox.Invoke(new Action(() =>
                {
                    aiTextBox.Text = temp.Text;
                }));
            }
            else
            {
                aiTextBox.Invoke(new Action(() =>
                {
                    aiTextBox.Text += "\nEpoch " + curEpoch + ": " + success + " / " + batchSize;
                }));
            }
        }
    }
}
