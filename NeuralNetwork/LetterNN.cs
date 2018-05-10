using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class LetterNN
    {
        /// <summary>
        /// Side length of input image
        /// </summary>
        public const int IMAGE_SIDE = 28;

        /// <summary>
        /// Offset to skip digits
        /// </summary>
        public const int MAP_OFFSET = 10;

        public static int[] MAPPING = new int[47]
        {
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
            75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
            85, 86, 87, 88, 89, 90, 97, 98, 100, 101,
            102, 103, 104, 110, 113, 114, 116
        };

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="nw">Digit neural network</param>
        /// <param name="layers">All specification of layer structure</param>
        public static void init(out NeuralNetwork<char> nw, int[] layers)
        {
            char[] meaning = new char[layers[layers.Length - 1]];
            for (int i = MAP_OFFSET; i < meaning.Length + MAP_OFFSET; i++)
            {
                meaning[i - MAP_OFFSET] = (char) MAPPING[i];
            }
            nw = new NeuralNetwork<char>(layers, meaning);
        }


        /// <summary>
        /// Transform bitmap data into double array
        /// </summary>
        /// <param name="data">Input bitmap</param>
        /// <returns>Float format result</returns>
        public static double[][] transformBitmapdata(Bitmap[] data)
        {
            double[][] result = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                result[i] = new double[IMAGE_SIDE * IMAGE_SIDE];
                // data bitmap size check handling isn't implemented
                for (int h = 0, index = 0; h < IMAGE_SIDE; h++)
                {
                    for (int w = 0; w < IMAGE_SIDE; w++)
                    {
                        Color c = data[i].GetPixel(w, h);
                        result[i][index++] = (c.R + c.G + c.B) / 3.0f / 255.0f;
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Transform bitmap data into double array
        /// </summary>
        /// <param name="data">Input bitmap</param>
        /// <returns>Float format result</returns>
        public static double[] transformBitmapdata(Bitmap data)
        {
            double[] result = new double[IMAGE_SIDE * IMAGE_SIDE];
            for (int h = 0, index = 0; h < IMAGE_SIDE; h++)
            {
                for (int w = 0; w < IMAGE_SIDE; w++)
                {
                    Color c = data.GetPixel(w, h);
                    result[index++] = (c.G + c.B) / 2.0f / 255.0f;
                }
            }
            return result;
        }

        /// <summary>
        /// Split data into training data and validset data
        /// </summary>
        /// <param name="data">input data</param>
        /// <param name="trainData">training data</param>
        /// <param name="validset">validset data</param>
        /// <param name="trainDataSize">train data size</param>
        public static void splitIntoTrainAndValidset(double[][] data, out double[][] trainData, out double[][] validset, int trainDataSize)
        {
            trainData = new double[trainDataSize][];
            validset = new double[data.Length - trainDataSize][];
            for (int i = 0; i < trainDataSize; i++)
            {
                trainData[i] = new double[data[i].Length];
                for (int j = 0; j < trainData[i].Length; j++)
                {
                    trainData[i][j] = data[i][j];
                }
            }
            for (int i = 0; i < validset.Length; i++)
            {
                validset[i] = new double[data[i].Length];
                for (int j = 0; j < validset[i].Length; j++)
                {
                    validset[i][j] = data[i + trainDataSize][j];
                }
            }
        }

        /// <summary>
        /// Split label into training label and validset labels
        /// </summary>
        /// <param name="labels">input labels</param>
        /// <param name="trainLabels">training labels</param>
        /// <param name="validsetLabels">validset labels</param>
        /// <param name="trainDataSize">train data size</param>
        public static void splitIntoTrainAndValidsetLabels(char[] labels, out char[] trainLabels, out char[] validsetLabels, int trainDataSize)
        {
            trainLabels = new char[trainDataSize];
            validsetLabels = new char[labels.Length - trainDataSize];
            for (int i = 0; i < trainDataSize; i++)
            {
                trainLabels[i] = labels[i];
            }
            for (int i = 0; i < validsetLabels.Length; i++)
            {
                validsetLabels[i] = labels[i + trainDataSize];
            }
        }
    }
}
